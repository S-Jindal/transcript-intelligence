import asyncio
from pathlib import Path

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from transcript_intelligence.config import Settings
from transcript_intelligence.io_utils import write_jsonl
from transcript_intelligence.logging_setup import get_logger
from transcript_intelligence.models import (
    CentroidSegmentRecord,
    ClusterMetadata,
    Finding,
    Segment,
    SourceSet,
    TopicLabel,
    TopicTerm,
)

log = get_logger(__name__)


class TopicLabelResponse(BaseModel):
    label: str
    description: str


class FindingItem(BaseModel):
    finding_type: str
    target: str
    value: str
    reason: str
    intensity: int | None = Field(default=None, ge=1, le=5)


class FindingsResponse(BaseModel):
    findings: list[FindingItem]


TOPIC_INSTRUCTIONS = """
We're performing topic modeling on a set of redacted transcripts with interactions between customers and account managers or support staff.
Name the cluster topic using the ranked terms and example segments from the same cluster.
Return a concise business label and one-sentence description.
Do not invent details absent from the evidence.
""".strip()

CUSTOMER_FINDINGS_INSTRUCTIONS = """
Extract sentiment and business findings from the redacted segment of a
customer-facing call (customer support or account manager).
If nothing meaningful is present, return an empty findings list.

Return zero or more objects. Each object must use these fields:

finding_type:
  One of: sentiment, process_friction, resolution, objection,
  renewal_risk, feature_request, opportunity, commitment,
  competitive_risk.
  Use sentiment for any affective signal toward a concrete target,
  including praise, satisfaction, frustration, anger, or
  dissatisfaction. 
  Use process_friction when the customer had to do excessive work to
  make progress (repeated retries, manual workarounds, long waits,
  re-explaining the issue, or ticket ping-pong). This is about burden
  of process, not mood alone — pair with sentiment when both apply.
  Use competitive_risk when a competitor product, vendor, or
  alternative is named or clearly implied as leverage, evaluation,
  or displacement pressure.
  Use the remaining types only for the matching business signal.

target:
  The concrete subject of the finding (product, feature, process,
  billing item, policy, competitor name, or speaker role such as
  customer or agent). Prefer role names over redacted person tokens
  like [PERSON_01]. Keep it short and reusable across calls.

value:
  For finding_type=sentiment: exactly "positive" or "negative"
  (treat uncertainty, hesitation, mixed tone, or frustration as
  negative).
  For competitive_risk: a concise phrase naming the competitor and
  the pressure.
  For all other finding_type values: a concise phrase stating the
  finding itself, not a full-sentence paraphrase of the segment.

reason:
  One or two sentences explaining why this finding is warranted,
  grounded only in evidence present in the segment. Do not invent
  details that are not in the text.

intensity:
  Optional integer from 1 (mild) to 5 (severe). Prefer including it
  for negative sentiment, renewal_risk, process_friction, and
  competitive_risk. Omit it when intensity is not meaningful.
""".strip()

INTERNAL_FINDINGS_INSTRUCTIONS = """
Extract sentiment and business findings from the redacted segment of an
internal discussion (engineering, product, support ops, or leadership).
If nothing meaningful is present, return an empty findings list.

Return zero or more objects. Each object must use these fields:

finding_type:
  One of: sentiment, operational_risk, delivery_risk, competitive_risk,
  renewal_risk, opportunity, commitment, capacity_constraint.
  Use sentiment for team tone toward a product, incident, roadmap item,
  or process (including frustration or confidence).
  Use operational_risk for reliability, outages, monitoring gaps, or
  customer-impacting incidents discussed internally.
  Use delivery_risk for slip risk on launches, fixes, or milestones.
  Use competitive_risk when a competitor is named as win/loss pressure
  or product gap motivation.
  Use renewal_risk when an at-risk account or renewal is discussed.
  Use opportunity for expansion, packaging, or product bets.
  Use commitment for explicit internal ownership or follow-ups.
  Use capacity_constraint when staffing, bandwidth, or prioritization
  is blocking work.

target:
  The concrete subject (product, service, incident, account theme,
  competitor, or team process). Prefer role or team labels over
  redacted person tokens like [PERSON_01]. Keep it short.

value:
  For finding_type=sentiment: exactly "positive" or "negative"
  (treat uncertainty or mixed tone as negative).
  For competitive_risk: a concise phrase naming the competitor and
  the pressure.
  For all other finding_type values: a concise phrase stating the
  finding itself, not a full-sentence paraphrase of the segment.

reason:
  One or two sentences explaining why this finding is warranted,
  grounded only in evidence present in the segment. Do not invent
  details that are not in the text.

intensity:
  Optional integer from 1 (mild) to 5 (severe). Prefer including it
  for negative sentiment, operational_risk, delivery_risk,
  renewal_risk, and competitive_risk. Omit it when intensity is not
  meaningful.
""".strip()


def _findings_instructions(source_set: SourceSet) -> str:
    if source_set == SourceSet.internal_discuss:
        return INTERNAL_FINDINGS_INSTRUCTIONS
    return CUSTOMER_FINDINGS_INSTRUCTIONS



async def _parse_with_retries(
    client: AsyncOpenAI,
    settings: Settings,
    instructions: str,
    input_text: str,
    text_format: type[BaseModel],
    request_id: str,
) -> BaseModel:
    for attempt in range(settings.llm_maximum_attempts):
        try:
            response = await client.responses.parse(
                model=settings.llm_model,
                instructions=instructions,
                input=input_text,
                text_format=text_format,
            )
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("empty parsed response")
            return parsed
        except Exception as error:
            if attempt + 1 >= settings.llm_maximum_attempts:
                raise
            delay = settings.llm_initial_backoff_seconds * (2**attempt)
            log.warning(
                "LLM call failed, retrying",
                request_id=request_id,
                attempt=attempt + 1,
                error=str(error),
                delay=delay,
            )
            await asyncio.sleep(delay)
    raise RuntimeError(f"llm failed for {request_id}")


async def label_topics(
    client: AsyncOpenAI,
    settings: Settings,
    metadata: list[ClusterMetadata],
    terms: list[TopicTerm],
    centroids: list[CentroidSegmentRecord],
    segments_by_id: dict[str, Segment],
    stage_dir: Path,
) -> list[TopicLabel]:
    semaphore = asyncio.Semaphore(settings.llm_concurrency)
    topics = [item for item in metadata if not item.is_outlier]

    async def label_one(cluster: ClusterMetadata) -> TopicLabel:
        topic_terms = [
            term.term
            for term in terms
            if term.topic_id == cluster.topic_id
        ]
        example_ids = [
            item.segment_id
            for item in centroids
            if item.topic_id == cluster.topic_id
        ]
        examples = "\n\n".join(
            segments_by_id[segment_id].text
            for segment_id in example_ids
            if segment_id in segments_by_id
        )
        prompt = (
            f"cluster size={cluster.cluster_size}\n"
            f"terms={topic_terms}\n"
            f"example segments:\n{examples}"
        )
        async with semaphore:
            parsed = await _parse_with_retries(
                client,
                settings,
                TOPIC_INSTRUCTIONS,
                prompt,
                TopicLabelResponse,
                cluster.topic_id,
            )
        assert isinstance(parsed, TopicLabelResponse)
        return TopicLabel(
            topic_version=cluster.topic_version,
            topic_id=cluster.topic_id,
            cluster_id=cluster.cluster_id,
            label=parsed.label,
            description=parsed.description,
            terms=topic_terms,
            centroid_segment_ids=example_ids,
            cluster_size=cluster.cluster_size,
            source_distribution=cluster.source_distribution,
            model=settings.llm_model,
            prompt_version=settings.topic_prompt_version,
        )

    tasks = [label_one(cluster) for cluster in topics]
    labels: list[TopicLabel] = []
    done = 0
    for task in asyncio.as_completed(tasks):
        labels.append(await task)
        done += 1
        log.info("topic labeling progress", done=done, total=len(tasks))
    write_jsonl(stage_dir / "topics.jsonl", labels)
    log.info("topic labeling finished", topics=len(labels))
    return labels


async def extract_findings(
    client: AsyncOpenAI,
    settings: Settings,
    segments: list[Segment],
    stage_dir: Path,
) -> list[Finding]:
    semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def extract_one(segment: Segment) -> list[Finding]:
        request_id = f"findings:{segment.segment_id}"
        async with semaphore:
            parsed = await _parse_with_retries(
                client,
                settings,
                _findings_instructions(segment.source_set),
                segment.text,
                FindingsResponse,
                request_id,
            )
        assert isinstance(parsed, FindingsResponse)
        return [
            Finding(
                finding_id=f"{segment.segment_id}:finding:{index}",
                request_id=request_id,
                transcript_id=segment.transcript_id,
                segment_id=segment.segment_id,
                source_set=segment.source_set,
                finding_type=item.finding_type,
                target=item.target,
                value=item.value,
                reason=item.reason,
                intensity=item.intensity,
                model=settings.llm_model,
                prompt_version=settings.findings_prompt_version,
            )
            for index, item in enumerate(parsed.findings)
        ]

    tasks = [extract_one(segment) for segment in segments]
    all_findings: list[Finding] = []
    done = 0
    for task in asyncio.as_completed(tasks):
        all_findings.extend(await task)
        done += 1
        if done % 20 == 0 or done == len(tasks):
            log.info(
                "findings progress",
                done=done,
                total=len(tasks),
            )
    write_jsonl(stage_dir / "findings.jsonl", all_findings)
    log.info("findings extraction finished", findings=len(all_findings))
    return all_findings
