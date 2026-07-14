import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel

from transcript_intelligence.config import Settings
from transcript_intelligence.io_utils import write_json, write_jsonl
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


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )


def _token_usage(response) -> TokenUsage:
    usage = getattr(response, "usage", None)
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
    )

class TopicLabelResponse(BaseModel):
    label: str
    description: str


class FindingItem(BaseModel):
    finding_type: str
    target: str
    value: Literal["positive", "negative"]
    reason: str


class FindingsResponse(BaseModel):
    findings: list[FindingItem]


TOPIC_INSTRUCTIONS = """
We're performing topic modeling on a set of redacted transcripts with interactions between customers and account managers or support staff.
Name the cluster topic using the ranked terms and example segments from the same cluster.
Return a concise business label and one-sentence description.
Do not invent details absent from the evidence.
""".strip()

CUSTOMER_CANONICAL_FINDING_TYPES = (
    "sentiment",
    "process_friction",
    "resolution",
    "objection",
    "renewal_risk",
    "feature_request",
    "opportunity",
    "commitment",
    "competitive_risk",
)
INTERNAL_CANONICAL_FINDING_TYPES = (
    "sentiment",
    "operational_risk",
    "delivery_risk",
    "competitive_risk",
    "renewal_risk",
    "opportunity",
    "commitment",
    "capacity_constraint",
)

CUSTOMER_FINDINGS_INSTRUCTIONS = """
Extract sentiment and business findings from the redacted segment of a
customer-facing call (customer support or account manager).
If nothing meaningful is present, return an empty findings list.
Return at most 3 findings — only the most salient signals in the segment.

Each object must use these fields:

finding_type:
  Prefer one of these canonical types when it fits:
  sentiment, process_friction, resolution, objection, renewal_risk,
  feature_request, opportunity, commitment, competitive_risk.
  Use sentiment for any affective signal toward a concrete target,
  including praise, satisfaction, frustration, anger, or
  dissatisfaction.
  Use process_friction when the customer had to do excessive work to
  make progress (repeated retries, manual workarounds, long waits,
  re-explaining the issue, or ticket ping-pong).
  Use competitive_risk when a competitor product, vendor, or
  alternative is named or clearly implied as leverage, evaluation,
  or displacement pressure.
  Only if a genuinely distinct, recurring business signal fits none of
  the canonical types, you may introduce a new concise snake_case
  finding_type. Do not create near-duplicates or synonyms of the
  canonical types.

target:
  The concrete subject of the finding (product, feature, process,
  billing item, policy, competitor name, or speaker role such as
  customer or agent). Prefer role names over redacted person tokens
  like [PERSON_01]. Keep it short and reusable across calls.

value:
  Exactly "positive" or "negative" — the business polarity of the
  finding. Use positive for favorable signals (praise, resolution,
  expansion, wins) and negative for unfavorable ones (dissatisfaction,
  risk, friction, competitive pressure). Treat uncertainty, hesitation,
  or mixed tone as negative.

reason:
  One or two sentences explaining why this finding is warranted,
  grounded only in evidence present in the segment. Do not invent
  details that are not in the text.
""".strip()

INTERNAL_FINDINGS_INSTRUCTIONS = """
Extract sentiment and business findings from the redacted segment of an
internal discussion (engineering, product, support ops, or leadership).
If nothing meaningful is present, return an empty findings list.
Return at most 3 findings — only the most salient signals in the segment.

Each object must use these fields:

finding_type:
  Prefer one of these canonical types when it fits:
  sentiment, operational_risk, delivery_risk, competitive_risk,
  renewal_risk, opportunity, commitment, capacity_constraint.
  Use sentiment for team tone toward a product, incident, roadmap item,
  or process (including frustration or confidence).
  Use operational_risk for reliability, outages, monitoring gaps, or
  customer-impacting incidents discussed internally.
  Use delivery_risk for slip risk on launches, fixes, or milestones.
  Use capacity_constraint when staffing, bandwidth, or prioritization
  is blocking work.
  Only if a genuinely distinct, recurring business signal fits none of
  the canonical types, you may introduce a new concise snake_case
  finding_type. Do not create near-duplicates or synonyms of the
  canonical types.

target:
  The concrete subject (product, service, incident, account theme,
  competitor, or team process). Prefer role or team labels over
  redacted person tokens like [PERSON_01]. Keep it short.

value:
  Exactly "positive" or "negative" — the business polarity of the
  finding. Use positive for favorable signals (confidence, wins,
  expansion bets) and negative for unfavorable ones (risk, slippage,
  outages, competitive pressure). Treat uncertainty or mixed tone as
  negative.

reason:
  One or two sentences explaining why this finding is warranted,
  grounded only in evidence present in the segment. Do not invent
  details that are not in the text.
""".strip()


def _findings_instructions(source_set: SourceSet) -> str:
    if source_set == SourceSet.internal_discuss:
        return INTERNAL_FINDINGS_INSTRUCTIONS
    return CUSTOMER_FINDINGS_INSTRUCTIONS


def _canonical_finding_types(source_set: SourceSet) -> frozenset[str]:
    if source_set == SourceSet.internal_discuss:
        return frozenset(INTERNAL_CANONICAL_FINDING_TYPES)
    return frozenset(CUSTOMER_CANONICAL_FINDING_TYPES)



async def _parse_with_retries(
    client: AsyncOpenAI,
    settings: Settings,
    instructions: str,
    input_text: str,
    text_format: type[BaseModel],
    request_id: str,
) -> tuple[BaseModel, TokenUsage]:
    usage_total = TokenUsage()
    for attempt in range(settings.llm_maximum_attempts):
        try:
            response = await client.responses.parse(
                model=settings.llm_model,
                instructions=instructions,
                input=input_text,
                text_format=text_format,
            )
            usage_total = usage_total + _token_usage(response)
            parsed = response.output_parsed
            if parsed is None:
                raise ValueError("empty parsed response")
            return parsed, usage_total
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

    async def label_one(
        cluster: ClusterMetadata,
    ) -> tuple[TopicLabel, TokenUsage]:
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
            parsed, usage = await _parse_with_retries(
                client,
                settings,
                TOPIC_INSTRUCTIONS,
                prompt,
                TopicLabelResponse,
                cluster.topic_id,
            )
        assert isinstance(parsed, TopicLabelResponse)
        return (
            TopicLabel(
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
            ),
            usage,
        )

    tasks = [label_one(cluster) for cluster in topics]
    labels: list[TopicLabel] = []
    usage_total = TokenUsage()
    done = 0
    for task in asyncio.as_completed(tasks):
        label, usage = await task
        labels.append(label)
        usage_total = usage_total + usage
        done += 1
        log.info("topic labeling progress", done=done, total=len(tasks))
    write_jsonl(stage_dir / "topics.jsonl", labels)
    log.info(
        "topic labeling finished",
        topics=len(labels),
        input_tokens=usage_total.input_tokens,
        output_tokens=usage_total.output_tokens,
    )
    return labels


async def extract_findings(
    client: AsyncOpenAI,
    settings: Settings,
    segments: list[Segment],
    stage_dir: Path,
) -> list[Finding]:
    semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def extract_one(
        segment: Segment,
    ) -> tuple[list[Finding], TokenUsage]:
        request_id = f"findings:{segment.segment_id}"
        async with semaphore:
            parsed, usage = await _parse_with_retries(
                client,
                settings,
                _findings_instructions(segment.source_set),
                segment.text,
                FindingsResponse,
                request_id,
            )
        assert isinstance(parsed, FindingsResponse)
        salient = parsed.findings[: settings.findings_per_segment_maximum]
        return (
            [
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
                    model=settings.llm_model,
                    prompt_version=settings.findings_prompt_version,
                )
                for index, item in enumerate(salient)
            ],
            usage,
        )

    def _is_proposal(finding: Finding) -> bool:
        return finding.finding_type not in _canonical_finding_types(
            finding.source_set
        )

    tasks = [extract_one(segment) for segment in segments]
    all_findings: list[Finding] = []
    usage_total = TokenUsage()
    proposal_progress: dict[str, int] = defaultdict(int)
    done = 0

    for task in asyncio.as_completed(tasks):
        findings, usage = await task
        all_findings.extend(findings)
        usage_total = usage_total + usage
        for finding in findings:
            if _is_proposal(finding):
                proposal_progress[finding.finding_type] += 1
        done += 1
        if done % 20 == 0 or done == len(tasks):
            log.info(
                "findings progress",
                done=done,
                total=len(tasks),
                proposed_types=len(proposal_progress),
                proposals=dict(proposal_progress),
            )

    proposals: dict[str, list[Finding]] = defaultdict(list)
    for finding in all_findings:
        if _is_proposal(finding):
            proposals[finding.finding_type].append(finding)

    def _segment_count(items: list[Finding]) -> int:
        return len({item.segment_id for item in items})

    promoted = {
        proposed_type
        for proposed_type, items in proposals.items()
        if _segment_count(items) >= settings.finding_proposal_promotion_minimum
    }
    kept_findings = [
        finding
        for finding in all_findings
        if finding.finding_type not in proposals
        or finding.finding_type in promoted
    ]

    proposal_report = [
        {
            "proposed_type": proposed_type,
            "finding_count": len(items),
            "segment_count": _segment_count(items),
            "promoted": proposed_type in promoted,
            "segment_ids": sorted({item.segment_id for item in items}),
            "finding_ids": [item.finding_id for item in items],
        }
        for proposed_type, items in sorted(
            proposals.items(),
            key=lambda entry: _segment_count(entry[1]),
            reverse=True,
        )
    ]
    write_json(stage_dir / "finding_type_proposals.json", proposal_report)
    write_jsonl(stage_dir / "findings.jsonl", kept_findings)
    log.info(
        "findings extraction finished",
        findings=len(kept_findings),
        dropped_proposals=len(all_findings) - len(kept_findings),
        proposed_types=len(proposals),
        promoted_types=sorted(promoted),
        input_tokens=usage_total.input_tokens,
        output_tokens=usage_total.output_tokens,
    )
    return kept_findings
