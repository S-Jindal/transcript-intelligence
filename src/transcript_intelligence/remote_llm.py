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
    EvidenceSpan,
    Finding,
    Segment,
    SourceSet,
    TopicLabel,
    TopicTerm,
)
from transcript_intelligence.quote_match import locate_quote

log = get_logger(__name__)


class TopicLabelResponse(BaseModel):
    label: str
    description: str


class FindingItem(BaseModel):
    finding_type: str
    target: str
    value: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    intensity: int | None = Field(default=None, ge=1, le=5)
    quote: str


class FindingsResponse(BaseModel):
    findings: list[FindingItem]


TOPIC_INSTRUCTIONS = """
We're performing topic modeling on a set of redacted transcripts with interactions between customers and account managers or sales staff.
Name the cluster topic using the ranked terms and example segments from the same cluster.
Return a concise business label and one-sentence description.
Do not invent details absent from the evidence.
""".strip()

FINDINGS_INSTRUCTIONS = """
Extract sentiment and business findings from the redacted segment of transcript with interactions between customers and account managers or sales staff.
Valence values must be positive or negative (treat uncertainty as negative).
Allowed finding_type values include:
sentiment, customer_effort, frustration, resolution, objection,
renewal_risk, feature_request, opportunity, commitment.
Every finding must include an exact quote copied from the segment text.
If nothing meaningful is present, return an empty findings list.
""".strip()


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
) -> tuple[list[Finding], list[EvidenceSpan]]:
    customer_segments = [
        segment
        for segment in segments
        if segment.source_set
        in {SourceSet.customer_support, SourceSet.account_manager}
    ]
    semaphore = asyncio.Semaphore(settings.llm_concurrency)

    async def extract_one(
        segment: Segment,
    ) -> tuple[list[Finding], list[EvidenceSpan]]:
        request_id = f"findings:{segment.segment_id}"
        async with semaphore:
            parsed = await _parse_with_retries(
                client,
                settings,
                FINDINGS_INSTRUCTIONS,
                segment.text,
                FindingsResponse,
                request_id,
            )
        assert isinstance(parsed, FindingsResponse)
        findings: list[Finding] = []
        evidence: list[EvidenceSpan] = []
        for index, item in enumerate(parsed.findings):
            match = locate_quote(
                segment.text,
                item.quote,
                settings.quote_max_edit_distance,
            )
            if match is None:
                log.warning(
                    "finding quote not found in segment, skipping",
                    segment_id=segment.segment_id,
                    finding_type=item.finding_type,
                )
                continue
            matched_quote = segment.text[match.start : match.end]
            finding_id = f"{segment.segment_id}:finding:{index}"
            evidence_id = f"{finding_id}:evidence"
            findings.append(
                Finding(
                    finding_id=finding_id,
                    request_id=request_id,
                    transcript_id=segment.transcript_id,
                    segment_id=segment.segment_id,
                    source_set=segment.source_set,
                    finding_type=item.finding_type,
                    target=item.target,
                    value=item.value,
                    reason=item.reason,
                    confidence=item.confidence,
                    intensity=item.intensity,
                    evidence_id=evidence_id,
                    model=settings.llm_model,
                    prompt_version=settings.findings_prompt_version,
                )
            )
            evidence.append(
                EvidenceSpan(
                    evidence_id=evidence_id,
                    finding_id=finding_id,
                    segment_id=segment.segment_id,
                    transcript_id=segment.transcript_id,
                    quote=matched_quote,
                    start=match.start,
                    end=match.end,
                    turn_ids=segment.turn_ids,
                )
            )
        return findings, evidence

    tasks = [extract_one(segment) for segment in customer_segments]
    all_findings: list[Finding] = []
    all_evidence: list[EvidenceSpan] = []
    done = 0
    for task in asyncio.as_completed(tasks):
        findings, evidence = await task
        all_findings.extend(findings)
        all_evidence.extend(evidence)
        done += 1
        if done % 20 == 0 or done == len(tasks):
            log.info(
                "findings progress",
                done=done,
                total=len(tasks),
            )
    write_jsonl(stage_dir / "findings.jsonl", all_findings)
    write_jsonl(stage_dir / "evidence.jsonl", all_evidence)
    log.info(
        "findings extraction finished",
        findings=len(all_findings),
        evidence=len(all_evidence),
    )
    return all_findings, all_evidence
