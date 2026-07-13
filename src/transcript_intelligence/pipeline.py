import asyncio
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

from transcript_intelligence.aggregation import aggregate_metrics
from transcript_intelligence.analytics import write_charts
from transcript_intelligence.classify import classify_transcripts
from transcript_intelligence.config import Settings
from transcript_intelligence.execution import Execution, StageStatus
from transcript_intelligence.ingest import ingest_transcripts
from transcript_intelligence.io_utils import read_jsonl, write_json, write_jsonl
from transcript_intelligence.logging_setup import get_logger
from transcript_intelligence.models import (
    CentroidSegmentRecord,
    Classification,
    ClusterMetadata,
    Finding,
    IngestedUtterance,
    Metric,
    PseudonymizedTranscript,
    RedactedUtterance,
    RedactionReport,
    ReviewItem,
    Segment,
    TopicAssignment,
    TopicLabel,
    TopicTerm,
    TranscriptRecord,
    Turn,
)
from transcript_intelligence.pii import PiiProcessor
from transcript_intelligence.remote_llm import extract_findings, label_topics
from transcript_intelligence.semantic import (
    cluster_segments,
    create_segments,
    embed_segments,
)
from transcript_intelligence.turns import materialize_turns

log = get_logger(__name__)


@dataclass
class PipelineDependencies:
    pii_processor: PiiProcessor
    embedding_model: SentenceTransformer
    llm_client: AsyncOpenAI


def _run_stage(execution: Execution, stage: str, work) -> None:
    if execution.is_complete(stage):
        log.info("skipping completed stage", stage=stage)
        return
    log.info("starting stage", stage=stage)
    execution.mark(stage, StageStatus.running)
    try:
        work()
    except Exception:
        execution.mark(stage, StageStatus.failed)
        log.exception("stage failed", stage=stage)
        raise
    execution.mark(stage, StageStatus.complete)
    log.info("stage finished", stage=stage)


def run_pipeline(
    settings: Settings,
    input_directory: Path,
    output_directory: Path,
    dependencies: PipelineDependencies,
) -> Execution:
    execution = Execution.allocate(output_directory)

    def ingest() -> None:
        ingest_transcripts(input_directory, execution.stage_dir("ingest"))

    _run_stage(execution, "ingest", ingest)
    transcripts = read_jsonl(
        execution.stage_dir("ingest") / "transcripts.jsonl",
        TranscriptRecord,
    )
    utterances = read_jsonl(
        execution.stage_dir("ingest") / "utterances.jsonl",
        IngestedUtterance,
    )

    def privacy() -> None:
        stage_dir = execution.stage_dir("privacy")
        mapping_dir = stage_dir / "pii_mappings"
        mapping_dir.mkdir(parents=True, exist_ok=True)
        by_transcript: dict[str, list[IngestedUtterance]] = defaultdict(list)
        for utterance in utterances:
            by_transcript[utterance.transcript_id].append(utterance)

        pseudonymized: list[PseudonymizedTranscript] = []
        reports: list[RedactionReport] = []
        redacted: list[RedactedUtterance] = []
        reviews: list[ReviewItem] = []
        for transcript in transcripts:
            result, report, redacted_utterances, mapping = (
                dependencies.pii_processor.process(
                    transcript,
                    sorted(
                        by_transcript[transcript.transcript_id],
                        key=lambda item: item.index,
                    ),
                )
            )
            pseudonymized.append(result)
            reports.append(report)
            redacted.extend(redacted_utterances)
            write_json(
                mapping_dir / f"{transcript.transcript_id}.json",
                mapping,
            )
            if report.review_required:
                reviews.append(
                    ReviewItem(
                        review_id=f"privacy:{transcript.transcript_id}",
                        review_type="privacy",
                        transcript_id=transcript.transcript_id,
                        reason="residual_or_conflict",
                    )
                )
        write_jsonl(
            stage_dir / "pseudonymized_transcripts.jsonl",
            pseudonymized,
        )
        write_jsonl(stage_dir / "redaction_reports.jsonl", reports)
        write_jsonl(stage_dir / "redacted_utterances.jsonl", redacted)
        write_jsonl(stage_dir / "review_queue.jsonl", reviews)
        log.info(
            "privacy redaction finished",
            transcripts=len(pseudonymized),
            review_flags=len(reviews),
        )

    _run_stage(execution, "privacy", privacy)
    redacted_utterances = read_jsonl(
        execution.stage_dir("privacy") / "redacted_utterances.jsonl",
        RedactedUtterance,
    )

    def classify() -> None:
        grouped: dict[str, list[RedactedUtterance]] = defaultdict(list)
        for utterance in redacted_utterances:
            grouped[utterance.transcript_id].append(utterance)
        for items in grouped.values():
            items.sort(key=lambda item: item.index)
        classifications = asyncio.run(
            classify_transcripts(
                dependencies.llm_client,
                settings,
                grouped,
            )
        )
        stage_dir = execution.stage_dir("classify")
        write_jsonl(stage_dir / "classifications.jsonl", classifications)
        write_jsonl(
            stage_dir / "review_queue.jsonl",
            [
                ReviewItem(
                    review_id=f"classify:{item.transcript_id}",
                    review_type="classify",
                    transcript_id=item.transcript_id,
                    reason=(
                        f"confidence {item.confidence:.2f} "
                        f"below {settings.classify_confidence_threshold}"
                    ),
                )
                for item in classifications
                if item.low_confidence
            ],
        )

    _run_stage(execution, "classify", classify)
    classifications = read_jsonl(
        execution.stage_dir("classify") / "classifications.jsonl",
        Classification,
    )

    def turns() -> None:
        write_jsonl(
            execution.stage_dir("turns") / "turns.jsonl",
            materialize_turns(redacted_utterances, classifications),
        )

    _run_stage(execution, "turns", turns)
    turns_data = read_jsonl(
        execution.stage_dir("turns") / "turns.jsonl",
        Turn,
    )

    def segments() -> None:
        write_jsonl(
            execution.stage_dir("segments") / "segments.jsonl",
            create_segments(
                turns_data,
                dependencies.embedding_model,
                settings,
            ),
        )

    _run_stage(execution, "segments", segments)
    segments_data = read_jsonl(
        execution.stage_dir("segments") / "segments.jsonl",
        Segment,
    )

    def embeddings() -> None:
        embed_segments(
            segments_data,
            dependencies.embedding_model,
            execution.stage_dir("embeddings"),
        )

    _run_stage(execution, "embeddings", embeddings)
    embedding_matrix = np.load(
        execution.stage_dir("embeddings") / "embeddings.npy"
    )

    def clustering() -> None:
        cluster_segments(
            segments_data,
            embedding_matrix,
            settings,
            execution.stage_dir("clustering"),
        )

    _run_stage(execution, "clustering", clustering)
    assignments = read_jsonl(
        execution.stage_dir("clustering") / "topic_assignments.jsonl",
        TopicAssignment,
    )
    metadata = read_jsonl(
        execution.stage_dir("clustering") / "cluster_metadata.jsonl",
        ClusterMetadata,
    )
    terms = read_jsonl(
        execution.stage_dir("topic_representation") / "topic_terms.jsonl",
        TopicTerm,
    )
    centroids = read_jsonl(
        execution.stage_dir("topic_representation")
        / "centroid_segments.jsonl",
        CentroidSegmentRecord,
    )

    def topic_labels() -> None:
        asyncio.run(
            label_topics(
                dependencies.llm_client,
                settings,
                metadata,
                terms,
                centroids,
                {segment.segment_id: segment for segment in segments_data},
                execution.stage_dir("topic_label"),
            )
        )

    _run_stage(execution, "topic_labels", topic_labels)
    labels = read_jsonl(
        execution.stage_dir("topic_label") / "topics.jsonl",
        TopicLabel,
    )

    def sentiment() -> None:
        asyncio.run(
            extract_findings(
                dependencies.llm_client,
                settings,
                segments_data,
                execution.stage_dir("sentiment"),
            )
        )

    _run_stage(execution, "sentiment", sentiment)
    findings = read_jsonl(
        execution.stage_dir("sentiment") / "findings.jsonl",
        Finding,
    )

    def aggregation() -> None:
        metrics, contributors = aggregate_metrics(
            transcripts,
            segments_data,
            assignments,
            labels,
            findings,
        )
        stage_dir = execution.stage_dir("aggregation")
        write_jsonl(stage_dir / "metrics.jsonl", metrics)
        write_jsonl(stage_dir / "metric_contributors.jsonl", contributors)
        log.info(
            "aggregation finished",
            metrics=len(metrics),
            contributors=len(contributors),
        )

    _run_stage(execution, "aggregation", aggregation)
    metrics = read_jsonl(
        execution.stage_dir("aggregation") / "metrics.jsonl",
        Metric,
    )

    def analytics() -> None:
        write_charts(metrics, execution.stage_dir("analytical"))

    _run_stage(execution, "analytics", analytics)
    execution.mark_complete()
    log.info("pipeline finished", path=str(execution.directory))
    return execution
