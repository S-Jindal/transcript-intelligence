from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceSet(StrEnum):
    customer_support = "customer-support"
    account_manager = "account-manager"
    internal_discuss = "internal-discuss"


class RawUtterance(StrictModel):
    index: int
    speaker_id: int
    speaker_name: str
    sentence: str
    time: float | None = None
    end_time: float | None = Field(default=None, alias="endTime")
    average_confidence: float | None = Field(
        default=None,
        alias="averageConfidence",
    )
    sentiment_type: str | None = Field(default=None, alias="sentimentType")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class TranscriptRecord(StrictModel):
    transcript_id: str
    relative_path: str
    transcript_datetime: datetime
    end_time: datetime | None = None
    duration_minutes: float | None = None
    utterance_count: int
    content_hash: str
    size_bytes: int
    ingestion_status: Literal["complete"] = "complete"


class IngestedUtterance(StrictModel):
    transcript_id: str
    index: int
    speaker_id: int
    speaker_name: str
    sentence: str
    time: float | None = None
    end_time: float | None = None


class EntitySpan(StrictModel):
    start: int
    end: int
    entity_type: str
    value: str
    score: float = Field(ge=0, le=1)
    detector: str
    action: Literal["redact", "preserve"] = "redact"
    conflict: bool = False


class RedactedUtterance(StrictModel):
    transcript_id: str
    index: int
    speaker_id: int
    speaker_name: str
    sentence: str
    time: float | None = None
    end_time: float | None = None


class PseudonymizedTranscript(StrictModel):
    transcript_id: str
    transcript_datetime: datetime
    replacement_count: int
    unique_replacement_count: int
    conflict_count: int
    redacted_character_ratio: float
    residual_findings: list[EntitySpan] = Field(default_factory=list)
    review_required: bool = False


class RedactionReport(StrictModel):
    transcript_id: str
    category_counts: dict[str, int]
    unique_category_counts: dict[str, int]
    replacement_count: int
    conflict_count: int
    redacted_character_ratio: float
    residual_category_counts: dict[str, int]
    review_required: bool


class ReviewItem(StrictModel):
    review_id: str
    review_type: Literal["privacy", "classify"]
    transcript_id: str
    reason: str
    status: Literal["pending", "resolved"] = "pending"


class Classification(StrictModel):
    transcript_id: str
    source_set: SourceSet
    confidence: float = Field(ge=0, le=1)
    rationale: str
    low_confidence: bool = False


class Turn(StrictModel):
    turn_id: str
    transcript_id: str
    source_set: SourceSet
    order: int
    speaker: str
    text: str
    time: float | None = None
    end_time: float | None = None


class Segment(StrictModel):
    segment_id: str
    transcript_id: str
    source_set: SourceSet
    order: int
    first_turn_id: str
    last_turn_id: str
    turn_ids: list[str]
    speakers: list[str]
    text: str
    token_count: int


class EmbeddingIndex(StrictModel):
    row: int
    segment_id: str
    transcript_id: str


class TopicAssignment(StrictModel):
    topic_version: str
    topic_id: str
    cluster_id: int
    segment_id: str
    transcript_id: str
    source_set: SourceSet
    is_outlier: bool = False


class ClusterMetadata(StrictModel):
    topic_version: str
    topic_id: str
    cluster_id: int
    cluster_size: int
    source_distribution: dict[str, int]
    is_outlier: bool


class TopicTerm(StrictModel):
    topic_version: str
    topic_id: str
    cluster_id: int
    rank: int
    term: str


class CentroidSegmentRecord(StrictModel):
    topic_version: str
    topic_id: str
    cluster_id: int
    rank: int
    segment_id: str


class TopicLabel(StrictModel):
    topic_version: str
    topic_id: str
    cluster_id: int
    label: str
    description: str
    terms: list[str]
    centroid_segment_ids: list[str]
    cluster_size: int
    source_distribution: dict[str, int]
    model: str
    prompt_version: str


class Finding(StrictModel):
    finding_id: str
    request_id: str
    transcript_id: str
    segment_id: str
    source_set: SourceSet
    finding_type: str
    target: str
    value: str
    reason: str
    confidence: float = Field(ge=0, le=1)
    intensity: int | None = Field(default=None, ge=1, le=5)
    model: str
    prompt_version: str


class Metric(StrictModel):
    metric_id: str
    chart_point_id: str
    metric_type: str
    source_set: str
    time_window: str
    category: str
    numerator: int
    denominator: int
    value: float
    filters: dict[str, Any] = Field(default_factory=dict)


class MetricContributor(StrictModel):
    chart_point_id: str
    membership_role: Literal["numerator", "denominator_only", "excluded"]
    exclusion_reason: str | None = None
    finding_id: str | None = None
    topic_id: str | None = None
    segment_id: str
    transcript_id: str
