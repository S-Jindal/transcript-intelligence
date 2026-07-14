from collections import defaultdict
from datetime import datetime

from transcript_intelligence.models import (
    Finding,
    Metric,
    MetricContributor,
    Segment,
    SourceSet,
    TopicAssignment,
    TopicLabel,
    TranscriptRecord,
)


def _month_key(value: datetime) -> str:
    return value.strftime("%Y-%m")


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _bucket_polarity(members: list[Finding]) -> str:
    positives = sum(1 for finding in members if finding.value == "positive")
    negatives = len(members) - positives
    if positives > negatives:
        return "positive"
    if negatives > positives:
        return "negative"
    return "neutral"


def aggregate_metrics(
    transcripts: list[TranscriptRecord],
    segments: list[Segment],
    assignments: list[TopicAssignment],
    labels: list[TopicLabel],
    findings: list[Finding],
) -> tuple[list[Metric], list[MetricContributor]]:
    datetime_by_transcript = {
        item.transcript_id: item.transcript_datetime for item in transcripts
    }
    label_by_topic = {item.topic_id: item.label for item in labels}
    metrics: list[Metric] = []
    contributors: list[MetricContributor] = []

    non_outlier = [
        item for item in assignments if not item.is_outlier
    ]
    by_source_month: dict[tuple[str, str], list[TopicAssignment]] = defaultdict(
        list
    )
    by_source: dict[str, list[TopicAssignment]] = defaultdict(list)
    for assignment in non_outlier:
        source = assignment.source_set.value
        month = _month_key(datetime_by_transcript[assignment.transcript_id])
        by_source_month[(source, month)].append(assignment)
        by_source[source].append(assignment)

    for (source, month), rows in sorted(by_source_month.items()):
        denominator = len(rows)
        for topic_id in sorted({row.topic_id for row in rows}):
            members = [row for row in rows if row.topic_id == topic_id]
            chart_point_id = f"topic_month|{source}|{month}|{topic_id}"
            metrics.append(
                Metric(
                    metric_id=chart_point_id,
                    chart_point_id=chart_point_id,
                    metric_type="topic_prevalence_monthly",
                    source_set=source,
                    time_window=month,
                    category=label_by_topic.get(topic_id, topic_id),
                    numerator=len(members),
                    denominator=denominator,
                    value=_rate(len(members), denominator),
                    distinct_transcripts=len(
                        {row.transcript_id for row in members}
                    ),
                )
            )
            contributors.extend(
                MetricContributor(
                    chart_point_id=chart_point_id,
                    membership_role="numerator",
                    topic_id=row.topic_id,
                    segment_id=row.segment_id,
                    transcript_id=row.transcript_id,
                )
                for row in members
            )

    for source, rows in sorted(by_source.items()):
        denominator = len(rows)
        for topic_id in sorted({row.topic_id for row in rows}):
            members = [row for row in rows if row.topic_id == topic_id]
            chart_point_id = f"topic_all|{source}|{topic_id}"
            metrics.append(
                Metric(
                    metric_id=chart_point_id,
                    chart_point_id=chart_point_id,
                    metric_type="topic_prevalence_all_time",
                    source_set=source,
                    time_window="all",
                    category=label_by_topic.get(topic_id, topic_id),
                    numerator=len(members),
                    denominator=denominator,
                    value=_rate(len(members), denominator),
                    distinct_transcripts=len(
                        {row.transcript_id for row in members}
                    ),
                )
            )
            contributors.extend(
                MetricContributor(
                    chart_point_id=chart_point_id,
                    membership_role="numerator",
                    topic_id=row.topic_id,
                    segment_id=row.segment_id,
                    transcript_id=row.transcript_id,
                )
                for row in members
            )

    finding_segments = [
        segment
        for segment in segments
        if segment.source_set
        in {
            SourceSet.customer_support,
            SourceSet.account_manager,
            SourceSet.internal_discuss,
        }
    ]
    findings_by_segment = defaultdict(list)
    for finding in findings:
        findings_by_segment[finding.segment_id].append(finding)

    for source in (
        SourceSet.customer_support.value,
        SourceSet.account_manager.value,
        SourceSet.internal_discuss.value,
    ):
        source_segments = [
            segment
            for segment in finding_segments
            if segment.source_set.value == source
        ]
        by_month: dict[str, list[Segment]] = defaultdict(list)
        for segment in source_segments:
            by_month[
                _month_key(datetime_by_transcript[segment.transcript_id])
            ].append(segment)

        for month, month_segments in sorted(by_month.items()):
            denominator = len(month_segments)
            sentiment_rows = [
                finding
                for segment in month_segments
                for finding in findings_by_segment[segment.segment_id]
                if finding.finding_type == "sentiment"
            ]
            categories = sorted(
                {
                    f"{finding.value}:{finding.target}"
                    for finding in sentiment_rows
                }
            )
            for category in categories:
                members = [
                    finding
                    for finding in sentiment_rows
                    if f"{finding.value}:{finding.target}" == category
                ]
                chart_point_id = (
                    f"sentiment|{source}|{month}|{category}"
                )
                metrics.append(
                    Metric(
                        metric_id=chart_point_id,
                        chart_point_id=chart_point_id,
                        metric_type="sentiment_distribution",
                        source_set=source,
                        time_window=month,
                        category=category,
                        numerator=len(members),
                        denominator=denominator,
                        value=_rate(len(members), denominator),
                        distinct_transcripts=len(
                            {item.transcript_id for item in members}
                        ),
                        polarity=_bucket_polarity(members),
                    )
                )
                contributors.extend(
                    MetricContributor(
                        chart_point_id=chart_point_id,
                        membership_role="numerator",
                        finding_id=item.finding_id,
                        segment_id=item.segment_id,
                        transcript_id=item.transcript_id,
                    )
                    for item in members
                )

            finding_categories = sorted(
                {
                    finding.finding_type
                    for segment in month_segments
                    for finding in findings_by_segment[segment.segment_id]
                    if finding.finding_type != "sentiment"
                }
            )
            for category in finding_categories:
                members = [
                    finding
                    for segment in month_segments
                    for finding in findings_by_segment[segment.segment_id]
                    if finding.finding_type == category
                ]
                chart_point_id = f"finding|{source}|{month}|{category}"
                metrics.append(
                    Metric(
                        metric_id=chart_point_id,
                        chart_point_id=chart_point_id,
                        metric_type="finding_prevalence",
                        source_set=source,
                        time_window=month,
                        category=category,
                        numerator=len(members),
                        denominator=denominator,
                        value=_rate(len(members), denominator),
                        distinct_transcripts=len(
                            {item.transcript_id for item in members}
                        ),
                        polarity=_bucket_polarity(members),
                    )
                )
                contributors.extend(
                    MetricContributor(
                        chart_point_id=chart_point_id,
                        membership_role="numerator",
                        finding_id=item.finding_id,
                        segment_id=item.segment_id,
                        transcript_id=item.transcript_id,
                    )
                    for item in members
                )

    return metrics, contributors
