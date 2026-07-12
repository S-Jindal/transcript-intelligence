from collections import defaultdict
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from transcript_intelligence.config import Settings
from transcript_intelligence.io_utils import write_json, write_jsonl
from transcript_intelligence.logging_setup import get_logger
from transcript_intelligence.models import (
    CentroidSegmentRecord,
    ClusterMetadata,
    EmbeddingIndex,
    Segment,
    SourceSet,
    TopicAssignment,
    TopicTerm,
    Turn,
)

log = get_logger(__name__)


def load_embedding_model(model_name: str, device: str) -> SentenceTransformer:
    log.info("loading_embedding_model", model=model_name, device=device)
    return SentenceTransformer(model_name, device=device)


def build_topic_model(minimum_cluster_size: int, random_state: int):
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    return BERTopic(
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=random_state,
        ),
        hdbscan_model=HDBSCAN(
            min_cluster_size=minimum_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        embedding_model=None,
        calculate_probabilities=False,
        verbose=False,
    )


def _token_count(model: SentenceTransformer, text: str) -> int:
    return len(model.tokenizer.encode(text))


def create_segments(
    turns: list[Turn],
    embedding_model: SentenceTransformer,
    settings: Settings,
) -> list[Segment]:
    grouped: dict[str, list[Turn]] = defaultdict(list)
    for turn in turns:
        grouped[turn.transcript_id].append(turn)

    segments: list[Segment] = []
    for transcript_id, transcript_turns in sorted(grouped.items()):
        ordered = sorted(transcript_turns, key=lambda item: item.order)
        embeddings = embedding_model.encode(
            [turn.text for turn in ordered],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        current: list[Turn] = []
        for index, turn in enumerate(ordered):
            candidate = current + [turn]
            candidate_text = "\n".join(
                f"{item.speaker}: {item.text}" for item in candidate
            )
            over_limit = (
                _token_count(embedding_model, candidate_text)
                > settings.maximum_segment_tokens
            )
            topic_shift = False
            if current and index > 0:
                similarity = float(
                    np.dot(embeddings[index - 1], embeddings[index])
                )
                topic_shift = (
                    similarity < settings.semantic_similarity_threshold
                )
            if current and (over_limit or topic_shift):
                segments.append(
                    _build_segment(transcript_id, segments, current)
                )
                current = [turn]
            else:
                current = candidate
        if current:
            segments.append(_build_segment(transcript_id, segments, current))

    log.info("segments_complete", segments=len(segments))
    return segments


def _build_segment(
    transcript_id: str,
    existing: list[Segment],
    turns: list[Turn],
) -> Segment:
    order = sum(item.transcript_id == transcript_id for item in existing)
    text = "\n".join(f"{turn.speaker}: {turn.text}" for turn in turns)
    return Segment(
        segment_id=f"{transcript_id}:seg:{order}",
        transcript_id=transcript_id,
        source_set=turns[0].source_set,
        order=order,
        first_turn_id=turns[0].turn_id,
        last_turn_id=turns[-1].turn_id,
        turn_ids=[turn.turn_id for turn in turns],
        speakers=list(dict.fromkeys(turn.speaker for turn in turns)),
        text=text,
        token_count=len(text.split()),
    )


def embed_segments(
    segments: list[Segment],
    embedding_model: SentenceTransformer,
    stage_dir: Path,
) -> np.ndarray:
    matrix = embedding_model.encode(
        [segment.text for segment in segments],
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    np.save(stage_dir / "embeddings.npy", matrix)
    write_jsonl(
        stage_dir / "segment_index.jsonl",
        [
            EmbeddingIndex(
                row=index,
                segment_id=segment.segment_id,
                transcript_id=segment.transcript_id,
            )
            for index, segment in enumerate(segments)
        ],
    )
    log.info("embeddings_complete", rows=len(segments))
    return matrix


def _fit_scope(
    scope_name: str,
    topic_version: str,
    scope_segments: list[Segment],
    embeddings: np.ndarray,
    segment_rows: dict[str, int],
    settings: Settings,
    stage_dir: Path,
) -> tuple[
    list[TopicAssignment],
    list[ClusterMetadata],
    list[TopicTerm],
    list[CentroidSegmentRecord],
    dict[int, list[str]],
]:
    if len(scope_segments) < settings.minimum_cluster_size:
        log.warning(
            "clustering_skipped_small_scope",
            scope=scope_name,
            segments=len(scope_segments),
        )
        return [], [], [], [], {}

    matrix = np.vstack(
        [embeddings[segment_rows[segment.segment_id]] for segment in scope_segments]
    )
    model = build_topic_model(
        settings.minimum_cluster_size,
        settings.topic_random_state,
    )
    topics, _ = model.fit_transform(
        [segment.text for segment in scope_segments],
        embeddings=matrix,
    )

    assignments = [
        TopicAssignment(
            topic_version=topic_version,
            topic_id=(
                f"{topic_version}:outlier"
                if topic == -1
                else f"{topic_version}:{topic}"
            ),
            cluster_id=int(topic),
            segment_id=segment.segment_id,
            transcript_id=segment.transcript_id,
            source_set=segment.source_set,
            is_outlier=topic == -1,
        )
        for segment, topic in zip(scope_segments, topics)
    ]

    topic_info = model.get_topic_info()
    metadata: list[ClusterMetadata] = []
    terms: list[TopicTerm] = []
    centroids: list[CentroidSegmentRecord] = []
    examples: dict[int, list[str]] = {}

    for _, row in topic_info.iterrows():
        cluster_id = int(row["Topic"])
        members = [
            assignment
            for assignment in assignments
            if assignment.cluster_id == cluster_id
        ]
        source_distribution = {
            source.value: sum(
                member.source_set == source for member in members
            )
            for source in SourceSet
            if any(member.source_set == source for member in members)
        }
        topic_id = (
            f"{topic_version}:outlier"
            if cluster_id == -1
            else f"{topic_version}:{cluster_id}"
        )
        metadata.append(
            ClusterMetadata(
                topic_version=topic_version,
                topic_id=topic_id,
                cluster_id=cluster_id,
                cluster_size=len(members),
                source_distribution=source_distribution,
                is_outlier=cluster_id == -1,
            )
        )
        if cluster_id == -1:
            continue

        topic_terms = model.get_topic(cluster_id) or []
        terms.extend(
            TopicTerm(
                topic_version=topic_version,
                topic_id=topic_id,
                cluster_id=cluster_id,
                rank=rank,
                term=term,
            )
            for rank, (term, _) in enumerate(topic_terms[:5], start=1)
        )

        member_vectors = np.vstack(
            [
                embeddings[segment_rows[member.segment_id]]
                for member in members
            ]
        )
        centroid = member_vectors.mean(axis=0)
        centroid = centroid / max(np.linalg.norm(centroid), 1e-12)
        ranked = sorted(
            zip(members, member_vectors),
            key=lambda item: float(np.dot(item[1], centroid)),
            reverse=True,
        )[: settings.centroid_segment_count]
        examples[cluster_id] = [member.segment_id for member, _ in ranked]
        centroids.extend(
            CentroidSegmentRecord(
                topic_version=topic_version,
                topic_id=topic_id,
                cluster_id=cluster_id,
                rank=rank,
                segment_id=member.segment_id,
            )
            for rank, (member, _) in enumerate(ranked, start=1)
        )

    viz_dir = stage_dir / "visualizations" / scope_name
    viz_dir.mkdir(parents=True, exist_ok=True)
    try:
        model.visualize_topics().write_html(str(viz_dir / "topics.html"))
    except Exception as error:
        log.warning("topic_viz_failed", scope=scope_name, error=str(error))

    return assignments, metadata, terms, centroids, examples


def cluster_segments(
    segments: list[Segment],
    embeddings: np.ndarray,
    settings: Settings,
    stage_dir: Path,
) -> tuple[
    list[TopicAssignment],
    list[ClusterMetadata],
    list[TopicTerm],
    list[CentroidSegmentRecord],
    dict[str, dict[int, list[str]]],
]:
    segment_rows = {
        segment.segment_id: index for index, segment in enumerate(segments)
    }
    customer_segments = [
        segment
        for segment in segments
        if segment.source_set
        in {SourceSet.customer_support, SourceSet.account_manager}
    ]
    internal_segments = [
        segment
        for segment in segments
        if segment.source_set == SourceSet.internal_discuss
    ]

    all_assignments: list[TopicAssignment] = []
    all_metadata: list[ClusterMetadata] = []
    all_terms: list[TopicTerm] = []
    all_centroids: list[CentroidSegmentRecord] = []
    example_map: dict[str, dict[int, list[str]]] = {}

    for scope_name, topic_version, scope_segments in (
        ("customer-topic-v1", "customer-topic-v1", customer_segments),
        ("internal-topic-v1", "internal-topic-v1", internal_segments),
    ):
        assignments, metadata, terms, centroids, examples = _fit_scope(
            scope_name,
            topic_version,
            scope_segments,
            embeddings,
            segment_rows,
            settings,
            stage_dir,
        )
        all_assignments.extend(assignments)
        all_metadata.extend(metadata)
        all_terms.extend(terms)
        all_centroids.extend(centroids)
        example_map[topic_version] = examples

    write_jsonl(stage_dir / "topic_assignments.jsonl", all_assignments)
    write_jsonl(stage_dir / "cluster_metadata.jsonl", all_metadata)
    write_json(
        stage_dir / "discovery.json",
        {
            "customer_segments": len(customer_segments),
            "internal_segments": len(internal_segments),
            "topics": len(
                [item for item in all_metadata if not item.is_outlier]
            ),
        },
    )
    representation_dir = stage_dir.parent / "topic_representation_stage"
    write_jsonl(representation_dir / "topic_terms.jsonl", all_terms)
    write_jsonl(
        representation_dir / "centroid_segments.jsonl",
        all_centroids,
    )
    log.info(
        "clustering_complete",
        assignments=len(all_assignments),
        topics=len([item for item in all_metadata if not item.is_outlier]),
    )
    return (
        all_assignments,
        all_metadata,
        all_terms,
        all_centroids,
        example_map,
    )
