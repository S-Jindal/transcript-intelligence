from collections import defaultdict

from transcript_intelligence.models import (
    Classification,
    RedactedUtterance,
    Turn,
)


def materialize_turns(
    utterances: list[RedactedUtterance],
    classifications: list[Classification],
) -> list[Turn]:
    source_by_transcript = {
        item.transcript_id: item.source_set for item in classifications
    }
    grouped: dict[str, list[RedactedUtterance]] = defaultdict(list)
    for utterance in utterances:
        grouped[utterance.transcript_id].append(utterance)

    return [
        Turn(
            turn_id=f"{transcript_id}:{utterance.index}",
            transcript_id=transcript_id,
            source_set=source_by_transcript[transcript_id],
            order=utterance.index,
            speaker=utterance.speaker_name,
            text=utterance.sentence,
            time=utterance.time,
            end_time=utterance.end_time,
        )
        for transcript_id, items in sorted(grouped.items())
        for utterance in sorted(items, key=lambda item: item.index)
    ]
