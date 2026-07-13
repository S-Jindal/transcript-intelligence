import hashlib
import json
from datetime import datetime
from pathlib import Path

from transcript_intelligence.io_utils import write_jsonl
from transcript_intelligence.logging_setup import get_logger
from transcript_intelligence.models import (
    IngestedUtterance,
    RawUtterance,
    TranscriptRecord,
)

log = get_logger(__name__)


def _parse_meeting_info(path: Path) -> tuple[datetime, datetime | None, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    start = datetime.fromisoformat(
        payload["startTime"].replace("Z", "+00:00")
    )
    end = (
        datetime.fromisoformat(payload["endTime"].replace("Z", "+00:00"))
        if payload.get("endTime")
        else None
    )
    return start, end, payload.get("duration")


def ingest_transcripts(input_directory: Path, stage_dir: Path) -> list[TranscriptRecord]:
    folders = sorted(
        path for path in input_directory.iterdir() if path.is_dir()
    )
    records: list[TranscriptRecord] = []
    utterances: list[IngestedUtterance] = []

    for folder in folders:
        transcript_path = folder / "transcript.json"
        meeting_info_path = folder / "meeting-info.json"
        if not transcript_path.exists():
            continue
        if not meeting_info_path.exists():
            raise FileNotFoundError(
                f"missing meeting-info.json for {folder.name}"
            )

        raw = json.loads(transcript_path.read_text(encoding="utf-8"))
        if not isinstance(raw.get("data"), list) or not raw["data"]:
            raise ValueError(f"empty transcript data in {transcript_path}")

        parsed = sorted(
            (
                RawUtterance.model_validate(item)
                for item in raw["data"]
            ),
            key=lambda item: item.index,
        )
        start, end, duration = _parse_meeting_info(meeting_info_path)
        content = transcript_path.read_bytes()
        transcript_id = folder.name
        records.append(
            TranscriptRecord(
                transcript_id=transcript_id,
                relative_path=str(
                    transcript_path.relative_to(input_directory)
                ),
                transcript_datetime=start,
                end_time=end,
                duration_minutes=duration,
                utterance_count=len(parsed),
                content_hash=hashlib.sha256(content).hexdigest(),
                size_bytes=len(content),
            )
        )
        utterances.extend(
            IngestedUtterance(
                transcript_id=transcript_id,
                index=item.index,
                speaker_id=item.speaker_id,
                speaker_name=item.speaker_name,
                sentence=item.sentence,
                time=item.time,
                end_time=item.end_time,
            )
            for item in parsed
        )

    if not records:
        raise FileNotFoundError(
            f"no transcript.json files found under {input_directory}"
        )

    write_jsonl(stage_dir / "transcripts.jsonl", records)
    write_jsonl(stage_dir / "utterances.jsonl", utterances)
    log.info(
        "ingest finished",
        transcripts=len(records),
        utterances=len(utterances),
    )
    return records
