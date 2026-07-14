import asyncio
import json

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from transcript_intelligence.config import Settings
from transcript_intelligence.logging_setup import get_logger
from transcript_intelligence.models import (
    Classification,
    RedactedUtterance,
    SourceSet,
)
from transcript_intelligence.remote_llm import (
    TokenUsage,
    _parse_with_retries,
)

log = get_logger(__name__)


class ClassificationResponse(BaseModel):
    source_set: SourceSet
    confidence: float = Field(ge=0, le=1)
    rationale: str


CLASSIFY_INSTRUCTIONS = """
Classify the call type from redacted transcript utterances.
Choose exactly one source_set:
- customer-support: customer reporting issues to support
- account-manager: renewals, adoption, account feedback with AM/CSM
- internal-discuss: internal engineering/product planning or sync
Return confidence between 0 and 1 and a short rationale.
""".strip()


async def _classify_one(
    client: AsyncOpenAI,
    settings: Settings,
    transcript_id: str,
    utterances: list[RedactedUtterance],
    semaphore: asyncio.Semaphore,
) -> tuple[Classification, TokenUsage]:
    window = utterances[: settings.classify_utterance_window]
    dialogue = "\n".join(
        f"{item.speaker_name}: {item.sentence}" for item in window
    )
    async with semaphore:
        parsed, usage = await _parse_with_retries(
            client,
            settings,
            CLASSIFY_INSTRUCTIONS,
            dialogue,
            ClassificationResponse,
            transcript_id,
        )
    assert isinstance(parsed, ClassificationResponse)
    return (
        Classification(
            transcript_id=transcript_id,
            source_set=parsed.source_set,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            low_confidence=(
                parsed.confidence
                < settings.classify_confidence_threshold
            ),
        ),
        usage,
    )


async def classify_transcripts(
    client: AsyncOpenAI,
    settings: Settings,
    utterances_by_transcript: dict[str, list[RedactedUtterance]],
) -> list[Classification]:
    semaphore = asyncio.Semaphore(settings.llm_concurrency)
    tasks = [
        _classify_one(
            client,
            settings,
            transcript_id,
            utterances,
            semaphore,
        )
        for transcript_id, utterances in sorted(
            utterances_by_transcript.items()
        )
    ]
    results: list[Classification] = []
    usage_total = TokenUsage()
    done = 0
    for task in asyncio.as_completed(tasks):
        result, usage = await task
        results.append(result)
        usage_total = usage_total + usage
        done += 1
        if done % 10 == 0 or done == len(tasks):
            log.info("classify progress", done=done, total=len(tasks))
    low = sum(item.low_confidence for item in results)
    log.info(
        "classify finished",
        transcripts=len(results),
        low_confidence=low,
        threshold=settings.classify_confidence_threshold,
        input_tokens=usage_total.input_tokens,
        output_tokens=usage_total.output_tokens,
        distribution=json.dumps(
            {
                source.value: sum(
                    item.source_set == source for item in results
                )
                for source in SourceSet
            }
        ),
    )
    return sorted(results, key=lambda item: item.transcript_id)
