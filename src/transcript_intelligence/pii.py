import re
from collections import Counter, defaultdict
from typing import Protocol

from transcript_intelligence.models import (
    EntitySpan,
    IngestedUtterance,
    PseudonymizedTranscript,
    RedactedUtterance,
    RedactionReport,
    TranscriptRecord,
)


class EntityDetector(Protocol):
    def detect(self, text: str) -> list[EntitySpan]: ...


class PresidioDetector:
    entities = (
        "PERSON",
        "ORGANIZATION",
        "PRODUCT",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "IP_ADDRESS",
        "US_SSN",
    )

    def __init__(self, spacy_model: str) -> None:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        engine = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        ).create_engine()
        self.analyzer = AnalyzerEngine(
            nlp_engine=engine,
            supported_languages=["en"],
        )

    def detect(self, text: str) -> list[EntitySpan]:
        return [
            EntitySpan(
                start=result.start,
                end=result.end,
                entity_type=result.entity_type,
                value=text[result.start : result.end],
                score=result.score,
                detector="presidio",
                action=(
                    "preserve"
                    if result.entity_type in {"ORGANIZATION", "PRODUCT"}
                    else "redact"
                ),
            )
            for result in self.analyzer.analyze(
                text=text,
                language="en",
                entities=list(self.entities),
                score_threshold=0.45,
            )
        ]


class RegexDetector:
    patterns = {
        "EMAIL_ADDRESS": re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
        "PHONE_NUMBER": re.compile(
            r"(?<!\w)(?:\+?\d[\d .()/-]{7,}\d)(?!\w)"
        ),
        "CREDIT_CARD": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
        "IP_ADDRESS": re.compile(
            r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"
        ),
        "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    }

    def __init__(self, account_patterns: tuple[str, ...]) -> None:
        self.account_patterns = [
            re.compile(pattern) for pattern in account_patterns
        ]

    @staticmethod
    def _valid_credit_card(value: str) -> bool:
        digits = [int(character) for character in value if character.isdigit()]
        return 13 <= len(digits) <= 19 and sum(
            digit
            if index % 2 == 0
            else (digit * 2 - 9 if digit > 4 else digit * 2)
            for index, digit in enumerate(reversed(digits))
        ) % 10 == 0

    def detect(self, text: str) -> list[EntitySpan]:
        findings = [
            EntitySpan(
                start=match.start(),
                end=match.end(),
                entity_type=entity_type,
                value=match.group(),
                score=1,
                detector=f"regex:{entity_type}",
            )
            for entity_type, pattern in self.patterns.items()
            for match in pattern.finditer(text)
            if entity_type != "CREDIT_CARD"
            or self._valid_credit_card(match.group())
        ]
        return findings + [
            EntitySpan(
                start=match.start(1 if match.lastindex else 0),
                end=match.end(1 if match.lastindex else 0),
                entity_type="ACCOUNT_ID",
                value=match.group(1 if match.lastindex else 0),
                score=1,
                detector="regex:ACCOUNT_ID",
            )
            for pattern in self.account_patterns
            for match in pattern.finditer(text)
        ]


class PiiProcessor:
    def __init__(
        self,
        regex_detector: RegexDetector,
        ner_detector: EntityDetector,
        allowlist: tuple[str, ...],
    ) -> None:
        self.regex_detector = regex_detector
        self.ner_detector = ner_detector
        self.allowlist = tuple(
            value.casefold() for value in allowlist if value.strip()
        )

    def _allowlist_spans(self, text: str) -> list[EntitySpan]:
        lowered = text.casefold()
        return [
            EntitySpan(
                start=match.start(),
                end=match.end(),
                entity_type="ALLOWLIST",
                value=text[match.start() : match.end()],
                score=1,
                detector="allowlist",
                action="preserve",
            )
            for value in self.allowlist
            for match in re.finditer(re.escape(value), lowered)
        ]

    @staticmethod
    def _resolve_group(group: list[EntitySpan]) -> EntitySpan:
        structured = [
            span for span in group if span.detector.startswith("regex:")
        ]
        if structured:
            winner = max(
                structured,
                key=lambda span: (span.end - span.start, span.score),
            )
        else:
            preserving = [span for span in group if span.action == "preserve"]
            winner = max(
                preserving or group,
                key=lambda span: (span.end - span.start, span.score),
            )
        signatures = {
            (span.start, span.end, span.entity_type, span.action)
            for span in group
        }
        return winner.model_copy(update={"conflict": len(signatures) > 1})

    def resolve(self, spans: list[EntitySpan]) -> list[EntitySpan]:
        groups: list[list[EntitySpan]] = []
        for span in sorted(spans, key=lambda item: (item.start, item.end)):
            if not groups or span.start >= max(item.end for item in groups[-1]):
                groups.append([span])
            else:
                groups[-1].append(span)
        return [self._resolve_group(group) for group in groups]

    def _placeholder(
        self,
        entity_type: str,
        value: str,
        counters: defaultdict[str, int],
        value_placeholders: dict[tuple[str, str], str],
        mapping: dict[str, str],
        aliases: list[tuple[str, str]],
    ) -> str:
        key = (entity_type, value.casefold())
        if key in value_placeholders:
            return value_placeholders[key]

        if entity_type == "PERSON":
            for (existing_type, existing_value), placeholder in value_placeholders.items():
                if existing_type != "PERSON":
                    continue
                parts = existing_value.split()
                if value.casefold() in parts or existing_value in value.casefold().split():
                    value_placeholders[key] = placeholder
                    aliases.append((value, placeholder))
                    return placeholder

        counters[entity_type] += 1
        placeholder = f"[{entity_type}_{counters[entity_type]:02d}]"
        value_placeholders[key] = placeholder
        mapping[placeholder] = value
        aliases.append((value, placeholder))
        return placeholder

    def process(
        self,
        transcript: TranscriptRecord,
        utterances: list[IngestedUtterance],
    ) -> tuple[
        PseudonymizedTranscript,
        RedactionReport,
        list[RedactedUtterance],
        dict[str, str],
    ]:
        counters: defaultdict[str, int] = defaultdict(int)
        value_placeholders: dict[tuple[str, str], str] = {}
        mapping: dict[str, str] = {}
        aliases: list[tuple[str, str]] = []

        speaker_names = list(
            dict.fromkeys(
                utterance.speaker_name for utterance in utterances
            )
        )
        speaker_tokens = {
            name: self._placeholder(
                "PERSON",
                name,
                counters,
                value_placeholders,
                mapping,
                aliases,
            )
            for name in speaker_names
        }

        # Unique first/last name parts reuse the speaker token in dialogue.
        part_owners: dict[str, set[str]] = defaultdict(set)
        for name, token in speaker_tokens.items():
            for part in name.split():
                part_owners[part.casefold()].add(token)
        for name, token in speaker_tokens.items():
            for part in name.split():
                if len(part_owners[part.casefold()]) == 1:
                    self._placeholder(
                        "PERSON",
                        part,
                        counters,
                        value_placeholders,
                        mapping,
                        aliases,
                    )

        join_text = "\n".join(
            f"{utterance.speaker_name}: {utterance.sentence}"
            for utterance in utterances
        )
        resolved = self.resolve(
            self.regex_detector.detect(join_text)
            + self.ner_detector.detect(join_text)
            + self._allowlist_spans(join_text)
        )
        redactions = [span for span in resolved if span.action == "redact"]
        for span in redactions:
            self._placeholder(
                span.entity_type,
                span.value,
                counters,
                value_placeholders,
                mapping,
                aliases,
            )

        replacements = sorted(
            {surface: placeholder for surface, placeholder in aliases}.items(),
            key=lambda item: len(item[0]),
            reverse=True,
        )

        def apply_map(text: str) -> str:
            result = text
            for surface, placeholder in replacements:
                result = re.sub(
                    rf"\b{re.escape(surface)}\b",
                    placeholder,
                    result,
                    flags=re.IGNORECASE,
                )
            return result

        redacted_utterances = [
            RedactedUtterance(
                transcript_id=utterance.transcript_id,
                index=utterance.index,
                speaker_id=utterance.speaker_id,
                speaker_name=speaker_tokens[utterance.speaker_name],
                sentence=apply_map(utterance.sentence),
                time=utterance.time,
                end_time=utterance.end_time,
            )
            for utterance in utterances
        ]
        redacted_join = "\n".join(
            f"{item.speaker_name}: {item.sentence}"
            for item in redacted_utterances
        )
        placeholder_ranges = [
            (match.start(), match.end())
            for match in re.finditer(r"\[[A-Z][A-Z0-9_]*_\d+\]", redacted_join)
        ]
        residual = [
            span.model_copy(update={"value": "[POTENTIAL_PII]"})
            for span in self.regex_detector.detect(redacted_join)
            if not any(
                start <= span.start and span.end <= end
                for start, end in placeholder_ranges
            )
        ]
        category_counts = Counter(
            entity_type for entity_type, _ in value_placeholders
        )
        report = RedactionReport(
            transcript_id=transcript.transcript_id,
            category_counts=dict(
                Counter(span.entity_type for span in redactions)
            ),
            unique_category_counts=dict(category_counts),
            replacement_count=len(redactions) + len(speaker_tokens),
            conflict_count=sum(span.conflict for span in resolved),
            redacted_character_ratio=(
                sum(span.end - span.start for span in redactions)
                / max(1, len(join_text))
            ),
            residual_category_counts=dict(
                Counter(span.entity_type for span in residual)
            ),
            review_required=bool(residual)
            or any(span.conflict for span in resolved),
        )
        return (
            PseudonymizedTranscript(
                transcript_id=transcript.transcript_id,
                transcript_datetime=transcript.transcript_datetime,
                replacement_count=report.replacement_count,
                unique_replacement_count=len(value_placeholders),
                conflict_count=report.conflict_count,
                redacted_character_ratio=report.redacted_character_ratio,
                residual_findings=residual,
                review_required=report.review_required,
            ),
            report,
            redacted_utterances,
            mapping,
        )
