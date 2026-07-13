from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class QuoteMatch:
    start: int
    end: int
    edit_distance: int


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0
    if not left:
        return len(right)
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, start=1):
        current = [index]
        for right_index, right_char in enumerate(right, start=1):
            insert = current[right_index - 1] + 1
            delete = previous[right_index] + 1
            replace = previous[right_index - 1] + (
                0 if left_char == right_char else 1
            )
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def locate_quote(
    segment_text: str,
    quote: str,
    max_edits: int,
) -> QuoteMatch | None:
    stripped = quote.strip()
    if not stripped or max_edits < 0:
        return None

    exact_start = segment_text.find(stripped)
    if exact_start >= 0:
        return QuoteMatch(
            start=exact_start,
            end=exact_start + len(stripped),
            edit_distance=0,
        )

    quote_length = len(stripped)
    min_window = max(1, quote_length - max_edits)
    max_window = quote_length + max_edits
    best: QuoteMatch | None = None

    for window_length in range(min_window, max_window + 1):
        if window_length > len(segment_text):
            break
        last_start = len(segment_text) - window_length
        for start in range(last_start + 1):
            window = segment_text[start : start + window_length]
            distance = levenshtein(window, stripped)
            if distance > max_edits:
                continue
            if (
                best is None
                or distance < best.edit_distance
                or (
                    distance == best.edit_distance
                    and start < best.start
                )
            ):
                best = QuoteMatch(
                    start=start,
                    end=start + window_length,
                    edit_distance=distance,
                )
                if distance == 0:
                    return best
    return best
