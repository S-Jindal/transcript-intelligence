# Bonus ideas

## Speaker roles instead of `[PERSON_xx]` labels

### Problem

After privacy redaction, speakers become generic placeholders like `[PERSON_01]` / `[PERSON_02]`. Downstream LLM stages (especially sentiment and findings) often put those tokens into `target` / `value`, which then show up in chart categories and `chart_point_id`s as noise such as `positive:[PERSON_02]`.

Roles like `customer`, `agent`, `account_manager`, or `internal` would be more useful for analysis and charts, and would avoid leaking placeholder IDs into stakeholder-facing labels.

### Proposed improvement

Extend the **classify** stage (already one structured LLM call per transcript) to also return a **speaker → role** map:

1. Take redacted utterances for the transcript (or a window that still covers every unique `speaker_id` / `[PERSON_xx]` at least once).
2. Ask the model to assign each observed speaker token a role given dialogue cues (greetings, “thanks for calling support”, internal planning language, etc.).
3. Persist something like:

```json
{
  "transcript_id": "...",
  "source_set": "customer-support",
  "speaker_roles": {
    "[PERSON_01]": "agent",
    "[PERSON_02]": "customer"
  }
}
```

4. When materializing turns (or before sentiment/findings), rewrite speaker labels to the role (optionally keep the person token in lineage metadata only).

Classification is a natural place for this: the model already reads opening context to choose call type, and the same evidence usually reveals who is support vs customer vs internal. Doing it once per transcript is cheaper than inventing roles again inside every findings call.

### Why not earlier

There is no reliable sibling role file in the dataset. Inferring roles is judgmental, so it belongs in an explicit LLM step with confidence/rationale rather than hard-coded heuristics alone. Privacy must stay first: role labeling should run only on **already redacted** text.

### Expected impact

- Cleaner sentiment/finding categories (`positive:customer` instead of `positive:[PERSON_02]`)
- Better chart readability and `chart_point_id` usefulness
- Slightly richer turn/segment context for topic labeling without re-introducing personal names

### Risks / follow-ups

- Ambiguous multi-party calls (two customers, vendor + partner, etc.) need an `unknown` / `other` role and low-confidence flags (same pattern as `CLASSIFY_CONFIDENCE_THRESHOLD`)
- Role errors propagate to all downstream labels — keep the raw `[PERSON_xx]` mapping for audit
- Optional later: small heuristic pre-seed (e.g. “thank you for calling … support”) before the LLM call to cut ambiguity
