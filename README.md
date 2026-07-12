# Transcript Intelligence

Batch analytics pipeline for ~100 call transcripts: local PII redaction, call-type classification, topic discovery, sentiment/findings, and evidence-linked Plotly charts.

## Input

```text
dataset/<ulid>/
  transcript.json      # required
  meeting-info.json    # required (startTime used as call datetime)
```

Other sibling files (`summary.json`, etc.) are ignored.

## Install

Use Python 3.11–3.13 (3.14 is not supported by spaCy wheels yet):

```shell
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m spacy download en_core_web_sm
cp .env.example .env   # set OPENAI_API_KEY
```

## Run

```shell
transcript-intelligence \
  --input interview-assignment/dataset \
  --output executions \
  --verbose
```

CLI flags are only `--input`, `--output`, and `--verbose`. All other settings come from `.env` (see `.env.example`), including `CLASSIFY_CONFIDENCE_THRESHOLD` for flagging low-confidence call-type labels.

## Pipeline

1. **Ingest** — load utterances + meeting `startTime`
2. **Privacy** — Presidio/regex PII; speaker names seeded first so body mentions share the same `[PERSON_xx]` tokens
3. **Classify** — LLM assigns `customer-support` / `account-manager` / `internal-discuss` (no speaker roles)
4. **Turns** — one turn per redacted utterance
5. **Segments / embeddings / BERTopic** — dual topic spaces (customer vs internal)
6. **Topic labels + sentiment/findings** — online structured LLM with quote evidence
7. **Aggregation / analytics** — monthly and all-time metrics → Plotly HTML under `analytical_stage/html/`

Resume: incomplete `execution_<id>` directories are reused; completed stages are skipped.

## Evidence lookup

```text
chart_point_id
  → aggregation_stage/metric_contributors.jsonl
  → segment_stage/segments.jsonl
  → ingest_stage/transcripts.jsonl
```

Low-confidence classifications are listed in `classify_stage/review_queue.jsonl`.

## Deferred

- Scalability / distributed workers
- Provider batch LLM mode
- Encrypted PII mapping files
- Soft-eval against provider `sentimentType`
- Production human-review workflow
