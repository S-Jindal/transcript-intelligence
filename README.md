# Transcript Intelligence

Batch analytics pipeline for ~100 call transcripts: local PII redaction, call-type classification, topic discovery, sentiment/findings, and evidence-linked Plotly charts.

![Pipeline overview](docs/pipeline.svg)

## Input

```text
dataset/<ulid>/
  transcript.json      # required — utterance list
  meeting-info.json    # required — startTime as call datetime
```

Other sibling files (`summary.json`, `speakers.json`, etc.) are ignored.

## Install

Use Python 3.11–3.13 (3.14 is not supported by spaCy wheels yet):

```shell
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
python -m spacy download en_core_web_sm
```

## Run

```shell
transcript-intelligence \
  --input interview-assignment/dataset \
  --output executions \
  --verbose
```

CLI flags are only `--input`, `--output`, and `--verbose`. All other settings come from `.env`. 

Incomplete `execution_<id>` directories are resumed; completed stages are skipped. A completed execution causes the next run to allocate `execution_{n+1}`. Before a non-complete stage runs, its stage output directory is cleared so stale artifacts are not mixed with new writes.

## Pipeline stages

### 1. Ingest

Discovers `*/transcript.json`, requires sibling `meeting-info.json`, validates utterances, and writes `transcripts.jsonl` + `utterances.jsonl`. Call datetime comes from `startTime` (not the folder ULID). `transcript_id` is the folder name.

### 2. Privacy (local)

**Microsoft Presidio** (spaCy NER) plus regex detectors for emails, phones, cards, account/case IDs. Speaker names are seeded into the pseudonym map first, then the same tokens are applied inside utterance text so `[PERSON_01]` is consistent as speaker and as a mention. Orgs/products on the allowlist are preserved. Reversible maps land under `privacy_stage/pii_mappings/`. Residuals go to a review queue but do not block the run.

Cloud LLM calls start only after this stage.

### 3. Classify (LLM)

Structured LLM call per transcript on the first N redacted utterances (`CLASSIFY_UTTERANCE_WINDOW`). Assigns `customer-support`, `account-manager`, or `internal-discuss` with confidence + rationale. No speaker-role invention (no ground-truth role file). Rows below `CLASSIFY_CONFIDENCE_THRESHOLD` are flagged in `classify_stage/review_queue.jsonl`.

### 4. Turns

One turn per redacted utterance (order = utterance index). Speaker is already a `[PERSON_xx]` token.

### 5. Segments + embeddings (local)

Adjacent turns are grouped into segments using **Sentence Transformers** `BAAI/bge-base-en-v1.5`: cosine similarity between turn embeddings decides topic shifts; a max token cap (`MAXIMUM_SEGMENT_TOKENS`, ≤500) force-splits before the model’s 512 limit. Segment vectors are written to `embeddings.npy`.

### 6. Clustering + topic representation (local) + labels (LLM)

**BERTopic** (UMAP → HDBSCAN → c-TF-IDF) discovers topics without a fixed taxonomy:

- **customer-topic-v1** — support ∪ account-manager segments together
- **internal-topic-v1** — internal-discuss alone

Outliers stay in lineage but are excluded from prevalence denominators. For each non-outlier cluster, top c-TF-IDF terms and centroid-nearest segments are selected automatically; an online LLM returns a short business label + description.

#### Topic-word configuration (c-TF-IDF)

Default BERTopic keeps English stop words and turns `[PERSON_01]` into tokens like `person_01`, which then dominate topic terms. This pipeline configures representation explicitly (clustering still uses BGE embeddings; only the **word list** changes):

| Setting | Value | Why |
|---|---|---|
| Placeholder strip | `[TAG_NN]` removed before tokenize | Stops `person02`-style leakage from PII tags |
| Stop words | sklearn English + conversational fillers (`yeah`, `ok`, `um`, `im`, …) + `person01` / `person_02`-style tokens | Chat noise is not in the default English list |
| `ngram_range` | `(1, 2)` | Unigrams and bigrams (e.g. `password reset`) |
| `min_df` | `2` | Drop rare one-off tokens |
| `reduce_frequent_words` | `True` on `ClassTfidfTransformer` | Uses `sqrt(TF)` so high-frequency fillers rank lower |

c-TF-IDF alone does **not** zero out words that appear in every cluster (its IDF uses total term count across class-docs, and within-topic ranking is still driven by TF). Stop lists + `reduce_frequent_words` are what keep terms business-relevant.

When a stage is re-run (`pending` / not `complete`), its output directory is deleted first (clustering also clears `topic_representation_stage/`) so old term files and charts cannot linger.

### 7. Sentiment / findings (LLM)

All call types. Each segment gets one structured LLM call using a prompt chosen by source set: customer-facing (support + account-manager) vs internal-discuss. Each row has `finding_type`, `target`, `value`, `reason`, and optional `intensity`.

Customer types emphasize buyer signals (`process_friction`, `renewal_risk`, `feature_request`, …). Internal types emphasize operating signals (`operational_risk`, `delivery_risk`, `capacity_constraint`, …). Affective signals (including frustration) use `finding_type=sentiment` with `value` of `positive` or `negative`. Competitor pressure uses `competitive_risk` in both prompts. Provider `sentimentType` on raw utterances is ignored.

Charts split those rows:

| Chart | Rows used | Category string |
|---|---|---|
| Sentiment | `finding_type == "sentiment"` | `{value}:{target}` (e.g. `negative:billing experience`) |
| Findings | all other types | `{finding_type}:{value}` (e.g. `renewal_risk:…`) |

`chart_point_id` is `sentiment|…` or `finding|…` plus source, month, and that category. Because text is redacted, the model sometimes puts `[PERSON_xx]` into `target`/`value`, which can pollute labels — see [BONUS.md](BONUS.md) for a speaker-role fix.

### 8. Aggregation + analytics

Pandas builds **segment-based** rates (not call- or customer-based). `metrics.jsonl` stores numerator, denominator, rate, and `distinct_transcripts`. `metric_contributors.jsonl` stores **numerator-only** lineage (denominator membership lists are omitted as redundant).

Monthly buckets are calendar months of `meeting-info` `startTime` (`YYYY-MM`) — e.g. `2026-03` means calls **in** March 2026, not “everything before March.” Plotly may display that as a date like “March 1, 2026”; treat it as the month label.

Plotly HTML lands under `analytical_stage/html/`.

## Analysis outputs

All artifacts for a run live under `execution_<id>/`.

### Business metric charts

Under `analytical_stage/html/`:

| File | Meaning |
|---|---|
| `topic_prevalence_monthly.html` | Top **5** topics by segment count per call type and month |
| `topic_prevalence_all_time.html` | Top **5** topics by segment count per call type |
| `sentiment_distribution.html` | Top **7** sentiment categories per call type and month |
| `finding_prevalence.html` | Top **7** finding categories per call type and month |
| `topic_hierarchy.html` | Dendrogram of **all** topic labels (both topic spaces) clustered by `label: description` embeddings |

The topic hierarchy is a single agglomerative tree (average linkage, cosine distance) over every topic from `topics.jsonl`, so related themes across customer-support, account-manager, and internal-discuss sit in the same branch. Merging stops once similarity drops below `TOPIC_MERGE_SIMILARITY_THRESHOLD` (default `0.6`); the dashed cut line marks that point and leaf markers are colored by each topic's dominant source set (legend included). The resulting merged groups are written to `analytical_stage/topic_hierarchy_groups.json`, where cross-set groups expose inter-set relationships (e.g. an internal outage postmortem topic grouped with customer outage escalation topics).

Shared chart behavior:

- Stacked bars; **largest segment share at the bottom**
- Number **inside** each box = **distinct transcripts** (calls), so one long call cannot look like many customers
- Hover shows segment numerator/denominator, rate, transcript count, and `chart_point_id`
- Sticky **copy bar**: hover fills a selectable `chart_point_id`; **Copy** or click the bar to copy
- Full-viewport layout; legend under the plot
- Sentiment/findings colors: **greens** for positive-leaning signals, **reds** for negative-leaning (findings mapped by type, e.g. `opportunity` vs `renewal_risk`)

Small denominators are directional, not statistically strong.

BERTopic Plotly views (when available) are under `clustering_stage/visualizations/`.

### How to read a chart point

1. Copy `chart_point_id` from the sticky bar (or chart hover / `analytical_stage/chart_manifest.json`).
2. Find matching **numerator** rows in `aggregation_stage/metric_contributors.jsonl` (denominator size is on the metric row).
3. Open the segment via `segment_id` in `segment_stage/segments.jsonl`.
4. Open call metadata via `transcript_id` in `ingest_stage/transcripts.jsonl`.

For an LLM finding, open `sentiment_stage/findings.jsonl` and follow `segment_id` (reason + fields explain the call-out; the segment is the evidence text).

Topic labels and membership:

```text
clustering_stage/topic_assignments.jsonl
topic_label_stage/topics.jsonl
```

Low-confidence call-type labels: `classify_stage/review_queue.jsonl`.

## Bonus ideas

See [BONUS.md](BONUS.md) 

## Deferred

- Scalability / distributed workers
- Provider batch LLM mode
- Encrypted PII mapping files
- Soft-eval against provider `sentimentType`
- Production human-review workflow
- Speaker-role rewriting in the live pipeline (documented in BONUS.md first)
