# Bonus ideas

## 1. Speaker identity mapping and an account-level renewal early-warning score

### Problem

After privacy redaction, speakers become generic placeholders such as `[PERSON_01]`. Downstream LLM stages often copy those tokens into `target` and `value`, which pollutes chart categories with labels like `positive:[PERSON_02]`. Separately, all charts aggregate by call type and month, but nobody in the business acts on a month. Decisions are made about accounts.

Both problems share the same root cause: the pipeline never establishes who is speaking and which customer they represent.

### Proposed improvement

`meeting-info.json` already contains `organizerEmail`, `allEmails`, and `invitees`. These fields support a deterministic mapping without any CRM integration:

1. During ingest, match each `speaker_name` against the invitee email addresses. Names are usually recoverable from the local part of the address, for example `jordan.whitfield@aegiscloud.com`.
2. Assign each matched speaker a side. Vendor-domain addresses identify Aegis staff, and any other domain identifies the customer. The customer domain also identifies the account for the whole call.
3. Store the role alongside the pseudonym in the privacy mapping, so that downstream stages see `customer` or `agent` instead of a bare person token. Store only a hashed or pseudonymized form of the customer domain to remain consistent with the privacy stage.
4. Roll findings up per account: recent `renewal_risk` and `frustration` volume and intensity, unresolved issues, and the sentiment trend across the last several calls. Publish a ranked account watchlist with the top contributing finding reasons attached through the existing `finding_id` to `segment_id` audit path.
5. Once a CRM export becomes available, weight the score by contract value so the watchlist reads as revenue at risk rather than raw counts.

This deterministic email-based mapping is preferable to inferring roles from dialogue cues with an LLM, because it is cheaper, reproducible, and grounded in metadata that already ships with every call.

### Why it drives action

A customer success leader receives a statement of the form "these five accounts show rising renewal risk, and here is the exact call evidence" instead of "renewal risk increased in April". That output is directly assignable work: save plays, executive sponsor calls, and targeted remediation. The cleaner role labels also improve every chart category and `chart_point_id` that stakeholders see.

### Risks

- An email domain does not always equal an account. Consultants and partners appear on calls, so unmatched speakers need an `unknown` role and a confidence flag rather than a forced assignment.
- Some speakers will not match any invitee, for example when a name is abbreviated in the transcript. A small LLM fallback on redacted text can cover the remainder.
- Accounts with few calls produce noisy trends. The watchlist should display call counts next to scores, with the same "directional, not statistical" caveat the charts already carry.

## 2. Commitment ledger for kept and broken promises

### Problem

The findings stage already extracts `commitment` rows, for example a promise that a credit will appear on the next invoice or that an issue will be escalated to engineering. Today these rows are counted in charts and then forgotten. Nobody verifies whether the promise was honored.

### Proposed improvement

1. Persist commitments into a ledger keyed by account, using the identity mapping from the first idea, together with the commitment text, the call date, and the owning side.
2. On each later call for the same account, provide the account's open commitments as context to the findings stage and ask whether any were referenced as fulfilled, still pending, or complained about.
3. Report two views: open commitments that have exceeded their stated timeframe, and accounts where a broken promise co-occurs with new frustration or renewal-risk findings.

### Why it drives action

Broken commitments are among the strongest and most fixable churn drivers. Support and account managers receive a concrete follow-up queue with the original promise and its evidence trail, rather than an abstract sentiment trend.

### Risks

- Fulfillment often happens outside calls, for example through email or a ticket system. The ledger therefore flags commitments as "not confirmed on calls" rather than "not done", and the reporting language must preserve that distinction to avoid unfair blame.

## 3. Internal versus customer alignment gap

### Problem

The pipeline deliberately builds two topic spaces, one for customer-facing calls and one for internal discussions. Nothing currently compares them, even though the comparison answers a question every leadership team asks: are we spending our internal effort on what customers actually escalate?

### Proposed improvement

1. Embed both topic sets into the same vector space by passing their labels and descriptions through the embedding model the pipeline already uses.
2. Match each customer theme to its nearest internal themes and score it on two axes: customer prevalence weighted by negative-finding intensity, and internal discussion share.
3. Flag the mismatches in both directions. Themes that are loud among customers but nearly absent internally are blind spots. Themes that dominate internal time with little customer signal are candidates for over-investment.

### Why it drives action

This becomes a prioritization instrument for engineering and operations leadership. It provides measurable evidence in both directions and refreshes automatically with every pipeline run, so roadmap reviews can start from data instead of anecdote.

### Risks

- Internal calls are a small sample in this corpus, roughly thirty transcripts. The gap score should be treated as a prompt for human review rather than an automatic verdict.

## 4. Evidence-grounded analyst copilot

### Problem

Stakeholders will always have questions the pre-built charts do not answer. The traditional response is either to wait for an analyst or to paste raw transcripts into an LLM, which is expensive and degrades accuracy because the context window fills with irrelevant conversation.

Measurements on this repository support a cheaper path. Answering a product-quality question from filtered findings consumed roughly one percent of the tokens that feeding the raw transcripts would have required, because retrieval operated on structured, ID-linked outputs instead of full text.

### Proposed improvement

1. Build a small question-answering agent, as a chat interface or a CLI, whose only tools are queries over `metrics.jsonl`, filters over `findings.jsonl` by type, keyword, or account, and segment lookup by `segment_id`.
2. Require every claim in an answer to cite its `finding_id` or `segment_id`, so each response inherits the same audit path the charts use. The agent never reads raw transcripts.
3. Apply an escalation rule: fetch segment text only when a finding's reason is insufficient to answer the question. Typical queries then stay within a few thousand tokens.

### Why it drives action

Leadership can self-serve questions such as "what is driving billing complaints this quarter?" between pipeline runs, at near-zero marginal cost, and every answer arrives with citations a human can verify in one click. It also operationalizes the central economic argument of this design: analysis should consume structured evidence, not raw text that pollutes the context window.

### Risks

- The copilot is only as current as the latest execution, so every answer should surface the execution timestamp.
- Findings inherit whatever extraction bias the LLM has. The mandatory citation requirement is the guardrail, because any questionable claim can be checked against its source segment immediately.
