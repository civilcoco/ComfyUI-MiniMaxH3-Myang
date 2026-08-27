# Split timeout failure analysis

## Parent claim and question

The long-video pipeline should still produce distinct chronological segment
prompts when a configured LLM times out or returns unusable JSON. The bounded
question was whether the reported failure was model-specific and whether a
model-independent fallback could avoid aborting or duplicating shots.

## Execution envelope

The check is CPU-only and uses mocked LLM responses; no GPU render, network
request, provider credentials, or user output files are required. The fixed
conditions are the existing segment-count/time math, prompt ladder, cache key,
media tag syntax, and downstream `plan_json` contract.

## Slices and evidence

1. **Transport inspection (claim-carrying).** All chat models called the same
   `_http_post_json(..., timeout=120)` path. The supplied trace failed with
   `The read operation timed out` before response parsing. This confirms a
   shared client failure boundary rather than a JSON-format or one-model issue.
2. **Timeout circuit breaker (claim-carrying).** One mocked read timeout after
   the adaptive wait now produces exactly one remote call, then five distinct
   local prompts. Evidence:
   `tests/test_splitter_with_media.py`,
   `test_timeouts_trip_the_local_fallback_circuit_breaker`.
3. **Empty and wrong-count responses (supporting).** Six empty replies and a
   persistent 3-of-5 reply both end in five distinct local prompts. The old
   whole-script/final-segment padding behavior is not reachable. Evidence:
   `test_empty_split_uses_distinct_local_timeline` and
   `test_wrong_segment_count_falls_back_without_padding`.
4. **Regression boundary (supporting).** Director, native-anchor, detail-pass,
   media-agent, splitter, service-config, and progress tests pass. The only
   initial failures were test-environment ACL and Node-path issues; both passed
   when rerun with the correct execution environment.

## Intervention

- HTTP read timeouts are reported with their actual timeout duration.
- One timeout after the adaptive/manual wait opens a circuit breaker instead of
  trying every remaining prompt variant.
- The default timeout is estimated from total system+user characters and the
  output budget (bounded to 180–900 seconds); each model can override it with a
  30–1800 second value in the settings panel.
- LLM payloads are accepted only when they contain exactly the required number
  of non-empty segments.
- A deterministic partitioner extracts explicit global preamble lines, splits
  ordered screenplay units, balances them into exactly N contiguous chunks,
  carries detected character-reference image tags, and marks likely hard cuts.
- `split_source` and `split_fallback_reason` are stored in `plan_json`, cache,
  logs, and the preview panel.

## Claim update and comparability

The original claim is strengthened: remote LLM quality remains preferred, but
remote availability and JSON compliance are no longer required for the video
graph to continue. Local fallback prompts are not semantically equivalent to an
LLM rewrite and should not be compared as equal-quality prose; they are a
reliability path that preserves order and distinct content without inventing a
successful LLM result.

## Next route

Stop this failure-analysis slice. Restart ComfyUI to load the new Python code,
then rerun the same workflow. A fallback run is successful when the preview
shows `本地时间线兜底`, `plan_json` contains the requested number of distinct
segments, and video generation proceeds instead of raising a splitter error.
