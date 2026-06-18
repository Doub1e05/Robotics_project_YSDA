# Mimic-Video Timestep Trace Metrics

The eval writes model-output metrics without episode-level averaging.

## Files
- `episode_traces.jsonl`: one JSON object per episode with `meta`, `chunks`, and nested `actions`.
- `chunk_metrics.csv` / `chunk_metrics.jsonl`: one row per model inference chunk.
- `candidate_chunk_metrics.csv` / `candidate_chunk_metrics.jsonl`: one row per (chunk, candidate) when using `catboost_select` / `hybrid`.
- `action_metrics.csv` / `action_metrics.jsonl`: one row per executed action, suitable for filtering actions.
- `summary.json`: only counts and eval labels, not metric aggregates.

## Detailed docs
See `docs/METRICS_DETAILS.md` (Russian) for a full explanation of metric sources and formulas.

## Chunk-Level Metrics
Chunk rows contain one copy of diagnostics that belong to the whole predicted action chunk,
including retained video/action latent diagnostics and chunk action variability. These values
are not duplicated on each action row.

## Action-Level Metrics
Action rows contain the raw 10D model action, split position/rotation/gripper fields, and
per-action norms/deltas derived from model actions.

## Failure Episode Trimming
For unsuccessful episodes, chunk and action metric rows are kept only for timesteps
`0 <= t < 120` (6 seconds at 20 Hz). Full-timeout rollout videos are still saved without
trimming.

## Removed Metrics
The requested removed metrics are not emitted in trace, chunk, action, or summary files.
