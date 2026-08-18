# LIBERO-PLUS / LIBERO-PRO setup validation (2026-08-18)

- LIBERO-PLUS assets downloaded and unpacked: 9.5G, 448799 files.
- LIBERO-PRO benchmark assets, BDDL and initial states are present; its optional datasets directory was created for the config path.
- Both variants import and construct their first LIBERO Spatial environment.
- mimic-video checkpoints are present (video 3.9G, action 998M, tokenizer 508M, T5 embeddings 42M).
- The Tesla T4 (15GB VRAM, SM75) requires FP16 rather than BF16. T4 compatibility fixes were added to the inference path.
- Full policy rollout now runs on T4 with FP16 and query-blocked PyTorch math SDPA. A full unblocked math-SDPA call still requests an additional 21.97GiB and fails; do not remove the SM75 chunking fallback.

Smoke outputs:
- eval_outputs/libero_spatial_plus/smoke_1ep_t4_math_backend/
- eval_outputs/libero_spatial_pro/smoke_1ep_t4_math_backend/
