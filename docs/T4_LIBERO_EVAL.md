# Running mimic-video LIBERO eval on a Tesla T4

The server GPU is a Tesla T4 (SM75, 15 GiB). Use the dedicated launcher:

```bash
MAX_CONTROL_STEPS=120 OUT_DIR="$PWD/eval_outputs/libero_spatial_plus/t4_run" \
  bash scripts/eval/run_mimic_libero_variant_one_episode.sh plus 0
```

Replace `plus` with `pro` for LIBERO-PRO. The launcher forces FP16 and disables CUDA graphs; both are required for this GPU.

## T4 compatibility path

FlashAttention requires SM80+, while a full PyTorch math-SDPA attention call attempts a 21.97-GiB allocation and fails. On SM75, `model/cosmos_predict2/module/attention.py` now uses exact query-blocked math SDPA (`q_block=64`): each query block attends to all keys, so attention results are mathematically unchanged but no full Q×K score matrix is materialized.

Validated on 2026-08-18:

- LIBERO-PLUS: one five-control-step episode completed without CUDA OOM; outputs at `eval_outputs/libero_spatial_plus/t4_12gb_profile_block64_1ep/`.
- LIBERO-PRO: one five-control-step episode completed without CUDA OOM; outputs at `eval_outputs/libero_spatial_pro/smoke_1ep_t4_chunked_math_v3/`.

The observed instantaneous peak in the sampled PLUS run was about **13.7 GiB**, not 12 GiB. It is below the 15-GiB T4 capacity, but there is only ~1.3 GiB headroom. A strict 12-GiB ceiling cannot be asserted from the available historical repository logs; attaining it would require a separately validated reduction in model/sequence/image configuration and could alter evaluation comparability.
