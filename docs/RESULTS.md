# Experiment Registry

Этот реестр отражает compact summaries, находившиеся в рабочем output tree на
2026-09-07. Он не заменяет `summary.json`, который остаётся source of truth.
Результаты с разными task set, seeds или selector parameters не являются
прямыми репликациями друг друга.

## SIMPLER-Bridge, MIMIC-Video Bridge checkpoint

Протокол во всех строках: `ftcosmos`, `stop_video_denoising_step=0`, четыре
задачи × 24 фиксированных episode variants, 96 rollout'ов.

| Run | Selector | Success | SR | Summary |
|---|---|---:|---:|---|
| Baseline | single sampled chunk | 45/96 | 46.88% | `eval_outputs/simpler_bridge/ftcosmos_stop0_full96_20260901_194203/summary.json` |
| Consensus medoid, early | K=3 | 28/96 | 29.17% | `eval_outputs/simpler_bridge/ftcosmos_stop0_consensus_medoid_only_k3_full96_20260902_072118/summary.json` |
| Consensus medoid, fixed `[0,1,2]` | K=3 | 52/96 | 54.17% | `eval_outputs/simpler_bridge/ftcosmos_stop0_consensus_medoid_only_k3_fixedseeds012_directresult_full96_20260902_090800/summary.json` |
| Rank fusion | K=3, action + hidden rank | 50/96 | 52.08% | `eval_outputs/simpler_bridge/rank_fusion_action_hidden_k3_full96_20260902_133327/summary.json` |

## INT-ACT Object OOD, 16-task expansion

Baseline uses 16 tasks × 24 episode variants × three sampling seeds,
1,152 rollout'ов total.

| Run | Overall SR | Important note | Summary |
|---|---:|---|---|
| Baseline MIMIC-Video | 216/1152, 18.75% aggregate | Individual model seed values differ; inspect per-seed data before averaging. | `eval_outputs/intact_simpler/mimic_video_object_ood_16tasks_3seeds_20260902_115024/summary.json` |
| Consensus medoid, old run | 426/1152, 36.98% aggregate | **Not three independent seeds.** The code then used candidate seeds `[0,1,2]` in every outer run, so seeds 0/1/2 are duplicate deterministic rollouts. Treat this as one fixed-candidate evaluation. | `eval_outputs/intact_simpler/mimic_video_object_ood_consensus_medoid_16tasks_3seeds_gpu7_20260902_200304/summary.json` |

The current custom-seed 8-task run is intentionally omitted until it has
completed all 192 rollout'ов and produced a final summary.

## LIBERO Spatial

| Run | Protocol | Result | Summary |
|---|---|---:|---|
| Baseline trace | 10 tasks × 10 episodes | 69/100, 69.00% | `eval_outputs/libero_spatial/baseline/trace_100ep/metrics/summary.json` |
| Consensus-only | 100 episodes | inspect local summary | `eval_outputs/libero_spatial/consensus_medoid/consensus_only_100ep_gpu1_20260713/metrics/summary.json` |

The LIBERO worktree also contains targeted pilot, CatBoost and latent-metric
experiments. They are catalogued in [experiments/README.md](../experiments/README.md)
and should not be elevated to headline numbers without a matched protocol.

## Interpretation rules

- Report the task set, episodes per task, `K`, executed prefix length and
  candidate seeds with every number.
- A model sampling seed changes diffusion noise, not the benchmark scene
  instance. Scene variation is determined by the benchmark `episode_id`.
- For fixed-candidate consensus, an outer loop over labels `seed=0,1,2` is not
  a three-seed estimate unless it changes the candidate seed set.
- The 8-task Object OOD config and the 16-task extension have different task
  compositions. Never pool their SR values.
