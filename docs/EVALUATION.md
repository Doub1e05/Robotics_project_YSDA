# Evaluation Guide

Все команды запускаются из корня репозитория. Долгие эксперименты запускаются
в `tmux`; подключение: `tmux attach -t <session>`. Каждый launcher печатает
точное имя сессии и output directory.

## SIMPLER-Bridge

### Baseline MIMIC-Video

Полный протокол: 4 задачи × 24 фиксированных episode variants = 96 rollout'ов.
Launcher скачивает T5-11B при необходимости, вычисляет embeddings на CPU и
запускает четыре rank-процесса.

```bash
bash scripts/eval/launch_mimic_simpler_bridge_ftcosmos_full_tmux.sh
```

По умолчанию использует GPU 4--7. Для другого output root:

```bash
OUT_ROOT="$PWD/eval_outputs/simpler_bridge/my_baseline" \
SESSION=my_bridge_baseline \
bash scripts/eval/launch_mimic_simpler_bridge_ftcosmos_full_tmux.sh
```

### Consensus medoid

Использует три candidate action chunks, control-aware medoid selector и
исполнение первых пяти действий выбранного chunk. Перед запуском должны быть
готовы Bridge T5 embeddings.

```bash
T5_EMBEDDINGS="$PWD/eval_outputs/simpler_bridge/ftcosmos_stop0_full96_20260901_194203/t5_embeddings/bridge_t5_11b_embeddings.pt" \
bash scripts/eval/launch_mimic_simpler_bridge_ftcosmos_consensus_medoid_only_full_tmux.sh
```

По умолчанию selector использует candidate seeds `0,1,2`. Чтобы задать другой
фиксированный набор, экспортируйте `MIMIC_VIDEO_CANDIDATE_SEEDS` до запуска
rank-script или используйте INT-ACT custom-seed launcher ниже. `sampling seed`
модели и `episode_id` среды — разные сущности.

### Latent diagnostic subset

Этот режим намеренно сохраняет большие `.diagnostics.json` и предназначен
только для анализа success/failure chunks, а не для обычного benchmark eval.

```bash
bash scripts/eval/launch_mimic_simpler_bridge_latent_diagnostics_30_tmux.sh
```

Для обычных запусков не передавайте `--vam-save-diagnostics`: по умолчанию
сохраняются видео, action visualizations и summary, но не latent diagnostics.

## INT-ACT Object OOD

### Full 16-task baseline

```bash
GPUS="4 5 6 7" \
bash scripts/eval/launch_mimic_intact_object_ood_3seed_4gpu_tmux.sh
```

Это 16 задач × 24 episode variants × 3 model sampling seeds. Collector:

```bash
model/.venv/bin/python scripts/eval/collect_intact_object_ood_sr.py \
  --out-root <OUT_ROOT>
```

### Official 8-task Object OOD with fixed consensus candidates

```bash
bash scripts/eval/launch_mimic_intact_object_ood_consensus_seeds_2_996_997_gpu5_tmux.sh
```

Этот launcher использует 8 задач из official INT-ACT OOD config, 24 episode
variants на задачу (всего 192 rollout'а), `K=3` и candidate seeds
`[2, 996, 997]`. Он сохраняет summary через:

```bash
model/.venv/bin/python scripts/eval/collect_intact_object_ood_8task_sr.py \
  --out-root <OUT_ROOT> --candidate-seeds 2,996,997
```

Если процесс остановился, продолжите только отсутствующие episode IDs на двух
GPU:

```bash
OUT_ROOT=<OUT_ROOT> \
SESSION=my_intact_resume \
bash scripts/eval/resume_mimic_intact_object_ood_consensus_8tasks_2gpu_tmux.sh
```

`resume` проверяет video filenames с `obj_episode_<id>` и не повторяет готовые
rollout'ы. Список GPU в launcher можно изменить перед запуском.

## LIBERO Spatial

Один rank baseline:

```bash
bash scripts/eval/run_baseline_libero_spatial_one_rank.sh 0 0
```

Consensus-only 100-episode run:

```bash
bash scripts/eval/launch_libero_spatial_consensus_only_100ep_tmux.sh
```

Для CatBoost strategy runs используйте `eval/libero/run.py` с
`--regen_strategy`; готовые compact artifacts находятся в
`artifacts/libero_spatial/`. Подробности о ранних LIBERO запусках:
[experiments/README.md](../experiments/README.md).

## Output contract

Обычный eval сохраняет:

```text
<OUT_ROOT>/summary.json
<OUT_ROOT>/summary.tsv
<OUT_ROOT>/logs/
<OUT_ROOT>/seed*/result/     # rollout videos и action visualizations
```

`summary.json` — источник чисел SR. Не подсчитывайте SR по log lines: один
episode может быть retried, а resume launcher сознательно пропускает готовые
video files.
