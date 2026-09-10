# Как запускать `mimic-video` eval на этом сервере

Ниже самый простой рабочий запуск **обычного** `mimic-video` eval для `libero_spatial`, без `catboost_select` и без других inference-стратегий.

## 1. Готовый скрипт

Используй:

```bash
cd /home/motovilovil/Robotics/robotics_project/mimic-video
bash scripts/eval/run_baseline_libero_spatial_one_rank.sh 0 3
```

Где:

- `0` — `eval_rank`
- `3` — номер GPU, который будет записан в `CUDA_VISIBLE_DEVICES`

По умолчанию скрипт запускает:

- модель `libero_spatial_one`
- `task_suite_name=libero_spatial`
- `max_eval_episodes=10`
- `max_control_steps=120`
- `regen_strategy=none`
- `--no-use-cuda-graphs`

Результаты пишутся в:

```bash
eval_outputs/libero_spatial/baseline/colleague_baseline_run/
```

## 2. Полезные env-переменные

Можно менять запуск без редактирования скрипта:

```bash
OUT_DIR=eval_outputs/libero_spatial/baseline/test_run \
MAX_EVAL_EPISODES=3 \
MAX_CONTROL_STEPS=120 \
WORLD_SIZE=1 \
bash scripts/eval/run_baseline_libero_spatial_one_rank.sh 0 3
```

Основные переменные:

- `OUT_DIR` — куда писать видео, метрики и лог
- `MAX_EVAL_EPISODES` — сколько эпизодов запускать
- `MAX_CONTROL_STEPS` — лимит шагов в rollout
- `WORLD_SIZE` — число параллельных rank-процессов
- `PYTHON` — путь до python из нужного conda/env
- `T5_EMB` — путь до `libero_t5_embeddings.pkl`

## 3. Что чаще всего вызывает `CUDA out of memory`

На этом проекте самые частые причины такие:

1. Запущено больше одного тяжёлого процесса на одном GPU.
   Проверяй `nvidia-smi` и убедись, что GPU не занят чужим eval.

2. Запуск без `--no-use-cuda-graphs`.
   В наших рабочих скриптах CUDA graphs отключены специально, потому что так стабильнее по памяти.

3. Не передан `--t5_embeddings_path`.
   Тогда `run.py` грузит text encoder вместо готовых embeddings, а это лишняя память.

4. Слишком большой параллелизм.
   Если у тебя один GPU, ставь `WORLD_SIZE=1` и запускай только один rank.

5. На том же GPU уже висят старые python-процессы.
   Перед запуском полезно проверить `nvidia-smi` и `ps aux | grep run.py`.

## 4. Безопасный порядок запуска

Если у коллеги постоянно OOM, лучше идти так:

1. Сначала короткий smoke-run:

```bash
cd /home/motovilovil/Robotics/robotics_project/mimic-video
MAX_EVAL_EPISODES=1 WORLD_SIZE=1 bash scripts/eval/run_baseline_libero_spatial_one_rank.sh 0 3
```

2. Если прошло, увеличить до `MAX_EVAL_EPISODES=3`.

3. Только потом запускать длинный eval.

## 5. Если всё равно OOM

Проверить по порядку:

- точно ли свободен GPU
- тот ли python/env используется
- существует ли файл `T5_EMB`
- не запущены ли два rank-процесса на одном и том же GPU

Если нужно максимально безопасно, сначала не запускать multi-GPU вообще:

```bash
WORLD_SIZE=1 MAX_EVAL_EPISODES=1 bash scripts/eval/run_baseline_libero_spatial_one_rank.sh 0 3
```

## 6. Где смотреть логи

- общий лог:

```bash
eval_outputs/libero_spatial/baseline/colleague_baseline_run/run_rank0.log
```

- метрики:

```bash
eval_outputs/libero_spatial/baseline/colleague_baseline_run/metrics/rank0/
```

- видео rollout:

```bash
eval_outputs/libero_spatial/baseline/colleague_baseline_run/videos/rank0/
```

## 7. Эквивалентная команда без скрипта

Если нужно запустить руками:

```bash
cd /home/motovilovil/Robotics/robotics_project/mimic-video
export CUDA_VISIBLE_DEVICES=3
export MUJOCO_GL=egl
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export WANDB_SILENT=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="/home/motovilovil/Robotics/robotics_project/mimic-video/model:/home/motovilovil/Robotics/robotics_project/mimic-video/eval/libero/LIBERO"

/home/motovilovil/miniconda3/envs/mimic_video_eval/bin/python eval/libero/run.py \
  --vam_experiment_name w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128 \
  --vam_video_model_path /home/motovilovil/Robotics/robotics_project/mimic-video/model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt \
  --vam_action_model_path /home/motovilovil/Robotics/robotics_project/mimic-video/model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt \
  --vam_dataset_statistics_path /home/motovilovil/Robotics/robotics_project/mimic-video/model/checkpoints/dataset_statistics/libero_spatial_one.json \
  --vam_img_horizon 5 --vam_lowdim_horizon 1 \
  --vam_stop_video_denoising_step 0 --vam_num_execute_actions 5 \
  --task_suite_name libero_spatial --num_trials_per_task 10 \
  --max_eval_episodes 1 --max_control_steps 120 \
  --eval_rank 0 --eval_world_size 1 \
  --t5_embeddings_path /home/motovilovil/Robotics_project_YSDA/model/checkpoints/libero_t5_embeddings.pkl \
  --regen_strategy none \
  --no-use-cuda-graphs
```

## Decoder action-token medoid

Полный обычный LIBERO Spatial прогон на одной GPU:

```bash
bash scripts/eval/launch_mimic_libero_spatial_decoder_token_gpu0_tmux.sh
```

По умолчанию используются fixed candidate seeds `[0,1,2]`, `K=3`, block 23 и
100 эпизодов. Нужные checkpoint-файлы и команда выборочной загрузки перечислены
в [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
