# Reproducibility Guide

Этот документ описывает минимальную подготовку машины для повторения eval в
этом форке. Он не предполагает, что большие файлы находятся в Git.

## Платформа

Поддерживаемая конфигурация: Linux, Python 3.10, NVIDIA GPU и CUDA 12.6
совместимый driver. SIMPLER/INT-ACT запускаются как отдельные процессы,
поэтому один процесс должен иметь в распоряжении одну GPU. Для рендеринга
SIMPLER нужен Vulkan (`libvulkan1`).

```bash
sudo apt install libvulkan1 tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
cd model
uv sync --extra cu126
source .venv/bin/activate
```

Установка SIMPLER-Bridge выполняется из корня репозитория:

```bash
model/.venv/bin/pip install -r eval/bridge/SimplerEnv/requirements.txt
model/.venv/bin/pip install -e eval/bridge/SimplerEnv/ManiSkill2_real2sim
model/.venv/bin/pip install -e eval/bridge/SimplerEnv
```

Для INT-ACT используйте его зафиксированные submodules и инструкции в
`eval/int-act/README.md`; launchers автоматически добавляют нужные пути в
`PYTHONPATH`.

## Обязательные model artifacts

Все пути ниже относительны к корню репозитория. Скрипты проверяют их до старта
rollout, поэтому отсутствие файла завершается понятной ошибкой.

| Артефакт | Где получить | Ожидаемый путь |
|---|---|---|
| Official MIMIC-Video Bridge checkpoint | `huggingface.co/jonpai/mimic-video` | `model/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt` |
| Bridge action decoder | тот же download | `model/checkpoints/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt` |
| Bridge normalization stats | тот же download | `model/checkpoints/dataset_statistics/bridge.json` |
| T5-11B text encoder | `jonpai/mimic-video` | `model/checkpoints/text_encoder/t5-11b/` |
| LIBERO Spatial/Object checkpoint'ы | `jonpai/mimic-video` | `model/checkpoints/` |
| LIBERO prompt embeddings | `nvidia/Cosmos-Policy-LIBERO-Predict2-2B` | `model/checkpoints/libero_t5_embeddings.pkl` |
| GR00T N1.7 LIBERO | NVIDIA checkpoint | `checkpoints/GR00T-N1.7-LIBERO/libero_spatial/` |
| GR00T N1.7 SIMPLER Bridge | NVIDIA checkpoint | путь, переданный в соответствующий launcher через `MODEL_PATH` |

Сначала авторизуйтесь в Hugging Face, затем скачайте official MIMIC-Video
checkpoint bundle. Название модели зависит от нужного benchmark'а:

```bash
hf auth login
model/.venv/bin/python model/scripts/download_checkpoints.py
```

Для минимального обычного LIBERO Spatial eval достаточно `libero_spatial_one`.
Текущая версия загрузчика содержит слишком широкий legacy pattern для video
backbone, поэтому воспроизводимый выборочный вариант:

```bash
model/.venv/bin/hf download jonpai/mimic-video \
  video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt \
  action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt \
  dataset_statistics/libero_spatial_one.json \
  --local-dir model/checkpoints

wget -O model/checkpoints/libero_t5_embeddings.pkl \
  https://huggingface.co/nvidia/Cosmos-Policy-LIBERO-Predict2-2B/resolve/main/libero_t5_embeddings.pkl
```

T5-11B занимает около 45.2 GB. Для повторяемой докачки используйте отдельный
resumable downloader:

```bash
bash scripts/eval/download_t5_11b_for_simpler_bridge.sh
```

Он скачивает `pytorch_model.bin`, `config.json` и tokenizer metadata в
`model/checkpoints/text_encoder/t5-11b/`. При прерванной загрузке скрипт
продолжает загрузку частей. Не добавляйте ни checkpoint'ы, ни `.venv/` в Git.

## Benchmark artifacts

### LIBERO

Установите локальный пакет:

```bash
model/.venv/bin/pip install -r eval/libero/LIBERO/requirements.txt
model/.venv/bin/pip install -e eval/libero/LIBERO
```

Official LIBERO datasets загружаются из `eval/libero/LIBERO`:

```bash
cd eval/libero/LIBERO
../../../../model/.venv/bin/python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

Для LIBERO-PRO нужны отдельные assets, BDDL и init files; см.
`LIBERO-PRO/README.md`. Они являются локальной зависимостью и не включаются в
коммит этого репозитория.

### SIMPLER-Bridge

Bridge assets поставляются с pinned `ManiSkill2_real2sim` checkout в
`eval/bridge/SimplerEnv/ManiSkill2_real2sim/data/`. Launcher'ы используют
`bridge_real_eval_1.png` из `data/real_inpainting/`. Если directory не
присутствует, инициализируйте submodule из upstream SIMPLER checkout согласно
`docs/ORIGINAL_README.md`.

### INT-ACT Object OOD

INT-ACT расположен в `eval/int-act/` и использует собственные pinned copies
SIMPLER и ManiSkill2. Перед запуском убедитесь, что есть:

```text
eval/int-act/third_party/ManiSkill2_real2sim/data/
eval/int-act/third_party/ManiSkill2_real2sim/data/real_inpainting/
```

Official INT-ACT 8-task Object OOD subset определён в
`eval/int-act/config/experiment/simpler/pi0_baseline_bridge_ev_ood.yaml`.
Расширенный 16-task набор используется только в специально помеченных
launcher'ах.

## Prompt embeddings

SIMPLER и INT-ACT launcher'ы сначала вычисляют T5 embeddings на CPU, затем
переиспользуют `.pt` файл на GPU. Не переносите encoding на GPU: T5-11B не
нужен в GPU памяти во время rollout.

- SIMPLER-Bridge: `scripts/eval/precompute_simpler_bridge_t5_embeddings.py`
- INT-ACT 16-task: `scripts/eval/precompute_intact_object_ood_t5_embeddings.py`
- INT-ACT 8-task: используется совместимый embedding dictionary 16-task
  launcher'а; он содержит все восемь инструкций.

## Проверка готовности

Перед обычным LIBERO Spatial decoder run проверьте:

```bash
test -s model/checkpoints/libero_t5_embeddings.pkl
test -s model/checkpoints/video_backbone/v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused.pt
test -s model/checkpoints/action_decoder/w2a_libero_spatial_one_v2w_libero_spatial_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000007540_fused_lr1.000e-04_layer20_bsz128_iter_000019998.pt
```

До долгого запуска проверьте наличие ключевых файлов:

```bash
for path in \
  model/.venv/bin/python \
  model/checkpoints/text_encoder/t5-11b/pytorch_model.bin \
  model/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt \
  model/checkpoints/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt \
  model/checkpoints/dataset_statistics/bridge.json; do
  test -f "$path" || echo "missing: $path"
done
```

Имена checkpoint'ов в этой проверке совпадают с SIMPLER-Bridge и INT-ACT
launcher'ами. При добавлении другого benchmark'а используйте проверки в
соответствующем `run_mimic_*` script как source of truth.
