# Перенос workspace на новый сервер

Этот документ переносит рабочее окружение для `mimic-video`, `openvla-oft`,
`LIBERO-PRO`, `LIBERO-plus` и `cosmos-policy`. Он сохраняет исходный код и
локальные коммиты, но намеренно не переносит видео, логи и сырые трассы
экспериментов. Именно они занимают основную часть текущего workspace и не нужны
для продолжения экспериментов.

## Что нужно на новом сервере

- Linux x86-64, NVIDIA GPU и установленный драйвер: `nvidia-smi` должен
  работать до установки Python-пакетов.
- CUDA-совместимый драйвер. `mimic-video` по умолчанию использует PyTorch
  CUDA 12.6 (`MIMIC_EXTRA=cu126`); при другом стеке CUDA выберите совместимый
  extra до запуска установки.
- Не менее 16 GB VRAM для OpenVLA LIBERO eval. Для больших прогонов
  `mimic-video` требуется существенно больше свободной памяти GPU.
- Доступ к GitHub и Hugging Face либо заранее скопированные bundle, checkpoints
  и datasets.
- Для `cosmos-policy` дополнительно Docker + NVIDIA Container Toolkit;
  его официальный путь запуска изолирован в контейнере.

## 1. На старом сервере: сохранить исходники и локальные коммиты

Из корня workspace:

```bash
bash mimic-video/scripts/migration/export_workspace_bundle.sh
```

Скрипт создаст каталог `transfer-bundle-<UTC timestamp>/`. В нём находятся Git
bundles всех пяти репозиториев, включая коммиты, которые ещё не были отправлены
на remote. Перенесите этот каталог на новую машину:

```bash
rsync -aP transfer-bundle-YYYYMMDDTHHMMSSZ/ user@new-host:/srv/robotics/
```

Если у вас уже есть private remote со всеми коммитами, вместо bundle можно
клонировать репозитории обычным `git clone`, но bundle надёжнее для текущей
рабочей ветки.

## 2. Перенести только нужные данные

### Обязательно для продолжения `mimic-video` evaluation

Скопируйте `mimic-video/model/checkpoints/`. В нём ожидаются как минимум:

- video backbone;
- action decoder;
- dataset statistics;
- T5 embeddings для LIBERO.

Команда, которую следует выполнить на старом сервере (путь назначения замените):

```bash
rsync -aP --prune-empty-dirs \
  mimic-video/model/checkpoints/ user@new-host:/srv/robotics/robotics_project/mimic-video/model/checkpoints/
```

При использовании селекторов CatBoost также скопируйте нужные компактные модели
из `mimic-video/artifacts/`. Полные dataset/кэши Hugging Face переносить не
обязательно: они скачиваются заново после `hf auth login`.

### Обязательно для LIBERO-PRO / OpenVLA

Код `LIBERO-PRO` уже содержит добавленные spatial task suites. Если нужны все
официальные suites, скопируйте или скачайте BDDL/init файлы согласно
`LIBERO-PRO/README.md`. Для `LIBERO-plus` отдельно потребуются внешние assets
в `LIBERO-plus/libero/libero/assets/`; они не хранятся в Git.

### Намеренно не переносить

- `mimic-video/eval_outputs/**/videos/`;
- `mimic-video/eval_outputs/**/logs/`;
- `.jsonl`, `.csv`, `.pt`, `episode_artifacts/` внутри `eval_outputs`;
- `__pycache__/`, `.venv/`, `hf_download/`, build-каталоги;
- `rollouts/` и результаты пробных запусков.

Если конкретный старый rollout нужен для статьи, скопируйте его отдельным
архивом. Он не нужен для воспроизводимости кода.

## 3. На новом сервере: восстановить код

```bash
mkdir -p /srv/robotics/robotics_project
cd /srv/robotics/robotics_project
# Сначала клонируется только mimic-video: в нём находятся migration scripts.
git clone /srv/robotics/transfer-bundle-YYYYMMDDTHHMMSSZ/bundles/mimic-video.bundle mimic-video
bash mimic-video/scripts/migration/restore_workspace_from_bundle.sh /srv/robotics/transfer-bundle-YYYYMMDDTHHMMSSZ
```

Скрипт откажется перезаписывать существующий репозиторий. Это защита от потери
данных. После восстановления проверьте `git status` во всех репозиториях.

## 4. Установить системные и Python-зависимости

```bash
cd /srv/robotics/robotics_project
bash mimic-video/scripts/migration/bootstrap_host.sh
# Перезапустите shell, если только что установили conda или uv.
bash mimic-video/scripts/migration/setup_python_environments.sh
```

Скрипт создаёт:

- conda environment `openvla-oft` (Python 3.10) для OpenVLA и LIBERO-PRO;
- `mimic-video/model/.venv` через `uv` для `mimic-video`.

`LIBERO-plus` устанавливается только при явном флаге, поскольку требует
отдельных assets:

```bash
INSTALL_LIBERO_PLUS=1 bash mimic-video/scripts/migration/setup_python_environments.sh
```

Не копируйте Hugging Face token между серверами. На новом сервере войдите в
свой аккаунт интерактивно:

```bash
hf auth login
```

## 5. Проверить установку

```bash
bash mimic-video/scripts/migration/verify_installation.sh
```

Это проверит Git-состояние, видимость GPU в обоих окружениях и импорт
consensus-medoid. Скрипт не скачивает модели и не запускает долгий rollout.

## 6. Продолжить experiments

Для `mimic-video` используйте скрипты из `mimic-video/scripts/eval/`. Они
создают результаты под `eval_outputs/`; bulk-артефакты уже исключены через
`.gitignore`. В частности:

```bash
cd /srv/robotics/robotics_project/mimic-video
bash scripts/eval/run_libero_spatial_consensus_only_100ep.sh
```

Перед запуском проверьте и при необходимости переопределите в окружении:

- `GPU` — номер GPU;
- `PYTHON` — путь к Python `mimic-video/model/.venv/bin/python`;
- `T5_EMB` — путь к `libero_t5_embeddings.pkl`;
- `OUT_DIR` — каталог для нового запуска;
- `LIBERO_CONFIG_PATH` — путь к конфигурации LIBERO.

Для OpenVLA / LIBERO-PRO используйте `LIBERO-PRO/scripts/run_eval.sh`. Он
ожидает conda environment `openvla-oft` или задаваемый через
`LIBERO_PRO_PYTHON` интерпретатор. При первом запуске задайте новый каталог
результатов через `LIBERO_PRO_OUTPUT_DIR`, чтобы старые результаты не смешались
с новыми.

## Cosmos Policy

`cosmos-policy` не смешивайте с двумя окружениями выше: его upstream-инструкция
требует Docker. На новом сервере:

```bash
cd /srv/robotics/robotics_project/cosmos-policy
docker build -t cosmos-policy docker
```

Дальше используйте команды из `cosmos-policy/SETUP.md`. Перед запуском убедитесь,
что `docker run --gpus all ...` видит вашу NVIDIA GPU.

## Диагностика

- `torch.cuda.is_available() == False`: сначала проверяйте драйвер через
  `nvidia-smi`, затем соответствие CUDA extra в `mimic-video/model/pyproject.toml`.
- Ошибки EGL/MuJoCo: запускайте с `MUJOCO_GL=egl` и убедитесь, что установлены
  пакеты из `bootstrap_host.sh`.
- Ошибка отсутствующих assets: для LIBERO-plus assets не входят в Git; проверьте
  `LIBERO-plus/libero/libero/assets/`.
- Ошибка отсутствующего checkpoint: убедитесь, что переменные в eval-скрипте
  указывают на `mimic-video/model/checkpoints/`, а не на абсолютный путь старого
  сервера.

## A100 paired LIBERO-PLUS Object evaluation

For the current baseline-versus-consensus experiment, follow [A100_LIBERO_PLUS_OBJECT_EVAL.md](A100_LIBERO_PLUS_OBJECT_EVAL.md). It documents checkpoint/assets downloads and starts tmux in strict baseline-first, consensus-only-second order with summary-only outputs. Use Ampere-or-newer hardware.

## Current GR00T N1.7 evaluation additions

The repository now contains separate policy servers and clients for
SIMPLER-Bridge, the official 8-task INT-ACT Object OOD subset, LIBERO-Pro,
LIBERO-Plus and standard LIBERO Spatial. Checkpoints and external benchmark
trees remain local-only. Copy `scripts/eval/groot_n17_*`, the matching
`collect_groot_n17_*` scripts, and the launchers together; they share output
contracts documented in [EVALUATION.md](EVALUATION.md).

Do not migrate `eval_outputs/**/logs`, rollout videos, downloaded checkpoints or
the external benchmark checkouts through Git. Only compact summaries explicitly
curated for a result snapshot should be committed.
