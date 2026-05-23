# mimic-video — навигация по репозиторию

Исследовательский форк вокруг [mimic-video](https://arxiv.org/abs/2512.15692): двухстадийная Video-Action Model (Video2World + World2Action), оценка на **LIBERO Spatial**, логирование диагностических метрик, обучение CatBoost-классификаторов и стратегии «умного» inference при rollout.

Оригинальная документация по установке, обучению и чекпоинтам: [docs/ORIGINAL_README.md](docs/ORIGINAL_README.md). Архитектура модели: [MODEL.md](MODEL.md). Данные: [DATA.md](DATA.md).

**Отчёт по экспериментам:** [docs/REPORT.md](docs/REPORT.md).

---

## Структура каталогов

```
mimic-video/
├── README.md                 # этот файл (навигация)
├── MODEL.md, DATA.md         # upstream: модель и данные
│
├── docs/                     # документация исследования
│   ├── REPORT.md             # отчёт: метрики, CatBoost, стратегии inference
│   ├── METRICS.md            # формат trace-логов (chunk / action)
│   └── ORIGINAL_README.md    # оригинальный README проекта
│
├── model/                    # код модели, пайплайны, обучение (Cosmos-Predict2)
├── data_preprocessing/       # препроцессинг видео и эмбеддингов
├── eval/                     # скрипты оценки
│   └── libero/run.py         # основной entrypoint для LIBERO + regen-стратегии
│
├── scripts/                  # скрипты анализа и обучения классификаторов
│   ├── analysis/
│   │   └── compute_chunk_metric_ks.py   # KS-тест success vs failure
│   └── training/
│       ├── build_episode_split_and_train_catboost.py  # split 70/30 + chunk CatBoost
│       └── train_action_regeneration_classifier.py    # action-level CatBoost
│
├── notebooks/                # Jupyter: визуализация и разовые эксперименты
│   ├── analysis.ipynb
│   ├── libero_mimic_eval_5eps.ipynb
│   └── train_chunk_regeneration_classifier.ipynb
│
├── artifacts/                # обученные артефакты (копии для удобного inference)
│   └── libero_spatial/v2/    # CatBoost v2, split, список val-failure эпизодов
│
├── experiments/              # описание прогонов (см. experiments/README.md)
├── eval_outputs/             # сырые логи eval (метрики, видео, summary)
│   └── libero_spatial/
│       ├── baseline/         # базовый прогон 100 эпизодов
│       ├── strategy_runs/    # catboost_select / action_dynamic / hybrid
│       ├── pilot_runs/       # короткие пилоты (6 эпизодов)
│       └── other/            # прочие бенчмарки (напр. libero_90)
│
├── logs/                     # общие лог-файлы длительных прогонов
└── assets/                   # иллюстрации
```

---

## Быстрый старт: оценка на LIBERO

1. Окружение и чекпоинты — см. [docs/ORIGINAL_README.md](docs/ORIGINAL_README.md) (раздел Environment Setup).
2. Базовый rollout с логированием метрик:

```bash
cd model && source .venv/bin/activate
cd ../eval/libero

python run.py \
  --output_dir ../../eval_outputs/libero_spatial/baseline/my_run \
  --num_tasks 10 --episodes_per_task 10 \
  --trace_metrics
```

3. Inference со стратегией выбора чанка (3 кандидата):

```bash
python run.py \
  --output_dir ../../eval_outputs/libero_spatial/strategy_runs/my_select3 \
  --regen_strategy catboost_select \
  --regen_model_path ../../artifacts/libero_spatial/v2/catboost_chunk_regen.cbm \
  --regen_num_candidates 3 \
  --max_control_steps 120
```

Параметры `regen_strategy`: `none`, `threshold`, `catboost_select`, `action_dynamic`, `hybrid_select_action_dynamic` — подробнее в [docs/REPORT.md](docs/REPORT.md).

---

## Где что искать

| Задача | Путь |
|--------|------|
| Запуск eval, regen-логика | `eval/libero/run.py` |
| Формат метрик chunk/action | `docs/METRICS.md` |
| Базовый прогон 100 эп. (69% success) | `eval_outputs/libero_spatial/baseline/trace_100ep/` |
| KS/JSD анализ распределений | `.../baseline/trace_100ep/metrics/chunk_ks_analysis/`, `chunk_jsd_analysis/` |
| CatBoost chunk/action v2 | `artifacts/libero_spatial/v2/` |
| Val split 70/30 | `artifacts/libero_spatial/v2/episode_split_70_30.json` |
| Прогоны стратегий | `eval_outputs/libero_spatial/strategy_runs/` |
| Переобучить CatBoost | `scripts/training/` |
| Пересчитать KS | `scripts/analysis/compute_chunk_metric_ks.py` |

---

## Примечания

- Для **неуспешных** эпизодов метрики в trace обрезаются до **6 с** (120 шагов @ 20 Hz); rollout по умолчанию тоже ограничен `max_control_steps=120`.
- Видео эпизодов сохраняются в `eval_outputs/.../videos/` без обрезки по времени.
- Ноутбуки в `notebooks/` могут содержать старые абсолютные пути — актуальные пути см. в таблице выше и в `experiments/README.md`.
