# Каталог экспериментов (eval_outputs)

Все прогоны лежат под `eval_outputs/libero_spatial/`. Имена папок короткие; полные логи — внутри `metrics/`, `videos/`, `run.log`.

## Baseline (100 эпизодов, LIBERO Spatial)

| Папка | Описание | Success |
|-------|----------|---------|
| [baseline/trace_100ep](../eval_outputs/libero_spatial/baseline/trace_100ep/) | 10 задач × 10 эп., trace-метрики, GPU 3 | **69/100 (69%)** |
| [baseline/modelonly_100ep](../eval_outputs/libero_spatial/baseline/modelonly_100ep/) | Ранний прогон без trace | — |
| [baseline/modelonly_3tasks](../eval_outputs/libero_spatial/baseline/modelonly_3tasks/) | Отладка на 3 задачах | — |

**Анализ на trace_100ep:**
- `metrics/chunk_ks_analysis/` — KS success vs failure
- `metrics/chunk_jsd_analysis/` — JSD
- `metrics/chunk_distribution_plots/` — графики
- `metrics/chunk_regeneration_classifiers_v2_70_30/` — CatBoost chunk + split
- `metrics/action_regeneration_classifiers_v2_70_30/` — CatBoost action

## Strategy runs (стратегии inference)

| Папка | `regen_strategy` | Эпизоды | Примечание |
|-------|------------------|---------|------------|
| [strategy_runs/catboost_select3_v2_val10fail_6sec](../eval_outputs/libero_spatial/strategy_runs/catboost_select3_v2_val10fail_6sec/) | `catboost_select` | 10 val-failure | 6 с, 3 кандидата |
| [strategy_runs/action_dynamic_v2_val10fail_6sec](../eval_outputs/libero_spatial/strategy_runs/action_dynamic_v2_val10fail_6sec/) | `action_dynamic` | 10 val-failure | динам. горизонт 3–8 |
| [strategy_runs/hybrid_v2_val10fail_6sec](../eval_outputs/libero_spatial/strategy_runs/hybrid_v2_val10fail_6sec/) | `hybrid_select_action_dynamic` | 10 val-failure | select + dynamic |

Список val-failure эпизодов: `artifacts/libero_spatial/v2/val_failure_eval_episodes.txt`.

## Pilot runs (короткие прогоны)

| Папка | Описание |
|-------|----------|
| [pilot_runs/catboost_select3_6eps_6sec](../eval_outputs/libero_spatial/pilot_runs/catboost_select3_6eps_6sec/) | 6 эпизодов, select×3 |
| [pilot_runs/threshold060_6eps_6sec](../eval_outputs/libero_spatial/pilot_runs/threshold060_6eps_6sec/) | regen по порогу 0.6 |
| [pilot_runs/threshold070_6eps_6sec](../eval_outputs/libero_spatial/pilot_runs/threshold070_6eps_6sec/) | regen по порогу 0.7 |
| [pilot_runs/threshold038_6eps](../eval_outputs/libero_spatial/pilot_runs/threshold038_6eps/) | порог 0.38 |

## Other

| Папка | Описание |
|-------|----------|
| [other/libero_90_19tasks_10eps](../eval_outputs/libero_spatial/other/libero_90_19tasks_10eps/) | LIBERO-90, 19 задач × 10 эп. |

## Обученные модели (копии)

`artifacts/libero_spatial/v2/` — CatBoost v2, split, val_failure list (удобно для `--regen_model_path`).

Подробный отчёт: [docs/REPORT.md](../docs/REPORT.md).
