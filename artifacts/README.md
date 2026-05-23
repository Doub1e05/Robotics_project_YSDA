# Артефакты обучения (копии для inference)

Канонические файлы также хранятся рядом с метриками базового прогона:
`eval_outputs/libero_spatial/baseline/trace_100ep/metrics/...`

## libero_spatial/v2/

| Файл | Назначение |
|------|------------|
| `catboost_chunk_regen.cbm` | Chunk-level P(failure) — стратегии `catboost_select`, `hybrid` |
| `catboost_action_regen.cbm` | Action-level P(failure) — `action_dynamic`, `hybrid` |
| `episode_split_70_30.json` | Фиксированный train/val split (70/30 эпизодов) |
| `val_failure_eval_episodes.txt` | 10 failure-эпизодов из val для быстрых тестов |

Пример запуска:

```bash
python eval/libero/run.py \
  --regen_strategy hybrid_select_action_dynamic \
  --regen_model_path artifacts/libero_spatial/v2/catboost_chunk_regen.cbm \
  --action_regen_model_path artifacts/libero_spatial/v2/catboost_action_regen.cbm
```
