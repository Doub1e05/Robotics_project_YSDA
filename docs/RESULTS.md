# Experiment Registry

Актуальный срез результатов на **2026-09-11**. Источник каждого числа —
завершённый `summary.json`; незавершённые прогоны приведены отдельно и не
участвуют в сравнении.

## Статистический протокол

- В таблицах указан episode-level success rate и **двусторонний 95% Wilson CI**
  для биномиальной доли.
- CI характеризует неопределённость одного набора эпизодов, а не разброс между
  model seeds.
- Для запусков на одинаковых эпизодах дополнительно применим парный exact
  McNemar/binomial test. Перекрытие двух отдельных CI не является тестом
  значимости разности.
- Нельзя объединять 8-task INT-ACT, его 16-task расширение, обычный LIBERO,
  LIBERO-Pro и LIBERO-Plus: это разные task distributions.

Wilson interval вычисляется как

```text
center = (p + z²/(2n)) / (1 + z²/n)
half   = z * sqrt(p(1-p)/n + z²/(4n²)) / (1 + z²/n)
z      = 1.959963984540054
```

## SIMPLER-Bridge — 4 задачи × 24 эпизода

### MIMIC-Video Bridge

| Метод | Candidate seeds | Success | SR (95% CI) |
|---|---:|---:|---:|
| Baseline | single seed | 45/96 | 46.88% [37.21, 56.78] |
| Consensus medoid | `[0,1,2]` | 52/96 | 54.17% [44.23, 63.78] |
| Decoder action-token medoid | `[2,996,997]` | 42/96 | 43.75% [34.26, 53.72] |

Summaries:

- `eval_outputs/simpler_bridge/ftcosmos_stop0_full96_20260901_194203/summary.json`
- `eval_outputs/simpler_bridge/ftcosmos_stop0_consensus_medoid_only_k3_fixedseeds012_directresult_full96_20260902_090800/summary.json`
- `eval_outputs/simpler_bridge/latent_medoid_four_selectors_fixedseeds_2_996_997_20260907_200056/decoder_action_token_medoid/summary.json`

### GR00T N1.7 Bridge

| Метод | Seed / candidates | Success | SR (95% CI) |
|---|---:|---:|---:|
| Baseline | `0` | 51/96 | 53.12% [43.22, 62.79] |
| Baseline | `1` | 49/96 | 51.04% [41.20, 60.81] |
| Baseline | `2` | 51/96 | 53.12% [43.22, 62.79] |
| Baseline | `3` | 50/96 | 52.08% [42.20, 61.80] |
| Consensus medoid | `[1,999,998]` | 39/96 | 40.62% [31.35, 50.63] |
| Consensus medoid | `[2,997,996]` | 41/96 | 42.71% [33.28, 52.70] |
| Consensus medoid | `[3,995,994]` | 58/96 | 60.42% [50.42, 69.62] |
| Decoder action-token medoid | `[1,999,998]` | 66/96 | 68.75% [58.91, 77.15] |
| Decoder action-token medoid | `[2,997,996]` | 46/96 | 47.92% [38.20, 57.80] |
| Decoder action-token medoid | `[3,995,994]` | 51/96 | 53.12% [43.22, 62.79] |

Парные episode-level проверки decoder medoid против соответствующего baseline:
seed 1 `p=0.0076`, seed 2 `p=0.473`, seed 3 `p=1.000`. Единственный большой
локальный выигрыш пока не воспроизводится между seeds.

## INT-ACT Object OOD — официальный набор 8 × 24

### MIMIC-Video Bridge

Baseline `[0]`, `[1]`, `[2]` и consensus `[0,1,2]` пересчитаны только по восьми
официальным OOD-задачам из 16-task summaries. Остальные consensus summaries
сразу собраны на official 8-task subset.

| Метод | Seed / candidates | Success | SR (95% CI) |
|---|---:|---:|---:|
| Baseline | `0` | 73/192 | 38.02% [31.45, 45.06] |
| Baseline | `1` | 33/192 | 17.19% [12.51, 23.15] |
| Baseline | `2` | 9/192 | 4.69% [2.49, 8.67] |
| Consensus medoid | `[0,1,2]` | 74/192 | 38.54% [31.95, 45.59] |
| Consensus medoid | `[2,996,997]` | 50/192 | 26.04% [20.35, 32.68] |
| Consensus medoid | `[3,998,999]` | 39/192 | 20.31% [15.23, 26.56] |

Decoder action-token medoid `[0,1,2]` выполняется на GPU 4; частичный результат
не публикуется как benchmark number.

### GR00T N1.7 Bridge

| Метод | Seed / candidates | Success | SR (95% CI) |
|---|---:|---:|---:|
| Baseline | `1` | 74/192 | 38.54% [31.95, 45.59] |
| Baseline | `2` | 53/192 | 27.60% [21.77, 34.32] |
| Baseline | `3` | 76/192 | 39.58% [32.94, 46.64] |
| Consensus medoid | `[1,999,998]` | 74/192 | 38.54% [31.95, 45.59] |
| Consensus medoid | `[2,997,996]` | 63/192 | 32.81% [26.57, 39.73] |
| Consensus medoid | `[3,995,994]` | 78/192 | 40.62% [33.93, 47.69] |
| Decoder action-token medoid | `[1,999,998]` | 85/192 | 44.27% [37.43, 51.34] |

Парные p-values consensus против baseline: `1.000`, `0.203`, `0.871` для
primary seeds 1, 2, 3. Для decoder `[1,999,998]` против baseline seed 1:
`p=0.161`.

## Обычный LIBERO Spatial — 10 задач × 10 эпизодов

| Модель | Метод | Success | SR (95% CI) |
|---|---|---:|---:|
| MIMIC-Video | Baseline | 69/100 | 69.00% [59.37, 77.22] |
| MIMIC-Video | Consensus medoid | 75/100 | 75.00% [65.70, 82.45] |
| GR00T N1.7 LIBERO | Decoder action-token medoid `[2,997,996]` | 100/100 | 100.00% [96.30, 100.00] |

MIMIC decoder medoid `[0,1,2]` выполняется на GPU 0. Первый ошибочно
унаследовавший GPU 1 процесс остановлен после 2/100 и исключён из результатов;
исправленный launcher использует изолированные переменные `MIMIC_GPU` и
`MIMIC_SESSION`. Вторая GR00T decoder-конфигурация `[3,995,994]` выполняется на
GPU 2. Их CI будет добавлен только после полного завершения.

## LIBERO-Pro Spatial — 4 OOD-среза × 10 задач × 10 эпизодов

| Модель | Метод | Seed / candidates | Success | SR (95% CI) |
|---|---|---:|---:|---:|
| GR00T N1.7 LIBERO | Baseline | `1` | 246/400 | 61.50% [56.64, 66.14] |
| GR00T N1.7 LIBERO | Consensus medoid | `[1,999,998]` | 248/400 | 62.00% [57.15, 66.62] |

Парный тест: 5 эпизодов улучшились, 3 ухудшились, `p=0.727`; подтверждённого
прироста нет. Decoder medoid и полные MIMIC-Video строки пока отсутствуют.

## LIBERO-Plus Spatial

Полного сопоставимого результата пока нет. Текущий GR00T baseline/consensus run
использует фиксированный first-100 OOD slice. Частичный baseline `52/52` нельзя
интерпретировать как `100%` benchmark SR: выполнение ещё продолжается.

## Текущий вывод

- **Consensus medoid:** устойчивого общего улучшения пока нет; эффект меняется
  между seeds и benchmarks.
- **Decoder action-token medoid:** есть сильный положительный сигнал у GR00T на
  SIMPLER seed 1, но он не воспроизводится на seeds 2 и 3; данных по LIBERO пока
  недостаточно.
- Для утверждения о качестве нужны одинаковые task/episode sets, несколько
  заранее заданных seed-групп и парный анализ, а не только сравнение отдельных CI.
