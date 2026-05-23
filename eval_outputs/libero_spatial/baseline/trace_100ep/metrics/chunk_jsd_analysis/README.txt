chunk_metric_jsd.csv — основная таблица (long format).
  jsd: Jensen–Shannon divergence (base 2), 0 = одинаковые распределения, 1 = максимально разные.
  js_distance: sqrt(jsd), метрика из scipy.spatial.distance.jensenshannon.
  rank_within_chunk: ранг метрики внутри chunk_id по jsd (1 = самая различимая).
  rank_global: глобальный ранг по jsd.

chunk_metric_jsd_pivot.csv — wide: строки=метрики, столбцы=chunk_id.
chunk_metric_jsd_by_jsd.csv — все пары (metric, chunk) отсортированы по убыванию jsd.
chunk_metric_jsd_top10_per_chunk.csv — top-10 метрик для каждого chunk_id.
