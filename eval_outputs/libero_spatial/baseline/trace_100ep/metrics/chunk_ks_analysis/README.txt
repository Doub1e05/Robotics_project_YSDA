chunk_metric_ks.csv — полная таблица KS-тестов (metric × chunk_id).
  ks_statistic: статистика Kolmogorov–Smirnov (чем больше, тем сильнее расхождение CDF).
  p_value: p-value двустороннего two-sample KS test (scipy.stats.ks_2samp).
  significant_at_alpha: True если p_value < 0.05.
  rank_within_chunk / rank_global: ранг по возрастанию p_value (1 = наиболее значимо).

chunk_metric_ks_significant.csv — только значимые пары (p < 0.05).
chunk_metric_ks_significant_summary.md — удобный markdown-отчёт по chunk и по метрикам.
chunk_metric_ks_significant.json — тот же список значимых пар в JSON.
chunk_metric_ks_pvalue_pivot.csv — pivot: p_value[metric, chunk_id].
chunk_metric_ks_significant_pivot.csv — pivot: True/False значимость.
