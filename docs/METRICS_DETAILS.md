# Метрики mimic-video (LIBERO eval): подробное описание

Этот документ объясняет, **как именно считаются и записываются метрики** в наших прогонах `mimic-video/eval/libero/run.py`.

Короткая справка по форматам файлов: [docs/METRICS.md](METRICS.md).  
Отчёт по экспериментам (что сравнивали и зачем): [docs/REPORT.md](REPORT.md).

---

## 1. На каких уровнях считаются метрики

В коде есть три «уровня» данных:

- **Episode (эпизод)**: один rollout в среде (task_id + episode_idx).
- **Chunk (чанк)**: один запрос к модели (одна генерация `(60,10)` действий).
- **Action (действие)**: один выполненный control-step в среде (часть предсказанного чанка).

### Как это связано во времени

- Каждые `vam_num_execute_actions` шагов (или чаще для динамических стратегий) политика делает **query** к модели и получает **чанк действий**.
- Затем политика «выдаёт» в среду действия из буфера чанка по одному.
- В трейсах это отражается так:
  - при каждом **query** создаётся запись `chunk` в `episode_traces.jsonl`;
  - каждый выполненный step добавляется как `action` внутрь последнего `chunk`.

---

## 2. Файлы метрик и что в них лежит

Все файлы пишутся в `.../metrics/`.

### 2.1. `episode_traces.jsonl` — источник истины

Одна строка = один эпизод. Структура:

- `meta`: служебная информация по эпизоду (success, причины остановки, лимиты по шагам, счётчики)
- `chunks`: список чанков; каждый чанк содержит:
  - `query_timestep`, `query_latency_sec`
  - `metrics`: chunk-метрики (см. ниже)
  - `actions`: список выполненных действий (action-метрики)

Именно из этого файла дальше делаются плоские таблицы CSV/JSONL.

### 2.2. `chunk_metrics.(csv|jsonl)`

Плоская таблица: **одна строка на чанк** (на один запрос к модели).

В строке есть:
- мета по эпизоду (`task_id`, `episode_idx`, `success`, …)
- индекс чанка (`chunk_id`, `inference_step_idx`, `query_timestep`)
- поля стратегии (например, `regen_probability`, `selected_candidate_idx`)
- и сами метрики из `chunk.metrics`

### 2.3. `action_metrics.(csv|jsonl)`

Плоская таблица: **одна строка на выполненное действие** (каждый control-step).

### 2.4. `candidate_chunk_metrics.(csv|jsonl)` (для `catboost_select`)

Если используется стратегия `catboost_select`/`hybrid`, то для каждого query мы логируем **метрики всех кандидатов** (например 3 seed’а).

Одна строка = *(чанк, кандидат)*. В строке есть:
- `candidate_idx`
- `catboost_failure_probability` и `catboost_success_probability`
- `is_selected`
- полный набор chunk-метрик кандидата (включая диагностические метрики модели)

Этот файл удобен для анализа «почему выбрали именно этот кандидат».

### 2.5. `summary.json`

Содержит только агрегаты по количествам (эпизоды/успехи/чанки/действия), без усреднения метрик.

---

## 3. Откуда берутся значения метрик

Метрики формируются как объединение двух источников:

1) **Производные от предсказанного чанка действий** (считаются нами в eval)
2) **Диагностика модели** (`model.last_diagnostics`) — вычисляется внутри пайплайна mimic-video во время inference и возвращается как словарь чисел

В коде это выглядит так:

- при query:
  - получаем `pred_actions` (shape `(60,10)`)
  - берём `diagnostics = model.last_diagnostics`
  - считаем `_chunk_action_metrics(actions_np)`
  - добавляем `_clean_model_metrics(diagnostics)` (фильтр: только finite числа)

---

## 4. Метрики, которые считаются напрямую из действий (реальные формулы)

Эти метрики **гарантированно определены** (потому что считаются в `eval/libero/run.py`).

### 4.1. Chunk-level: `_chunk_action_metrics(raw_chunk)`

Пусть `chunk` — массив формы `(T, 10)` (обычно `T=60`):

- **`predicted_action_count`**: \(T\)
- **`chunk_action_variance`**: дисперсия по всем элементам `chunk[:, :9]` (без gripper), т.е.
  \[
  \mathrm{Var}(\text{flatten}(chunk[:, :9]))
  \]
- **`chunk_action_delta_norm_mean`**:
  - сначала считаем разности между соседними действиями по continuous-части:
    \[
    \Delta_t = chunk[t, :9] - chunk[t-1, :9]
    \]
  - затем берём нормы \(\|\Delta_t\|_2\) и усредняем по \(t\)
- **`chunk_action_delta_norm_max`**: максимум \(\|\Delta_t\|_2\)
- **`chunk_gripper_switches`**: число смен знака у `sign(chunk[:,9])` между соседними шагами

Интерпретация:
- большие `*_delta_norm_*` часто означают «рывки»/нестабильность в планируемых действиях;
- `chunk_gripper_switches` полезен для выявления «дребезга» команд захвата.

### 4.2. Action-level: `_action_metrics(raw_action, previous_raw_action)`

Пусть `raw_action` — 10D действие модели.

Мы сохраняем:
- **`model_action`**: полный 10D вектор
- **`model_delta_position`**: первые 3 компоненты
- **`model_rotation_6d`**: компоненты 3:9
- **`model_gripper_command`**: компонент 9
- **`model_gripper_sign`**: `sign(model_gripper_command)`
- **`model_action_norm`**: \(\|raw_action[:9]\|_2\)
- **`model_translation_norm`**: \(\|raw_action[:3]\|_2\)
- **`model_rotation6d_norm`**: \(\|raw_action[3:9]\|_2\)
- **`model_action_delta_norm`**: \(\|raw_action[:9] - previous_raw_action[:9]\|_2\) (для первого действия — `NaN`)

Важно: это метрики **модельного действия**, до преобразования в 7D действие среды.

---

## 5. Диагностические метрики модели (`model.last_diagnostics`)

Большая часть «интересных» метрик (типа `flow_prediction_error`, `plan_drift`, `video_latent_*`, `sampling_stability`, …) приходит из `model.last_diagnostics`.

### Что важно понимать

- Эти значения **не пересчитываются** нами вручную в `run.py`. Мы просто:
  - берём словарь `last_diagnostics` у пайплайна;
  - оставляем только конечные числа (`_clean_model_metrics`);
  - записываем их в `chunk.metrics`.
- Поэтому «точные формулы» для них живут **внутри реализации mimic-video/Cosmos-Predict2**.
- Для практического анализа (KS/JSD/CatBoost) этого достаточно, потому что нам важны:
  - стабильность и воспроизводимость метрик в рамках одной версии кода,
  - различимость распределений success/failure.

### Список признаков, использованных в chunk CatBoost

Он зафиксирован в:

`eval_outputs/libero_spatial/baseline/trace_100ep/metrics/chunk_regeneration_classifiers_v2_70_30/catboost_chunk_regen_features.json`

Там 36 признаков, включая 5 вычисляемых из действий (`chunk_action_*`, `chunk_gripper_switches`, `query_latency_sec`) и набор диагностик (`flow_prediction_error`, `plan_drift`, `video_latent_*`, …).

---

## 6. Метрики CatBoost и выбор кандидата

### 6.1. Что предсказывает chunk CatBoost

Chunk CatBoost обучен на метке:

- `needs_regeneration = 1` для failure-эпизодов (`success == False`)
- `needs_regeneration = 0` для success-эпизодов

То есть `predict_proba(...)[1]` интерпретируется как:

- **`catboost_failure_probability`** = \(P(\text{failure})\)
- **`catboost_success_probability`** = \(1 - P(\text{failure})\)

### 6.2. Стратегия `catboost_select` (3 кандидата)

Для одного query генерируем 3 кандидата (seed’ы `base_seed + 0/1/2`), считаем \(P(\text{failure})\) для каждого и выбираем:

- `selected_candidate_idx = argmin P(failure)`

Именно поэтому в логах «лучший кандидат» = минимальная вероятность неуспеха.

---

## 7. Обрезка метрик для failure-эпизодов (важно для анализа)

Чтобы «длинные» timeout-эпизоды не доминировали по числу строк, для failure-эпизодов метрики обрезаются:

- сохраняем только действия/чанки с `timestep < 120` (6 секунд при 20 Hz)

Видео при этом может сохраняться полностью (если rollout шёл дольше), но **табличные метрики** для failure будут укорочены.

Это влияет на любые сравнения распределений, поэтому в анализе нужно помнить: failure-метрики отражают «первые 6 секунд поведения».

---

## 8. Практические подсказки для анализа

- Хотите сравнить метрики success vs failure:
  - используйте `chunk_metrics.csv` и фильтр по `success`.
- Хотите понять, почему select выбрал именно кандидата:
  - используйте `candidate_chunk_metrics.csv` и сравнивайте `is_selected=1` vs `0`.
- Хотите смотреть динамику внутри эпизода:
  - берите `episode_traces.jsonl` и раскладывайте `chunks[*].actions[*]`.

