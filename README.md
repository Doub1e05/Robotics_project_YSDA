# MIMIC-Video Planning Experiments

Исследовательский репозиторий для test-time planning поверх
[mimic-video](https://arxiv.org/abs/2512.15692), Video-Action Model (VAM) на
Cosmos-Predict2. Проект сохраняет upstream-код модели и добавляет воспроизводимые
запуски, planning-стратегии и анализ для LIBERO, SIMPLER-Bridge и INT-ACT.

> Веса, датасеты, виртуальные окружения, rollout-видео и подробные diagnostics
> намеренно не хранятся в Git. Полный список внешних зависимостей и команд
> загрузки находится в [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## С чего начать

1. Создайте окружение и скачайте официальные MIMIC-Video checkpoint'ы по
   [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).
2. Для SIMPLER/INT-ACT дополнительно установите SIMPLER-Bridge зависимости и
   скачайте T5-11B. Это также описано в документе выше.
3. Запустите короткий smoke test или полный benchmark из
   [docs/EVALUATION.md](docs/EVALUATION.md).
4. Сверьте полученный `summary.json` с состоянием экспериментов в
   [docs/RESULTS.md](docs/RESULTS.md).

## Карта репозитория

| Путь | Назначение |
|---|---|
| `model/` | upstream Cosmos-Predict2, Video2World и World2Action пайплайны |
| `eval/libero/` | LIBERO evaluator и planning-стратегии для него |
| `eval/bridge/SimplerEnv/` | SIMPLER-Bridge evaluator и VAM wrapper |
| `eval/int-act/` | INT-ACT benchmark и его pinned SIMPLER/ManiSkill2 dependencies |
| `scripts/eval/` | поддерживаемые launchers, collectors и smoke tests |
| `eval/libero/consensus_medoid.py` | control-aware distance и consensus-medoid selector |
| `docs/` | протоколы, запуск, результаты и исходная upstream документация |
| `artifacts/` | компактные CatBoost artifacts для LIBERO экспериментов |
| `eval_outputs/` | локальные результаты; большие артефакты игнорируются Git |
| `paper/` | черновик статьи |

## Planning-методы

| Метод | Код | Включение |
|---|---|---|
| Baseline MIMIC-Video | `eval/libero/run.py`, `video_action_model.py` | один sampled chunk на query |
| Consensus medoid | `eval/libero/consensus_medoid.py` | `--vam-consensus-medoid-only --vam-consensus-num-candidates K` |
| Fixed candidate seeds | `video_action_model.py` | `MIMIC_VIDEO_CANDIDATE_SEEDS=2,996,997` |
| Action/hidden rank fusion | `video_action_model.py` | `--vam-consensus-rank-fusion` |
| Latent diagnostics | `video2world2action.py` и Bridge evaluator | `--vam-diagnostics-mode all --vam-save-diagnostics` |
| LIBERO CatBoost selectors | `eval/libero/run.py`, `scripts/training/` | `--regen_strategy ...` |

`K` означает число candidate action chunks на одном model query. `H` или
`--vam-num-execute-actions` означает длину префикса выбранного chunk, который
исполняется до следующего query. Для текущих SIMPLER/INT-ACT запусков обычно
используются `K=3`, `H=5` и `stop_video_denoising_step=0`.

Подробное описание control-aware consensus-medoid находится в
[docs/CONSENSUS_MEDOID_ONLY.md](docs/CONSENSUS_MEDOID_ONLY.md). Это baseline
selector для экспериментов, а не заявление о его методической новизне:
близкий подход KeyStone уже использует multi-sample action-space medoid
selection.

## Поддерживаемые entry points

| Цель | Команда / launcher |
|---|---|
| LIBERO Spatial baseline | `scripts/eval/run_baseline_libero_spatial_one_rank.sh` |
| LIBERO consensus-only | `scripts/eval/launch_libero_spatial_consensus_only_100ep_tmux.sh` |
| SIMPLER-Bridge baseline | `scripts/eval/launch_mimic_simpler_bridge_ftcosmos_full_tmux.sh` |
| SIMPLER-Bridge consensus | `scripts/eval/launch_mimic_simpler_bridge_ftcosmos_consensus_medoid_only_full_tmux.sh` |
| SIMPLER-Bridge latent diagnostic subset | `scripts/eval/launch_mimic_simpler_bridge_latent_diagnostics_30_tmux.sh` |
| INT-ACT 16-task baseline | `scripts/eval/launch_mimic_intact_object_ood_3seed_4gpu_tmux.sh` |
| INT-ACT 8-task consensus with custom seeds | `scripts/eval/launch_mimic_intact_object_ood_consensus_seeds_2_996_997_gpu5_tmux.sh` |

Все длинные launcher'ы создают `tmux`-сессию и печатают её имя вместе с
путём к output directory. Они сохраняют `summary.json` и `summary.tsv` после
окончания, а `resume` launcher'ы используют уже существующие video files для
пропуска завершённых `episode_id`.

## Результаты и ограничения

Актуальные компактные summaries, их протоколы и известные ограничения
собраны в [docs/RESULTS.md](docs/RESULTS.md). Не сравнивайте запуски с разными
наборами задач, количеством rollout'ов или candidate seeds как прямые
репликации друг друга.

## Upstream

- [Оригинальный README mimic-video](docs/ORIGINAL_README.md)
- [Модель и training configuration](MODEL.md)
- [Формат данных](DATA.md)
- [Лицензия](LICENSE)
