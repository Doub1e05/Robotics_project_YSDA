# Значимые отличия success vs failure (Kolmogorov–Smirnov)

Критерий: `p_value < 0.05` (двусторонний two-sample KS test).
Всего пар (metric, chunk_id): 936
Значимых пар: 475

## chunk_id = 0 (23 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `chunk_action_delta_norm_mean` | 9.12729e-11 | 0.7036 | 69 | 31 | 0.0181854 |
| `flow_prediction_error` | 2.06764e-09 | 0.660122 | 69 | 31 | 0.0148771 |
| `action_chunk_temporal_consistency` | 2.06764e-09 | 0.660122 | 69 | 31 | -0.0117623 |
| `plan_drift` | 2.06764e-09 | 0.660122 | 69 | 31 | 0.877751 |
| `action_entropy` | 8.43713e-08 | 0.602151 | 69 | 31 | 0.0078076 |
| `video_action_mutual_information` | 8.43713e-08 | 0.602151 | 69 | 31 | -0.00757301 |
| `mlp_activation_mean` | 1.08572e-05 | 0.513324 | 69 | 31 | 0.0135687 |
| `task_semantic_alignment` | 5.7004e-05 | 0.477793 | 69 | 31 | -0.0128772 |
| `video_latent_variance_mean` | 0.000153507 | 0.455353 | 69 | 31 | -0.0537181 |
| `representation_drift_rate` | 0.000212809 | 0.447405 | 69 | 31 | -0.482131 |
| `decoder_hidden_norm_mean` | 0.00073302 | 0.41655 | 69 | 31 | 0.0204899 |
| `object_interaction_confidence` | 0.00073302 | 0.41655 | 69 | 31 | 0.00234003 |
| `video_latent_norm_std` | 0.00119437 | 0.40346 | 69 | 31 | 0.2081 |
| `sampling_stability` | 0.00119437 | 0.40346 | 69 | 31 | -0.00145942 |
| `chunk_action_variance` | 0.00150399 | 0.397382 | 69 | 31 | 0.00552951 |
| `video_latent_entropy` | 0.00336432 | 0.374474 | 69 | 31 | -0.000179537 |
| `decoder_hidden_norm_std` | 0.017237 | 0.323048 | 69 | 31 | 0.00506825 |
| `mlp_sparsity` | 0.0237454 | 0.311828 | 69 | 31 | -0.00508727 |
| `video_action_alignment_score` | 0.0294441 | 0.30388 | 69 | 31 | 0.00105163 |
| `video_action_cosine_similarity` | 0.0294441 | 0.30388 | 69 | 31 | 0.00105163 |
| `goal_latent_distance` | 0.030394 | 0.302478 | 69 | 31 | -0.361175 |
| `latent_success_alignment` | 0.0375546 | 0.29453 | 69 | 31 | 0.00143548 |
| `residual_stream_norm` | 0.0409965 | 0.291258 | 69 | 31 | 0.568629 |

## chunk_id = 1 (21 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `chunk_action_delta_norm_mean` | 4.02803e-11 | 0.71482 | 69 | 31 | 0.0208336 |
| `flow_prediction_error` | 2.64839e-09 | 0.656849 | 69 | 31 | 0.0168601 |
| `action_chunk_temporal_consistency` | 2.64839e-09 | 0.656849 | 69 | 31 | -0.0132948 |
| `plan_drift` | 2.64839e-09 | 0.656849 | 69 | 31 | 0.994749 |
| `action_entropy` | 1.48547e-06 | 0.552127 | 69 | 31 | 0.00793696 |
| `video_action_mutual_information` | 1.48547e-06 | 0.552127 | 69 | 31 | -0.00768373 |
| `decoder_hidden_norm_std` | 2.64661e-06 | 0.540907 | 69 | 31 | 0.00512518 |
| `mlp_activation_mean` | 1.08572e-05 | 0.513324 | 69 | 31 | 0.0142718 |
| `task_semantic_alignment` | 3.16899e-05 | 0.490416 | 69 | 31 | -0.0129321 |
| `video_latent_entropy` | 4.55451e-05 | 0.482468 | 69 | 31 | -0.000198847 |
| `video_latent_norm_std` | 0.000364978 | 0.434315 | 69 | 31 | 0.240292 |
| `sampling_stability` | 0.000364978 | 0.434315 | 69 | 31 | -0.0016907 |
| `video_latent_variance_mean` | 0.000516559 | 0.424965 | 69 | 31 | -0.0419737 |
| `chunk_action_delta_norm_max` | 0.000535769 | 0.424497 | 69 | 31 | 0.0680552 |
| `decoder_hidden_norm_mean` | 0.00150399 | 0.397382 | 69 | 31 | 0.0202997 |
| `object_interaction_confidence` | 0.00150399 | 0.397382 | 69 | 31 | 0.0022932 |
| `chunk_action_variance` | 0.00150399 | 0.397382 | 69 | 31 | 0.00639088 |
| `attention_sparsity` | 0.00442058 | 0.366526 | 69 | 31 | 2.94934e-05 |
| `representation_drift_rate` | 0.0184059 | 0.320243 | 69 | 31 | -0.315508 |
| `goal_latent_distance` | 0.0222928 | 0.313698 | 69 | 31 | -0.258269 |
| `mlp_sparsity` | 0.0420566 | 0.29079 | 69 | 31 | -0.00516752 |

## chunk_id = 2 (24 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `chunk_action_delta_norm_mean` | 1.20591e-10 | 0.700327 | 69 | 31 | 0.0234972 |
| `flow_prediction_error` | 7.56234e-10 | 0.674614 | 69 | 31 | 0.020988 |
| `action_chunk_temporal_consistency` | 7.56234e-10 | 0.674614 | 69 | 31 | -0.0164198 |
| `plan_drift` | 7.56234e-10 | 0.674614 | 69 | 31 | 1.23829 |
| `decoder_hidden_norm_std` | 3.20021e-05 | 0.489481 | 69 | 31 | 0.00645124 |
| `task_semantic_alignment` | 3.68998e-05 | 0.487144 | 69 | 31 | -0.0134505 |
| `action_entropy` | 6.10788e-05 | 0.476391 | 69 | 31 | 0.00635736 |
| `video_action_mutual_information` | 6.10788e-05 | 0.476391 | 69 | 31 | -0.00612983 |
| `chunk_action_delta_norm_max` | 6.56037e-05 | 0.474521 | 69 | 31 | 0.0830958 |
| `video_latent_norm_std` | 0.000176003 | 0.45208 | 69 | 31 | 0.229712 |
| `mlp_activation_mean` | 0.000176003 | 0.45208 | 69 | 31 | 0.0109735 |
| `sampling_stability` | 0.000176003 | 0.45208 | 69 | 31 | -0.00162098 |
| `video_latent_entropy` | 0.000689234 | 0.417952 | 69 | 31 | -0.000187151 |
| `decoder_hidden_norm_mean` | 0.00462309 | 0.365124 | 69 | 31 | 0.0162653 |
| `object_interaction_confidence` | 0.00462309 | 0.365124 | 69 | 31 | 0.00183215 |
| `representation_drift_rate` | 0.0104696 | 0.339411 | 69 | 31 | -0.274033 |
| `chunk_action_variance` | 0.0151138 | 0.326788 | 69 | 31 | 0.00647353 |
| `attention_sparsity` | 0.0156449 | 0.326321 | 69 | 31 | 2.53044e-05 |
| `video_latent_delta_l2_std` | 0.0161695 | 0.324918 | 69 | 31 | 0.24012 |
| `action_chunk_smoothness` | 0.0252876 | 0.309023 | 69 | 31 | 0.000423825 |
| `mlp_sparsity` | 0.0272321 | 0.306685 | 69 | 31 | -0.00584307 |
| `video_latent_variance_mean` | 0.0277399 | 0.30575 | 69 | 31 | -0.0341507 |
| `action_uncertainty` | 0.0459947 | 0.286583 | 69 | 31 | 0.00428009 |
| `goal_latent_distance` | 0.048704 | 0.284712 | 69 | 31 | -0.279467 |

## chunk_id = 3 (22 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 5.47101e-09 | 0.645629 | 69 | 31 | 0.021882 |
| `action_chunk_temporal_consistency` | 5.47101e-09 | 0.645629 | 69 | 31 | -0.0170141 |
| `plan_drift` | 5.47101e-09 | 0.645629 | 69 | 31 | 1.29104 |
| `chunk_action_delta_norm_mean` | 5.47101e-09 | 0.645629 | 69 | 31 | 0.0221643 |
| `task_semantic_alignment` | 3.68998e-05 | 0.487144 | 69 | 31 | -0.0137471 |
| `mlp_activation_mean` | 4.28241e-05 | 0.484338 | 69 | 31 | 0.0082955 |
| `action_entropy` | 5.7004e-05 | 0.477793 | 69 | 31 | 0.00489687 |
| `video_action_mutual_information` | 5.7004e-05 | 0.477793 | 69 | 31 | -0.00471457 |
| `chunk_action_delta_norm_max` | 0.000142403 | 0.456755 | 69 | 31 | 0.0741989 |
| `decoder_hidden_norm_std` | 0.00106111 | 0.405797 | 69 | 31 | 0.00452025 |
| `decoder_hidden_norm_mean` | 0.00150399 | 0.397382 | 69 | 31 | 0.0140718 |
| `object_interaction_confidence` | 0.00150399 | 0.397382 | 69 | 31 | 0.00158011 |
| `video_latent_norm_std` | 0.0016958 | 0.394109 | 69 | 31 | 0.1691 |
| `representation_drift_rate` | 0.0016958 | 0.394109 | 69 | 31 | -0.316233 |
| `sampling_stability` | 0.0016958 | 0.394109 | 69 | 31 | -0.00119466 |
| `video_latent_entropy` | 0.00515657 | 0.361851 | 69 | 31 | -0.000152092 |
| `goal_latent_distance` | 0.0277399 | 0.30575 | 69 | 31 | -0.308888 |
| `attention_sparsity` | 0.0322672 | 0.300608 | 69 | 31 | 2.15295e-05 |
| `video_latent_variance_mean` | 0.0363799 | 0.295933 | 69 | 31 | -0.0366426 |
| `mlp_sparsity` | 0.0370504 | 0.295465 | 69 | 31 | -0.00552283 |
| `video_latent_delta_l2_std` | 0.040298 | 0.292193 | 69 | 31 | 0.17757 |
| `chunk_action_variance` | 0.0459947 | 0.286583 | 69 | 31 | 0.00728462 |

## chunk_id = 4 (19 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 2.48883e-06 | 0.542309 | 69 | 31 | 0.020421 |
| `action_chunk_temporal_consistency` | 2.48883e-06 | 0.542309 | 69 | 31 | -0.0158289 |
| `plan_drift` | 2.48883e-06 | 0.542309 | 69 | 31 | 1.20484 |
| `chunk_action_delta_norm_mean` | 2.48883e-06 | 0.542309 | 69 | 31 | 0.0206433 |
| `action_entropy` | 0.000124177 | 0.460028 | 69 | 31 | 0.00380499 |
| `task_semantic_alignment` | 0.00013367 | 0.458158 | 69 | 31 | -0.0133752 |
| `video_action_mutual_information` | 0.000201174 | 0.448808 | 69 | 31 | -0.00365445 |
| `mlp_activation_mean` | 0.000320587 | 0.437588 | 69 | 31 | 0.00761994 |
| `chunk_action_delta_norm_max` | 0.000608101 | 0.421225 | 69 | 31 | 0.0699711 |
| `decoder_hidden_norm_std` | 0.000609177 | 0.42029 | 69 | 31 | 0.00447072 |
| `attention_sparsity` | 0.00102961 | 0.4072 | 69 | 31 | 3.02728e-05 |
| `decoder_hidden_norm_mean` | 0.00150399 | 0.397382 | 69 | 31 | 0.0169794 |
| `object_interaction_confidence` | 0.00150399 | 0.397382 | 69 | 31 | 0.00190602 |
| `video_latent_entropy` | 0.00354392 | 0.373072 | 69 | 31 | -0.000128097 |
| `representation_drift_rate` | 0.00664621 | 0.353904 | 69 | 31 | -0.323147 |
| `mlp_sparsity` | 0.011178 | 0.337541 | 69 | 31 | -0.00299673 |
| `video_latent_norm_std` | 0.0178242 | 0.321646 | 69 | 31 | 0.125822 |
| `sampling_stability` | 0.0178242 | 0.321646 | 69 | 31 | -0.000889599 |
| `goal_latent_distance` | 0.0202675 | 0.316971 | 69 | 31 | -0.218128 |

## chunk_id = 5 (19 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `decoder_hidden_norm_mean` | 4.55451e-05 | 0.482468 | 69 | 31 | 0.0181958 |
| `object_interaction_confidence` | 4.55451e-05 | 0.482468 | 69 | 31 | 0.00204429 |
| `task_semantic_alignment` | 5.7004e-05 | 0.477793 | 69 | 31 | -0.0132079 |
| `chunk_action_delta_norm_max` | 0.000142403 | 0.456755 | 69 | 31 | 0.0864812 |
| `action_entropy` | 0.000201174 | 0.448808 | 69 | 31 | 0.00308472 |
| `video_latent_entropy` | 0.000414927 | 0.431043 | 69 | 31 | -0.00010626 |
| `chunk_action_delta_norm_mean` | 0.000502188 | 0.426367 | 69 | 31 | 0.0219828 |
| `attention_sparsity` | 0.000516559 | 0.424965 | 69 | 31 | 2.66683e-05 |
| `decoder_hidden_norm_std` | 0.000609177 | 0.42029 | 69 | 31 | 0.00518545 |
| `flow_prediction_error` | 0.000693772 | 0.417017 | 69 | 31 | 0.0203565 |
| `action_chunk_temporal_consistency` | 0.000693772 | 0.417017 | 69 | 31 | -0.015747 |
| `plan_drift` | 0.000693772 | 0.417017 | 69 | 31 | 1.20103 |
| `mlp_activation_mean` | 0.00073302 | 0.41655 | 69 | 31 | 0.00721788 |
| `representation_drift_rate` | 0.00102961 | 0.4072 | 69 | 31 | -0.324445 |
| `video_action_mutual_information` | 0.00112318 | 0.40533 | 69 | 31 | -0.00296052 |
| `video_action_alignment_score` | 0.00943433 | 0.342683 | 69 | 31 | 0.0010119 |
| `video_action_cosine_similarity` | 0.00943433 | 0.342683 | 69 | 31 | 0.0010119 |
| `mlp_sparsity` | 0.0101016 | 0.340813 | 69 | 31 | -0.00375643 |
| `video_latent_variance_mean` | 0.03864 | 0.293595 | 69 | 31 | -0.0321686 |

## chunk_id = 6 (19 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `task_semantic_alignment` | 5.7004e-05 | 0.477793 | 69 | 31 | -0.012776 |
| `chunk_action_delta_norm_max` | 8.72921e-05 | 0.467976 | 69 | 31 | 0.0833242 |
| `decoder_hidden_norm_mean` | 0.000100768 | 0.464703 | 69 | 31 | 0.0215183 |
| `object_interaction_confidence` | 0.000100768 | 0.464703 | 69 | 31 | 0.00241124 |
| `attention_sparsity` | 0.000117102 | 0.460496 | 69 | 31 | 2.77642e-05 |
| `representation_drift_rate` | 0.000136151 | 0.457223 | 69 | 31 | -0.367836 |
| `flow_prediction_error` | 0.000289204 | 0.439458 | 69 | 31 | 0.0261229 |
| `action_chunk_temporal_consistency` | 0.000289204 | 0.439458 | 69 | 31 | -0.0196539 |
| `plan_drift` | 0.000289204 | 0.439458 | 69 | 31 | 1.54125 |
| `action_entropy` | 0.000364978 | 0.434315 | 69 | 31 | 0.0026426 |
| `video_action_mutual_information` | 0.000647378 | 0.419822 | 69 | 31 | -0.00253621 |
| `video_latent_entropy` | 0.00073302 | 0.41655 | 69 | 31 | -9.06579e-05 |
| `mlp_activation_mean` | 0.00073302 | 0.41655 | 69 | 31 | 0.00800937 |
| `chunk_action_delta_norm_mean` | 0.00150399 | 0.397382 | 69 | 31 | 0.0221667 |
| `action_chunk_smoothness` | 0.00240311 | 0.384292 | 69 | 31 | -0.00669841 |
| `video_latent_norm_std` | 0.0189475 | 0.319776 | 69 | 31 | 0.0893314 |
| `sampling_stability` | 0.0189475 | 0.319776 | 69 | 31 | -0.00063089 |
| `decoder_hidden_norm_std` | 0.0285613 | 0.304348 | 69 | 31 | 0.00549409 |
| `instruction_attention_score` | 0.0294441 | 0.30388 | 69 | 31 | 0.00724662 |

## chunk_id = 7 (21 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `chunk_action_delta_norm_mean` | 3.77382e-05 | 0.486209 | 69 | 31 | 0.0199279 |
| `task_semantic_alignment` | 6.56037e-05 | 0.474521 | 69 | 31 | -0.0129401 |
| `chunk_action_delta_norm_max` | 7.56088e-05 | 0.471248 | 69 | 31 | 0.0719831 |
| `representation_drift_rate` | 0.000449762 | 0.428237 | 69 | 31 | -0.385661 |
| `video_latent_entropy` | 0.00190688 | 0.390837 | 69 | 31 | -0.000114029 |
| `flow_prediction_error` | 0.00198166 | 0.389434 | 69 | 31 | 0.0318976 |
| `action_chunk_temporal_consistency` | 0.00198166 | 0.389434 | 69 | 31 | -0.0241914 |
| `plan_drift` | 0.00198166 | 0.389434 | 69 | 31 | 1.88196 |
| `action_chunk_smoothness` | 0.00442058 | 0.366526 | 69 | 31 | -0.0323266 |
| `decoder_hidden_norm_mean` | 0.0046512 | 0.364656 | 69 | 31 | 0.0202732 |
| `object_interaction_confidence` | 0.0046512 | 0.364656 | 69 | 31 | 0.0022574 |
| `video_latent_norm_std` | 0.00574321 | 0.358579 | 69 | 31 | 0.115498 |
| `mlp_activation_mean` | 0.00574321 | 0.358579 | 69 | 31 | 0.00738804 |
| `sampling_stability` | 0.00574321 | 0.358579 | 69 | 31 | -0.000814904 |
| `action_entropy` | 0.00739778 | 0.350631 | 69 | 31 | 0.00213373 |
| `video_action_mutual_information` | 0.00739778 | 0.350631 | 69 | 31 | -0.0020106 |
| `attention_sparsity` | 0.010803 | 0.338008 | 69 | 31 | 2.06283e-05 |
| `instruction_attention_score` | 0.0141766 | 0.329593 | 69 | 31 | 0.00696453 |
| `action_uncertainty` | 0.0222928 | 0.313698 | 69 | 31 | 0.0019293 |
| `decoder_hidden_norm_std` | 0.0227827 | 0.31323 | 69 | 31 | 0.00502958 |
| `residual_stream_norm` | 0.0438069 | 0.28892 | 69 | 31 | 0.0927857 |

## chunk_id = 8 (13 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `task_semantic_alignment` | 3.68998e-05 | 0.487144 | 69 | 31 | -0.0133492 |
| `video_latent_norm_std` | 0.00024495 | 0.444133 | 69 | 31 | 0.17264 |
| `sampling_stability` | 0.00024495 | 0.444133 | 69 | 31 | -0.00121622 |
| `representation_drift_rate` | 0.000250162 | 0.44273 | 69 | 31 | -0.321771 |
| `video_latent_entropy` | 0.00073302 | 0.41655 | 69 | 31 | -0.000159404 |
| `video_action_alignment_score` | 0.00190688 | 0.390837 | 69 | 31 | 0.0111688 |
| `video_action_cosine_similarity` | 0.00190688 | 0.390837 | 69 | 31 | 0.0111688 |
| `chunk_action_delta_norm_max` | 0.00190688 | 0.390837 | 69 | 31 | 0.0489929 |
| `decoder_hidden_norm_mean` | 0.0025418 | 0.382422 | 69 | 31 | 0.0172272 |
| `object_interaction_confidence` | 0.0025418 | 0.382422 | 69 | 31 | 0.00190768 |
| `action_uncertainty` | 0.00595818 | 0.357176 | 69 | 31 | -0.00206921 |
| `instruction_attention_score` | 0.00871526 | 0.345489 | 69 | 31 | 0.0011031 |
| `chunk_action_variance` | 0.0136712 | 0.330061 | 69 | 31 | 0.00459301 |

## chunk_id = 9 (20 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `task_semantic_alignment` | 6.13614e-05 | 0.475923 | 69 | 31 | -0.0127869 |
| `representation_drift_rate` | 0.000136151 | 0.457223 | 69 | 31 | -0.284504 |
| `video_latent_norm_std` | 0.000571282 | 0.422627 | 69 | 31 | 0.181256 |
| `sampling_stability` | 0.000571282 | 0.422627 | 69 | 31 | -0.00127165 |
| `video_latent_entropy` | 0.00190688 | 0.390837 | 69 | 31 | -0.000166332 |
| `decoder_hidden_norm_mean` | 0.0046512 | 0.364656 | 69 | 31 | 0.0217505 |
| `object_interaction_confidence` | 0.0046512 | 0.364656 | 69 | 31 | 0.00243416 |
| `residual_stream_norm` | 0.00574729 | 0.358111 | 69 | 31 | -0.0636719 |
| `mlp_activation_mean` | 0.00943433 | 0.342683 | 69 | 31 | 0.00769721 |
| `decoder_hidden_norm_std` | 0.0164427 | 0.324451 | 69 | 31 | -0.00658984 |
| `action_uncertainty` | 0.0166935 | 0.323516 | 69 | 31 | 0.00221138 |
| `instruction_attention_score` | 0.0207915 | 0.316503 | 69 | 31 | 0.00557838 |
| `action_chunk_smoothness` | 0.0260516 | 0.308555 | 69 | 31 | 0.0477414 |
| `chunk_action_variance` | 0.0277399 | 0.30575 | 69 | 31 | 0.00801102 |
| `video_latent_delta_l2_std` | 0.040298 | 0.292193 | 69 | 31 | 0.128378 |
| `latent_success_alignment` | 0.040298 | 0.292193 | 69 | 31 | 0.000218426 |
| `action_entropy` | 0.0409965 | 0.291258 | 69 | 31 | 0.00150689 |
| `video_action_mutual_information` | 0.0409965 | 0.291258 | 69 | 31 | -0.00134159 |
| `chunk_gripper_switches` | 0.0420566 | 0.29079 | 69 | 31 | -0.451613 |
| `latent_oscillation_score` | 0.0438069 | 0.28892 | 69 | 31 | -0.000133153 |

## chunk_id = 10 (17 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `task_semantic_alignment` | 3.68998e-05 | 0.487144 | 69 | 31 | -0.0125814 |
| `chunk_action_variance` | 9.92213e-05 | 0.465171 | 69 | 31 | 0.0120811 |
| `video_action_alignment_score` | 0.00132915 | 0.400655 | 69 | 31 | -0.00342355 |
| `video_action_cosine_similarity` | 0.00132915 | 0.400655 | 69 | 31 | -0.00342355 |
| `action_uncertainty` | 0.00203787 | 0.388032 | 69 | 31 | 0.00915917 |
| `video_latent_entropy` | 0.00282969 | 0.379617 | 69 | 31 | -0.000140878 |
| `decoder_hidden_norm_mean` | 0.0046512 | 0.364656 | 69 | 31 | 0.0232601 |
| `object_interaction_confidence` | 0.0046512 | 0.364656 | 69 | 31 | 0.00260839 |
| `video_latent_norm_std` | 0.00745864 | 0.350164 | 69 | 31 | 0.15452 |
| `sampling_stability` | 0.00745864 | 0.350164 | 69 | 31 | -0.00108456 |
| `mlp_activation_mean` | 0.00821792 | 0.347359 | 69 | 31 | 0.00722876 |
| `latent_success_alignment` | 0.0091307 | 0.343619 | 69 | 31 | 0.000359629 |
| `instruction_attention_score` | 0.00963013 | 0.342216 | 69 | 31 | 0.000127515 |
| `residual_stream_norm` | 0.0129085 | 0.332398 | 69 | 31 | 0.0752526 |
| `representation_drift_rate` | 0.0222928 | 0.313698 | 69 | 31 | -0.275803 |
| `decoder_hidden_norm_std` | 0.0237454 | 0.311828 | 69 | 31 | -0.0057275 |
| `latent_oscillation_score` | 0.0370504 | 0.295465 | 69 | 31 | -0.000136221 |

## chunk_id = 11 (16 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `task_semantic_alignment` | 3.68998e-05 | 0.487144 | 69 | 31 | -0.0145779 |
| `decoder_hidden_norm_std` | 0.00102961 | 0.4072 | 69 | 31 | -0.00566297 |
| `action_uncertainty` | 0.00203787 | 0.388032 | 69 | 31 | 0.0122637 |
| `residual_stream_norm` | 0.00284131 | 0.379149 | 69 | 31 | 0.0965041 |
| `chunk_action_variance` | 0.00639066 | 0.355306 | 69 | 31 | 0.0101455 |
| `mlp_sparsity` | 0.00825263 | 0.346891 | 69 | 31 | -0.00314088 |
| `video_action_alignment_score` | 0.0104696 | 0.339411 | 69 | 31 | -0.00251183 |
| `video_action_cosine_similarity` | 0.0104696 | 0.339411 | 69 | 31 | -0.00251183 |
| `instruction_attention_score` | 0.0149628 | 0.327723 | 69 | 31 | -0.000670133 |
| `representation_drift_rate` | 0.0151138 | 0.326788 | 69 | 31 | -0.347865 |
| `decoder_hidden_norm_mean` | 0.0197884 | 0.317906 | 69 | 31 | 0.00788509 |
| `object_interaction_confidence` | 0.0197884 | 0.317906 | 69 | 31 | 0.000846061 |
| `latent_success_alignment` | 0.0237892 | 0.31136 | 69 | 31 | 0.000511296 |
| `video_latent_variance_mean` | 0.0324681 | 0.30014 | 69 | 31 | -0.0344039 |
| `latent_oscillation_score` | 0.0363799 | 0.295933 | 69 | 31 | -0.000115653 |
| `video_action_mutual_information` | 0.040298 | 0.292193 | 69 | 31 | 0.000893641 |

## chunk_id = 12 (12 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `task_semantic_alignment` | 3.68998e-05 | 0.487144 | 69 | 31 | -0.0177681 |
| `decoder_hidden_norm_std` | 0.000689234 | 0.417952 | 69 | 31 | -0.00911038 |
| `mlp_sparsity` | 0.00574321 | 0.358579 | 69 | 31 | -0.00588593 |
| `action_uncertainty` | 0.00614023 | 0.355774 | 69 | 31 | 0.0071015 |
| `instruction_attention_score` | 0.00963013 | 0.342216 | 69 | 31 | -0.00136243 |
| `decoder_hidden_norm_mean` | 0.0117186 | 0.335671 | 69 | 31 | -0.0164619 |
| `object_interaction_confidence` | 0.0117186 | 0.335671 | 69 | 31 | -0.00196704 |
| `attention_sparsity` | 0.0119734 | 0.334736 | 69 | 31 | 1.81929e-05 |
| `chunk_action_variance` | 0.017237 | 0.323048 | 69 | 31 | 0.00582636 |
| `mlp_activation_mean` | 0.0207915 | 0.316503 | 69 | 31 | -0.00515567 |
| `chunk_action_delta_norm_mean` | 0.0252876 | 0.309023 | 69 | 31 | 0.00872669 |
| `video_latent_variance_mean` | 0.03864 | 0.293595 | 69 | 31 | -0.0276796 |

## chunk_id = 13 (18 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 1.59935e-05 | 0.504909 | 69 | 31 | 0.0633519 |
| `action_chunk_temporal_consistency` | 1.59935e-05 | 0.504909 | 69 | 31 | -0.0444944 |
| `plan_drift` | 1.59935e-05 | 0.504909 | 69 | 31 | 3.73776 |
| `task_semantic_alignment` | 3.16899e-05 | 0.490416 | 69 | 31 | -0.0169265 |
| `instruction_attention_score` | 0.000214988 | 0.446938 | 69 | 31 | -0.0114581 |
| `decoder_hidden_norm_std` | 0.000535769 | 0.424497 | 69 | 31 | -0.0102262 |
| `mlp_sparsity` | 0.000773064 | 0.415147 | 69 | 31 | -0.00741078 |
| `action_chunk_smoothness` | 0.00142736 | 0.398784 | 69 | 31 | -0.0465355 |
| `chunk_action_delta_norm_mean` | 0.00190688 | 0.390837 | 69 | 31 | 0.0173541 |
| `chunk_gripper_switches` | 0.00442058 | 0.366526 | 69 | 31 | 1.58626 |
| `video_latent_norm_std` | 0.00550149 | 0.359046 | 69 | 31 | 0.180603 |
| `sampling_stability` | 0.00550149 | 0.359046 | 69 | 31 | -0.00127767 |
| `video_latent_entropy` | 0.011178 | 0.337541 | 69 | 31 | -0.000174626 |
| `chunk_action_variance` | 0.017237 | 0.323048 | 69 | 31 | 0.00610596 |
| `decoder_hidden_norm_mean` | 0.0189475 | 0.319776 | 69 | 31 | -0.0102552 |
| `object_interaction_confidence` | 0.0189475 | 0.319776 | 69 | 31 | -0.00120764 |
| `video_latent_variance_mean` | 0.0272321 | 0.306685 | 69 | 31 | -0.00635627 |
| `video_latent_delta_l2_std` | 0.0277399 | 0.30575 | 69 | 31 | 0.208096 |

## chunk_id = 14 (20 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 3.49344e-08 | 0.616176 | 69 | 31 | 0.090264 |
| `action_chunk_temporal_consistency` | 3.49344e-08 | 0.616176 | 69 | 31 | -0.0639087 |
| `plan_drift` | 3.49344e-08 | 0.616176 | 69 | 31 | 5.32557 |
| `action_chunk_smoothness` | 3.41803e-06 | 0.535764 | 69 | 31 | -0.075056 |
| `chunk_action_delta_norm_mean` | 3.79256e-06 | 0.534362 | 69 | 31 | 0.0216596 |
| `task_semantic_alignment` | 3.16899e-05 | 0.490416 | 69 | 31 | -0.0160284 |
| `chunk_gripper_switches` | 3.92151e-05 | 0.485741 | 69 | 31 | 2.56802 |
| `video_latent_delta_l2_std` | 0.000212809 | 0.447405 | 69 | 31 | 0.304327 |
| `video_latent_entropy` | 0.000364978 | 0.434315 | 69 | 31 | -0.000217054 |
| `chunk_action_variance` | 0.00112318 | 0.40533 | 69 | 31 | 0.00806577 |
| `instruction_attention_score` | 0.00119437 | 0.40346 | 69 | 31 | -0.0121874 |
| `mlp_sparsity` | 0.00190688 | 0.390837 | 69 | 31 | -0.00638071 |
| `latent_success_alignment` | 0.00284131 | 0.379149 | 69 | 31 | -0.0015027 |
| `video_latent_norm_std` | 0.0184059 | 0.320243 | 69 | 31 | 0.200455 |
| `sampling_stability` | 0.0184059 | 0.320243 | 69 | 31 | -0.00141769 |
| `video_latent_cosine_initial_final` | 0.0197884 | 0.317906 | 69 | 31 | -0.00438023 |
| `residual_stream_norm` | 0.0197884 | 0.317906 | 69 | 31 | -0.657698 |
| `action_entropy` | 0.0249291 | 0.309958 | 69 | 31 | 0.00309071 |
| `video_action_mutual_information` | 0.0249291 | 0.309958 | 69 | 31 | -0.00286842 |
| `video_latent_variance_mean` | 0.0272321 | 0.306685 | 69 | 31 | 0.0150104 |

## chunk_id = 15 (23 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `chunk_action_delta_norm_mean` | 5.85106e-06 | 0.52704 | 68 | 31 | 0.020217 |
| `flow_prediction_error` | 7.95659e-06 | 0.518975 | 68 | 31 | 0.056422 |
| `action_chunk_temporal_consistency` | 7.95659e-06 | 0.518975 | 68 | 31 | -0.0420342 |
| `plan_drift` | 7.95659e-06 | 0.518975 | 68 | 31 | 3.3289 |
| `task_semantic_alignment` | 1.63125e-05 | 0.504269 | 68 | 31 | -0.0160708 |
| `video_latent_delta_l2_std` | 1.87623e-05 | 0.503321 | 68 | 31 | 0.330544 |
| `latent_success_alignment` | 4.26628e-05 | 0.483871 | 68 | 31 | -0.00193408 |
| `chunk_action_variance` | 5.45073e-05 | 0.480076 | 68 | 31 | 0.00959654 |
| `instruction_attention_score` | 0.00012196 | 0.460152 | 68 | 31 | -0.0162086 |
| `mlp_sparsity` | 0.000639398 | 0.421252 | 68 | 31 | -0.00692837 |
| `video_latent_cosine_initial_final` | 0.00129061 | 0.401328 | 68 | 31 | -0.0048596 |
| `video_latent_entropy` | 0.00136521 | 0.400854 | 68 | 31 | -0.0002078 |
| `action_entropy` | 0.00199832 | 0.389469 | 68 | 31 | 0.00522908 |
| `latent_oscillation_score` | 0.00266584 | 0.38093 | 68 | 31 | -0.000219922 |
| `video_latent_variance_mean` | 0.003669 | 0.371917 | 68 | 31 | 0.0254383 |
| `video_action_mutual_information` | 0.00598853 | 0.357211 | 68 | 31 | -0.00499673 |
| `residual_stream_norm` | 0.00851969 | 0.345825 | 68 | 31 | -0.92182 |
| `goal_latent_distance` | 0.00929841 | 0.344402 | 68 | 31 | 0.229891 |
| `chunk_gripper_switches` | 0.0158536 | 0.325427 | 68 | 31 | 1.50664 |
| `decoder_hidden_norm_std` | 0.0186814 | 0.321157 | 68 | 31 | -0.00686711 |
| `video_action_alignment_score` | 0.0283942 | 0.305977 | 68 | 31 | 0.00676271 |
| `video_action_cosine_similarity` | 0.0283942 | 0.305977 | 68 | 31 | 0.00676271 |
| `action_chunk_smoothness` | 0.0427495 | 0.289848 | 68 | 31 | -0.0398192 |

## chunk_id = 16 (21 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 4.19286e-06 | 0.534491 | 65 | 31 | 0.0801427 |
| `action_chunk_temporal_consistency` | 4.19286e-06 | 0.534491 | 65 | 31 | -0.0577219 |
| `plan_drift` | 4.19286e-06 | 0.534491 | 65 | 31 | 4.72842 |
| `chunk_action_delta_norm_mean` | 1.8847e-05 | 0.505211 | 65 | 31 | 0.0244798 |
| `task_semantic_alignment` | 2.05356e-05 | 0.502233 | 65 | 31 | -0.0156954 |
| `chunk_action_variance` | 0.00021566 | 0.451117 | 65 | 31 | 0.00913424 |
| `latent_success_alignment` | 0.000314813 | 0.440695 | 65 | 31 | -0.00182598 |
| `video_latent_delta_l2_std` | 0.000328885 | 0.440199 | 65 | 31 | 0.31767 |
| `chunk_gripper_switches` | 0.00037403 | 0.436228 | 65 | 31 | 2.0794 |
| `instruction_attention_score` | 0.000582933 | 0.42531 | 65 | 31 | -0.00735926 |
| `action_entropy` | 0.00263324 | 0.385112 | 65 | 31 | 0.00585435 |
| `video_action_mutual_information` | 0.00263324 | 0.385112 | 65 | 31 | -0.00563666 |
| `residual_stream_norm` | 0.00364452 | 0.37469 | 65 | 31 | -0.981138 |
| `video_latent_variance_mean` | 0.00725634 | 0.354342 | 65 | 31 | 0.0250129 |
| `mlp_activation_mean` | 0.00905128 | 0.346898 | 65 | 31 | 0.00783046 |
| `video_latent_entropy` | 0.0112005 | 0.340447 | 65 | 31 | -0.000186042 |
| `action_chunk_smoothness` | 0.0144254 | 0.331514 | 65 | 31 | -0.0527543 |
| `representation_drift_rate` | 0.0169786 | 0.326551 | 65 | 31 | -0.0394608 |
| `mlp_sparsity` | 0.0197886 | 0.320596 | 65 | 31 | -0.00520761 |
| `latent_oscillation_score` | 0.0197886 | 0.320596 | 65 | 31 | -0.000246555 |
| `action_uncertainty` | 0.0425333 | 0.292804 | 65 | 31 | 0.00496866 |

## chunk_id = 17 (24 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 9.13266e-10 | 0.677419 | 61 | 31 | 0.0822188 |
| `action_chunk_temporal_consistency` | 9.13266e-10 | 0.677419 | 61 | 31 | -0.0592229 |
| `plan_drift` | 9.13266e-10 | 0.677419 | 61 | 31 | 4.85091 |
| `chunk_action_delta_norm_mean` | 2.4128e-06 | 0.546801 | 61 | 31 | 0.0259142 |
| `instruction_attention_score` | 2.51925e-05 | 0.499736 | 61 | 31 | -0.0213868 |
| `task_semantic_alignment` | 2.52928e-05 | 0.499207 | 61 | 31 | -0.0147728 |
| `chunk_gripper_switches` | 2.79679e-05 | 0.49762 | 61 | 31 | 2.10841 |
| `chunk_action_variance` | 9.09056e-05 | 0.478583 | 61 | 31 | 0.0115269 |
| `latent_success_alignment` | 0.000212617 | 0.450555 | 61 | 31 | -0.00194732 |
| `action_entropy` | 0.000249121 | 0.44844 | 61 | 31 | 0.00729284 |
| `video_action_mutual_information` | 0.000249121 | 0.44844 | 61 | 31 | -0.00704731 |
| `action_chunk_smoothness` | 0.000267909 | 0.447911 | 61 | 31 | -0.0488671 |
| `video_latent_variance_mean` | 0.00045759 | 0.432575 | 61 | 31 | 0.0391477 |
| `video_latent_delta_l2_std` | 0.000758612 | 0.418826 | 61 | 31 | 0.354326 |
| `latent_oscillation_score` | 0.00138453 | 0.402433 | 61 | 31 | -0.000289114 |
| `mlp_activation_mean` | 0.00152855 | 0.400317 | 61 | 31 | 0.0109471 |
| `mlp_sparsity` | 0.0026391 | 0.384453 | 61 | 31 | -0.00848317 |
| `residual_stream_norm` | 0.00424898 | 0.370175 | 61 | 31 | -1.11008 |
| `video_latent_entropy` | 0.00975996 | 0.349022 | 61 | 31 | -0.000203038 |
| `video_latent_norm_mean` | 0.0118937 | 0.337388 | 61 | 31 | -0.402516 |
| `score_norm_mean` | 0.0118937 | 0.337388 | 61 | 31 | -0.00503139 |
| `video_latent_cosine_initial_final` | 0.0192188 | 0.320994 | 61 | 31 | -0.00288163 |
| `video_latent_norm_std` | 0.0294792 | 0.312004 | 61 | 31 | 0.150628 |
| `sampling_stability` | 0.0294792 | 0.312004 | 61 | 31 | -0.00106553 |

## chunk_id = 18 (24 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 3.79342e-08 | 0.639684 | 53 | 31 | 0.0688558 |
| `action_chunk_temporal_consistency` | 3.79342e-08 | 0.639684 | 53 | 31 | -0.0487222 |
| `plan_drift` | 3.79342e-08 | 0.639684 | 53 | 31 | 4.06249 |
| `task_semantic_alignment` | 7.19644e-05 | 0.491783 | 53 | 31 | -0.0146965 |
| `video_latent_variance_mean` | 0.000138883 | 0.477785 | 53 | 31 | 0.0429968 |
| `latent_success_alignment` | 0.000158832 | 0.472915 | 53 | 31 | -0.00177834 |
| `latent_oscillation_score` | 0.00042202 | 0.44857 | 53 | 31 | -0.000262528 |
| `chunk_action_variance` | 0.000525403 | 0.443092 | 53 | 31 | 0.00890385 |
| `instruction_attention_score` | 0.000562698 | 0.440657 | 53 | 31 | -0.0161217 |
| `chunk_gripper_switches` | 0.000804928 | 0.432136 | 53 | 31 | 1.76993 |
| `chunk_action_delta_norm_mean` | 0.0012116 | 0.421181 | 53 | 31 | 0.0194572 |
| `video_latent_entropy` | 0.00138152 | 0.416312 | 53 | 31 | -0.000238047 |
| `video_latent_delta_l2_std` | 0.00158066 | 0.413268 | 53 | 31 | 0.41159 |
| `action_chunk_smoothness` | 0.00158066 | 0.413268 | 53 | 31 | -0.0354251 |
| `residual_stream_norm` | 0.00623702 | 0.370663 | 53 | 31 | -0.966859 |
| `video_latent_norm_mean` | 0.00935196 | 0.357273 | 53 | 31 | -0.305533 |
| `score_norm_mean` | 0.00935196 | 0.357273 | 53 | 31 | -0.00381915 |
| `latent_path_efficiency` | 0.0104212 | 0.35423 | 53 | 31 | -2.1506e-07 |
| `video_latent_norm_std` | 0.0162062 | 0.338405 | 53 | 31 | 0.229231 |
| `sampling_stability` | 0.0162062 | 0.338405 | 53 | 31 | -0.00161759 |
| `decoder_hidden_norm_std` | 0.039845 | 0.305539 | 53 | 31 | -0.00220985 |
| `mlp_activation_mean` | 0.041996 | 0.303104 | 53 | 31 | 0.00880313 |
| `query_latency_sec` | 0.0433071 | 0.302495 | 53 | 31 | -0.0509158 |
| `representation_drift_rate` | 0.0457797 | 0.300061 | 53 | 31 | 0.0944046 |

## chunk_id = 19 (20 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 2.93596e-07 | 0.649642 | 36 | 31 | 0.0665958 |
| `action_chunk_temporal_consistency` | 2.93596e-07 | 0.649642 | 36 | 31 | -0.0461827 |
| `plan_drift` | 2.93596e-07 | 0.649642 | 36 | 31 | 3.92915 |
| `video_latent_entropy` | 0.000121771 | 0.516129 | 36 | 31 | -0.000239741 |
| `latent_success_alignment` | 0.000295713 | 0.492832 | 36 | 31 | -0.00172021 |
| `chunk_action_delta_norm_mean` | 0.000518591 | 0.478495 | 36 | 31 | 0.018206 |
| `task_semantic_alignment` | 0.000796153 | 0.465054 | 36 | 31 | -0.0142496 |
| `latent_path_efficiency` | 0.00153203 | 0.446237 | 36 | 31 | -2.3892e-07 |
| `chunk_action_variance` | 0.00175192 | 0.441756 | 36 | 31 | 0.00659205 |
| `instruction_attention_score` | 0.0020027 | 0.437276 | 36 | 31 | -0.0190605 |
| `mlp_sparsity` | 0.00218913 | 0.43638 | 36 | 31 | -0.00828256 |
| `residual_stream_norm` | 0.00262333 | 0.428315 | 36 | 31 | -0.806374 |
| `chunk_gripper_switches` | 0.00300492 | 0.423835 | 36 | 31 | 1.81272 |
| `action_chunk_smoothness` | 0.00778117 | 0.391577 | 36 | 31 | -0.0207699 |
| `video_latent_delta_l2_std` | 0.00827397 | 0.390681 | 36 | 31 | 0.350595 |
| `video_latent_norm_mean` | 0.0117257 | 0.37724 | 36 | 31 | -0.277786 |
| `score_norm_mean` | 0.0117257 | 0.37724 | 36 | 31 | -0.00347233 |
| `video_latent_variance_mean` | 0.0125015 | 0.376344 | 36 | 31 | 0.0339998 |
| `decoder_hidden_norm_std` | 0.0207314 | 0.356631 | 36 | 31 | -0.00577766 |
| `query_latency_sec` | 0.0364333 | 0.334229 | 36 | 31 | -0.0669947 |

## chunk_id = 20 (21 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `flow_prediction_error` | 2.47064e-05 | 0.580645 | 29 | 31 | 0.0612935 |
| `action_chunk_temporal_consistency` | 2.47064e-05 | 0.580645 | 29 | 31 | -0.0432061 |
| `plan_drift` | 2.47064e-05 | 0.580645 | 29 | 31 | 3.61632 |
| `video_latent_entropy` | 2.81495e-05 | 0.576196 | 29 | 31 | -0.000241354 |
| `action_chunk_smoothness` | 0.000931727 | 0.483871 | 29 | 31 | -0.0167473 |
| `chunk_gripper_switches` | 0.000931727 | 0.483871 | 29 | 31 | 1.86541 |
| `residual_stream_norm` | 0.000954012 | 0.481646 | 29 | 31 | -0.825733 |
| `latent_success_alignment` | 0.000954012 | 0.481646 | 29 | 31 | -0.00171457 |
| `instruction_attention_score` | 0.00099558 | 0.479422 | 29 | 31 | -0.017533 |
| `task_semantic_alignment` | 0.00099558 | 0.479422 | 29 | 31 | -0.0163124 |
| `chunk_action_delta_norm_mean` | 0.00116616 | 0.474972 | 29 | 31 | 0.0149808 |
| `latent_path_efficiency` | 0.00266497 | 0.449388 | 29 | 31 | -3.02725e-07 |
| `video_latent_norm_mean` | 0.00708724 | 0.414905 | 29 | 31 | -0.358712 |
| `score_norm_mean` | 0.00708724 | 0.414905 | 29 | 31 | -0.0044839 |
| `chunk_action_variance` | 0.00743548 | 0.412681 | 29 | 31 | 0.00389593 |
| `video_latent_norm_std` | 0.00845456 | 0.408231 | 29 | 31 | 0.234795 |
| `sampling_stability` | 0.00845456 | 0.408231 | 29 | 31 | -0.00165903 |
| `representation_drift_rate` | 0.015223 | 0.388209 | 29 | 31 | 0.0383277 |
| `video_latent_delta_l2_std` | 0.0267829 | 0.36485 | 29 | 31 | 0.33412 |
| `video_action_alignment_score` | 0.0343951 | 0.352614 | 29 | 31 | -0.000574948 |
| `video_action_cosine_similarity` | 0.0343951 | 0.352614 | 29 | 31 | -0.000574948 |

## chunk_id = 21 (18 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `instruction_attention_score` | 2.53436e-06 | 0.769892 | 15 | 31 | -0.030238 |
| `task_semantic_alignment` | 2.53436e-06 | 0.769892 | 15 | 31 | -0.0279467 |
| `video_latent_norm_std` | 6.30261e-06 | 0.739785 | 15 | 31 | 0.440732 |
| `sampling_stability` | 6.30261e-06 | 0.739785 | 15 | 31 | -0.00311388 |
| `latent_path_efficiency` | 2.63175e-05 | 0.705376 | 15 | 31 | -5.58316e-07 |
| `video_latent_delta_l2_std` | 3.21332e-05 | 0.701075 | 15 | 31 | 0.588008 |
| `video_latent_entropy` | 5.50935e-05 | 0.675269 | 15 | 31 | -0.000373734 |
| `query_latency_sec` | 0.00167554 | 0.569892 | 15 | 31 | 0.00603246 |
| `goal_latent_distance` | 0.0017098 | 0.567742 | 15 | 31 | -1.28472 |
| `residual_stream_norm` | 0.00213737 | 0.546237 | 15 | 31 | -0.521394 |
| `representation_drift_rate` | 0.00823642 | 0.503226 | 15 | 31 | 0.168728 |
| `video_latent_cosine_initial_final` | 0.00881098 | 0.483871 | 15 | 31 | 0.00580605 |
| `flow_prediction_error` | 0.00881098 | 0.483871 | 15 | 31 | 0.0290359 |
| `action_chunk_temporal_consistency` | 0.00881098 | 0.483871 | 15 | 31 | -0.0179426 |
| `plan_drift` | 0.00881098 | 0.483871 | 15 | 31 | 1.71312 |
| `latent_success_alignment` | 0.017751 | 0.451613 | 15 | 31 | -0.00174125 |
| `chunk_action_variance` | 0.017751 | 0.451613 | 15 | 31 | 0.00557553 |
| `mlp_activation_mean` | 0.0338785 | 0.419355 | 15 | 31 | 0.0150425 |

## chunk_id = 22 (20 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `instruction_attention_score` | 2.63175e-05 | 0.705376 | 15 | 31 | -0.0333766 |
| `task_semantic_alignment` | 2.63175e-05 | 0.705376 | 15 | 31 | -0.0273166 |
| `latent_path_efficiency` | 7.43765e-05 | 0.673118 | 15 | 31 | -5.93031e-07 |
| `video_latent_entropy` | 0.000270705 | 0.634409 | 15 | 31 | -0.000324276 |
| `decoder_hidden_norm_std` | 0.000381417 | 0.610753 | 15 | 31 | 0.0140639 |
| `video_latent_delta_l2_std` | 0.000665753 | 0.604301 | 15 | 31 | 0.569711 |
| `video_latent_norm_std` | 0.000703843 | 0.6 | 15 | 31 | 0.351701 |
| `sampling_stability` | 0.000703843 | 0.6 | 15 | 31 | -0.0024803 |
| `residual_stream_norm` | 0.00468477 | 0.513978 | 15 | 31 | -0.738244 |
| `chunk_action_variance` | 0.00468477 | 0.513978 | 15 | 31 | 0.0084043 |
| `chunk_action_delta_norm_mean` | 0.00716449 | 0.507527 | 15 | 31 | 0.0221093 |
| `goal_latent_distance` | 0.00823642 | 0.503226 | 15 | 31 | -1.32735 |
| `flow_prediction_error` | 0.017751 | 0.451613 | 15 | 31 | 0.0291268 |
| `mlp_activation_mean` | 0.017751 | 0.451613 | 15 | 31 | 0.0151369 |
| `action_chunk_temporal_consistency` | 0.017751 | 0.451613 | 15 | 31 | -0.0194278 |
| `plan_drift` | 0.017751 | 0.451613 | 15 | 31 | 1.71848 |
| `decoder_hidden_norm_mean` | 0.0338785 | 0.419355 | 15 | 31 | 0.0213318 |
| `object_interaction_confidence` | 0.0338785 | 0.419355 | 15 | 31 | 0.00265674 |
| `video_latent_norm_mean` | 0.0363923 | 0.417204 | 15 | 31 | -0.282205 |
| `score_norm_mean` | 0.0363923 | 0.417204 | 15 | 31 | -0.00352764 |

## chunk_id = 23 (20 метрик)

| metric | p_value | ks_statistic | n_success | n_failure | failure_mean − success_mean |
|--------|---------|--------------|-----------|-----------|---------------------------|
| `instruction_attention_score` | 3.65925e-06 | 0.806452 | 12 | 31 | -0.0428272 |
| `task_semantic_alignment` | 3.65925e-06 | 0.806452 | 12 | 31 | -0.0339938 |
| `latent_path_efficiency` | 3.30543e-05 | 0.741935 | 12 | 31 | -5.9093e-07 |
| `video_latent_norm_std` | 0.000313685 | 0.666667 | 12 | 31 | 0.308414 |
| `video_latent_delta_l2_std` | 0.000313685 | 0.666667 | 12 | 31 | 0.537113 |
| `video_latent_entropy` | 0.000313685 | 0.666667 | 12 | 31 | -0.000300462 |
| `sampling_stability` | 0.000313685 | 0.666667 | 12 | 31 | -0.00217526 |
| `video_action_alignment_score` | 0.000571908 | 0.645161 | 12 | 31 | -0.0107212 |
| `video_action_cosine_similarity` | 0.000571908 | 0.645161 | 12 | 31 | -0.0107212 |
| `goal_latent_distance` | 0.00788559 | 0.537634 | 12 | 31 | -1.24188 |
| `video_latent_norm_mean` | 0.0118115 | 0.516129 | 12 | 31 | -0.378846 |
| `decoder_hidden_norm_mean` | 0.0118115 | 0.516129 | 12 | 31 | 0.017899 |
| `residual_stream_norm` | 0.0118115 | 0.516129 | 12 | 31 | -1.12019 |
| `action_chunk_smoothness` | 0.0118115 | 0.516129 | 12 | 31 | 0.0496166 |
| `object_interaction_confidence` | 0.0118115 | 0.516129 | 12 | 31 | 0.00222428 |
| `score_norm_mean` | 0.0118115 | 0.516129 | 12 | 31 | -0.00473563 |
| `chunk_action_delta_norm_mean` | 0.0152382 | 0.505376 | 12 | 31 | 0.0212323 |
| `chunk_action_variance` | 0.0222399 | 0.483871 | 12 | 31 | 0.00744759 |
| `mlp_activation_mean` | 0.0398783 | 0.451613 | 12 | 31 | 0.0119454 |
| `latent_success_alignment` | 0.0398783 | 0.451613 | 12 | 31 | -0.00257832 |

## Сводка по метрикам (на каких chunk значимо)

| metric | число значимых chunk | chunk_id |
|--------|---------------------|----------|
| `task_semantic_alignment` | 24 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 |
| `video_latent_entropy` | 22 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 |
| `chunk_action_variance` | 20 | 0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 |
| `chunk_action_delta_norm_mean` | 19 | 0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 23 |
| `video_latent_norm_std` | 18 | 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 17, 18, 20, 21, 22, 23 |
| `sampling_stability` | 18 | 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 13, 14, 17, 18, 20, 21, 22, 23 |
| `flow_prediction_error` | 18 | 0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 |
| `action_chunk_temporal_consistency` | 18 | 0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 |
| `plan_drift` | 18 | 0, 1, 2, 3, 4, 5, 6, 7, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 |
| `instruction_attention_score` | 18 | 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 |
| `decoder_hidden_norm_std` | 17 | 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 18, 19, 22 |
| `mlp_activation_mean` | 17 | 0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 12, 16, 17, 18, 21, 22, 23 |
| `decoder_hidden_norm_mean` | 16 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23 |
| `object_interaction_confidence` | 16 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 22, 23 |
| `representation_drift_rate` | 16 | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 18, 20, 21 |
| `residual_stream_norm` | 15 | 0, 7, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 |
| `video_latent_variance_mean` | 14 | 0, 1, 2, 3, 5, 11, 12, 13, 14, 15, 16, 17, 18, 19 |
| `video_latent_delta_l2_std` | 14 | 2, 3, 9, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23 |
| `video_action_mutual_information` | 14 | 0, 1, 2, 3, 4, 5, 6, 7, 9, 11, 14, 15, 16, 17 |
| `mlp_sparsity` | 14 | 0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 16, 17, 19 |
| `action_chunk_smoothness` | 13 | 2, 6, 7, 9, 13, 14, 15, 16, 17, 18, 19, 20, 23 |
| `action_entropy` | 13 | 0, 1, 2, 3, 4, 5, 6, 7, 9, 14, 15, 16, 17 |
| `latent_success_alignment` | 13 | 0, 9, 10, 11, 14, 15, 16, 17, 18, 19, 20, 21, 23 |
| `goal_latent_distance` | 9 | 0, 1, 2, 3, 4, 15, 21, 22, 23 |
| `chunk_gripper_switches` | 9 | 9, 13, 14, 15, 16, 17, 18, 19, 20 |
| `action_uncertainty` | 8 | 2, 7, 8, 9, 10, 11, 12, 16 |
| `attention_sparsity` | 8 | 1, 2, 3, 4, 5, 6, 7, 12 |
| `chunk_action_delta_norm_max` | 8 | 1, 2, 3, 4, 5, 6, 7, 8 |
| `video_action_alignment_score` | 8 | 0, 5, 8, 10, 11, 15, 20, 23 |
| `video_action_cosine_similarity` | 8 | 0, 5, 8, 10, 11, 15, 20, 23 |
| `latent_oscillation_score` | 7 | 9, 10, 11, 15, 16, 17, 18 |
| `latent_path_efficiency` | 6 | 18, 19, 20, 21, 22, 23 |
| `video_latent_norm_mean` | 6 | 17, 18, 19, 20, 22, 23 |
| `score_norm_mean` | 6 | 17, 18, 19, 20, 22, 23 |
| `video_latent_cosine_initial_final` | 4 | 14, 15, 17, 21 |
| `query_latency_sec` | 3 | 18, 19, 21 |
