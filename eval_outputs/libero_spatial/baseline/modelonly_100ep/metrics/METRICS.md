# Mimic-Video LIBERO Model-Only Metrics

Each episode produces one row in `episodes_metrics.jsonl` and `episodes_metrics.csv`.

The metric columns in this file are computed only from values produced by the model:
video2world hidden/context latents, denoising sigma, latent-energy attention proxies,
precomputed language embeddings, and predicted action chunks. Environment rewards,
end-effector state, simulator state, and rendered frame statistics are intentionally
not persisted as metrics.

## Non-Metric Labels / Identifiers
- `task_id`: LIBERO task index, kept only for grouping.
- `episode_idx`: Trial index within the task, kept only for grouping.
- `total_episode_idx`: Global episode counter, kept only for grouping.
- `task_description`: Language instruction, kept only for grouping.
- `success`: Environment success label for later supervised analysis; not used to compute model metrics.
- `termination_reason`: Eval outcome label (`success`, `timeout`, `exception`, `stopped`).
- `exception`: Exception text if rollout failed.
- `step_count`: Rollout length label for context, not a model metric.
- `num_policy_queries`: Number of model queries made in the episode.

## Model Metric Families
- `video_latent_*`: Statistics of `crossattn_emb` returned by video2world at the chosen denoising step.
- `latent_*`, `representation_*`, `future_latent_*`, `goal_latent_*`: Latent trajectory, entropy, SNR, path, and drift diagnostics.
- `cross_attention_*`, `attention_*`, `instruction_*`: Attention proxies computed from latent-token energy distributions and language embedding alignment.
- `decoder_hidden_*`, `mlp_*`, `residual_stream_norm`: Action decoder and action-hidden proxy diagnostics.
- `action_*`, `plan_*`, `inverse_dynamics_*`: Predicted action chunk entropy, smoothness, uncertainty, drift, and consistency diagnostics.
- `video_action_*`, `language_action_alignment`, `tri_modal_alignment_score`: Alignment proxies between video latents, language embedding, and action chunks.
- `score_*`, `sampling_*`, `solver_*`, `flow_*`, `remaining_noise_norm`: Sampling and denoising proxy diagnostics from latent deltas and sigma.

## Notes
Some requested metrics name quantities that are not exposed by the public model API
(for example raw attention maps). For those, the implementation uses deterministic
proxies from available model outputs, such as softmax-normalized latent-token energy.

## Emitted Columns
- `action_chunk_smoothness`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `action_chunk_temporal_consistency`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `action_entropy`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `action_uncertainty`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `activation_explosion_score`: Sampling/flow proxy derived from latent deltas and denoising sigma, averaged over policy queries.
- `attention_collapse_score`: Attention/semantic proxy derived from latent energy distributions and precomputed language embedding, averaged over policy queries.
- `attention_peakiness`: Attention/semantic proxy derived from latent energy distributions and precomputed language embedding, averaged over policy queries.
- `attention_sparsity`: Attention/semantic proxy derived from latent energy distributions and precomputed language embedding, averaged over policy queries.
- `causal_transition_confidence`: Requested model diagnostic averaged over policy queries in the episode.
- `cross_attention_entropy`: Attention/semantic proxy derived from latent energy distributions and precomputed language embedding, averaged over policy queries.
- `cross_attention_max_weight`: Attention/semantic proxy derived from latent energy distributions and precomputed language embedding, averaged over policy queries.
- `decoder_hidden_norm_mean`: Requested model diagnostic averaged over policy queries in the episode.
- `decoder_hidden_norm_std`: Requested model diagnostic averaged over policy queries in the episode.
- `episode_idx`: Trial index within the task.
- `exception`: Exception text if rollout failed inside the control loop.
- `flow_prediction_error`: Sampling/flow proxy derived from latent deltas and denoising sigma, averaged over policy queries.
- `flow_vector_norm`: Sampling/flow proxy derived from latent deltas and denoising sigma, averaged over policy queries.
- `future_latent_prediction_error`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `future_visual_consistency`: Requested model diagnostic averaged over policy queries in the episode.
- `goal_latent_distance`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `goal_progress_velocity`: Requested model diagnostic averaged over policy queries in the episode.
- `instruction_attention_score`: Attention/semantic proxy derived from latent energy distributions and precomputed language embedding, averaged over policy queries.
- `inverse_dynamics_reconstruction_error`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `language_action_alignment`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `latent_freeze_fraction`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_information_density`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_motion_coherence`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_oscillation_score`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_path_efficiency`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_path_length`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_residual_energy`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_snr`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_success_alignment`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `latent_update_norm`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `mlp_activation_mean`: Requested model diagnostic averaged over policy queries in the episode.
- `mlp_sparsity`: Requested model diagnostic averaged over policy queries in the episode.
- `num_policy_queries`: Number of VAM model forward passes.
- `object_interaction_confidence`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `partial_denoise_distance`: Requested model diagnostic averaged over policy queries in the episode.
- `plan_consistency`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `plan_drift`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `predicted_episode_success`: Requested model diagnostic averaged over policy queries in the episode.
- `remaining_noise_norm`: Requested model diagnostic averaged over policy queries in the episode.
- `representation_drift_rate`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `representation_entropy`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `residual_stream_norm`: Requested model diagnostic averaged over policy queries in the episode.
- `sampling_stability`: Sampling/flow proxy derived from latent deltas and denoising sigma, averaged over policy queries.
- `score_norm_mean`: Sampling/flow proxy derived from latent deltas and denoising sigma, averaged over policy queries.
- `solver_step_error`: Sampling/flow proxy derived from latent deltas and denoising sigma, averaged over policy queries.
- `step_count`: Number of control steps after warmup.
- `success`: Simulator success flag.
- `task_description`: Natural language task instruction.
- `task_id`: LIBERO task index.
- `task_semantic_alignment`: Attention/semantic proxy derived from latent energy distributions and precomputed language embedding, averaged over policy queries.
- `temporal_attention_stability`: Attention/semantic proxy derived from latent energy distributions and precomputed language embedding, averaged over policy queries.
- `termination_reason`: Why the episode ended: success, timeout, exception, or stopped.
- `total_episode_idx`: Global episode counter.
- `tri_modal_alignment_score`: Alignment proxy between video latent context, action chunk, and language embedding, averaged over policy queries.
- `video_action_alignment_score`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `video_action_cosine_similarity`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `video_action_mutual_information`: Action-decoder diagnostic derived from the predicted action chunk, averaged over policy queries.
- `video_latent_cosine_initial_final`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_cosine_t_t1_mean`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_delta_l2_mean`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_delta_l2_std`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_entropy`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_norm_mean`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_norm_std`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_temporal_consistency`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_temporal_smoothness`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
- `video_latent_variance_mean`: Model diagnostic from video2world cross-attention latent/context tensors, averaged over policy queries in the episode.
