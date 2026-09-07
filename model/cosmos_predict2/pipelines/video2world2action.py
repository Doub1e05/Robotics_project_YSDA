import torch
from torch import nn
import torch.nn.functional as F

from cosmos_predict2.pipelines.video2world import DeviceMesh, Video2WorldPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline


class Video2World2ActionPipeline(nn.Module):
    def __init__(
        self,
        video2world_pipeline: Video2WorldPipeline,
        world2action_pipeline: World2ActionPipeline,
        diagnostics_mode: str = "default",
        representation_layer_indices: list[int] | None = None,
        decoder_capture_block_indices: list[int] | None = None,
        replay_payload_mode: str = "full",
    ) -> None:
        super().__init__()

        self.video2world_pipeline = video2world_pipeline
        self.world2action_pipeline = world2action_pipeline
        if diagnostics_mode not in {"default", "encoder_hidden", "decoder_hidden", "all"}:
            raise ValueError(f"Unsupported diagnostics_mode={diagnostics_mode!r}")
        if replay_payload_mode not in {"full", "decoder_only", "none"}:
            raise ValueError(f"Unsupported replay_payload_mode={replay_payload_mode!r}")
        self.diagnostics_mode = diagnostics_mode
        self.replay_payload_mode = replay_payload_mode
        decoder_layer_idx = int(self.world2action_pipeline.config.xattn_layer_idx)
        self.representation_layer_indices = sorted(
            {int(idx) for idx in (representation_layer_indices or [decoder_layer_idx])}
        )
        self.decoder_capture_block_indices = sorted({int(idx) for idx in (decoder_capture_block_indices or [])})
        self.last_diagnostics: dict[str, object] = {}
        self.last_replay_payload: dict[str, object] = {}
        self.last_decoder_hidden_states_by_block: dict[int, torch.Tensor] = {}

    @staticmethod
    def _resolve_decoder_block_request(
        decoder_hidden_state_list: list[torch.Tensor],
        requested_block_idx: int,
    ) -> tuple[int, torch.Tensor]:
        num_blocks = len(decoder_hidden_state_list)
        if requested_block_idx == num_blocks:
            actual_block_idx = num_blocks - 1
        else:
            actual_block_idx = requested_block_idx
        if actual_block_idx < 0 or actual_block_idx >= num_blocks:
            raise IndexError(
                f"Requested decoder block {requested_block_idx} is out of range for {num_blocks} captured blocks."
            )
        return actual_block_idx, decoder_hidden_state_list[actual_block_idx]

    @staticmethod
    def _entropy(prob: torch.Tensor) -> torch.Tensor:
        return -(prob * torch.log(prob.clamp_min(1e-12))).sum(dim=-1)

    def _collect_default_diagnostics(
        self,
        *,
        flat_crossattn_emb: torch.Tensor,
        actions: torch.Tensor,
        prompt_embedding: torch.Tensor | None,
        video_sigma: torch.Tensor,
    ) -> dict[str, float]:
        lat = flat_crossattn_emb.detach().float()
        act = actions.detach().float()
        if lat.ndim != 3:
            lat = lat.reshape(lat.shape[0], -1, lat.shape[-1])
        if act.ndim == 3:
            act_flat = act.reshape(act.shape[0], -1)
            act_steps = act
        else:
            act_flat = act.reshape(act.shape[0], -1)
            act_steps = act.reshape(act.shape[0], 1, -1)

        token_norms = torch.linalg.norm(lat, dim=-1)
        latent_delta = lat[:, 1:, :] - lat[:, :-1, :] if lat.shape[1] > 1 else torch.zeros_like(lat[:, :0, :])
        latent_delta_l2 = torch.linalg.norm(latent_delta, dim=-1) if latent_delta.numel() else torch.zeros(1, device=lat.device)
        latent_cos_adj = (
            F.cosine_similarity(lat[:, 1:, :], lat[:, :-1, :], dim=-1)
            if lat.shape[1] > 1
            else torch.ones(1, device=lat.device)
        )
        latent_cos_initial_final = F.cosine_similarity(lat[:, 0, :], lat[:, -1, :], dim=-1)
        latent_var = lat.var(dim=1)
        latent_energy = token_norms.square()
        latent_prob = latent_energy / (latent_energy.sum(dim=-1, keepdim=True) + 1e-12)
        latent_entropy = self._entropy(latent_prob)
        max_entropy = torch.log(torch.tensor(float(latent_prob.shape[-1]), device=lat.device)).clamp_min(1e-12)
        normalized_latent_entropy = latent_entropy / max_entropy
        latent_path_length = latent_delta_l2.sum(dim=-1) if latent_delta_l2.ndim > 1 else latent_delta_l2
        latent_direct_distance = torch.linalg.norm(lat[:, -1, :] - lat[:, 0, :], dim=-1)

        attention_proxy = torch.softmax(token_norms, dim=-1)
        attention_max = attention_proxy.max(dim=-1).values
        attention_sparsity = (attention_proxy < (1.0 / attention_proxy.shape[-1])).float().mean(dim=-1)

        action_delta = (
            act_steps[:, 1:, :] - act_steps[:, :-1, :] if act_steps.shape[1] > 1 else torch.zeros_like(act_steps[:, :0, :])
        )
        action_delta_l2 = torch.linalg.norm(action_delta, dim=-1) if action_delta.numel() else torch.zeros(1, device=lat.device)
        action_norm = torch.linalg.norm(act_steps, dim=-1)
        action_energy = act_flat.abs()
        action_prob = action_energy / (action_energy.sum(dim=-1, keepdim=True) + 1e-12)
        action_entropy = self._entropy(action_prob)
        max_action_entropy = torch.log(torch.tensor(float(action_prob.shape[-1]), device=lat.device)).clamp_min(1e-12)
        action_entropy_norm = action_entropy / max_action_entropy

        pooled_lat = lat.mean(dim=1)
        min_dim = min(pooled_lat.shape[-1], act_flat.shape[-1])
        video_action_cosine = F.cosine_similarity(pooled_lat[:, :min_dim], act_flat[:, :min_dim], dim=-1)
        prompt_norm = (
            torch.linalg.norm(prompt_embedding.detach().float()) if prompt_embedding is not None else torch.tensor(0.0, device=lat.device)
        )
        language_action_proxy = 1.0 / (1.0 + (prompt_norm / (torch.linalg.norm(act_flat, dim=-1).mean() + 1e-6)))

        noise_scale = video_sigma.detach().float().abs().mean()
        path_efficiency = latent_direct_distance / (latent_path_length + 1e-6)
        score_norm = token_norms / (noise_scale + 1e-6)

        metrics = {
            "video_latent_norm_mean": token_norms.mean(),
            "video_latent_norm_std": token_norms.std(unbiased=False),
            "video_latent_delta_l2_std": latent_delta_l2.std(unbiased=False),
            "video_latent_cosine_initial_final": latent_cos_initial_final.mean(),
            "video_latent_variance_mean": latent_var.mean(),
            "video_latent_entropy": normalized_latent_entropy.mean(),
            "flow_prediction_error": action_delta_l2.mean(),
            "latent_path_efficiency": path_efficiency.mean(),
            "attention_sparsity": attention_sparsity.mean(),
            "video_action_alignment_score": video_action_cosine.mean(),
            "decoder_hidden_norm_mean": action_norm.mean(),
            "decoder_hidden_norm_std": action_norm.std(unbiased=False),
            "mlp_activation_mean": act_flat.abs().mean(),
            "mlp_sparsity": (act_flat.abs() < 1e-3).float().mean(),
            "residual_stream_norm": torch.linalg.norm(pooled_lat, dim=-1).mean(),
            "action_entropy": action_entropy_norm.mean(),
            "action_uncertainty": act_steps.std(unbiased=False),
            "action_chunk_temporal_consistency": 1.0 / (1.0 + action_delta_l2.mean()),
            "action_chunk_smoothness": 1.0 / (1.0 + action_delta_l2.std(unbiased=False)),
            "goal_latent_distance": latent_direct_distance.mean(),
            "representation_drift_rate": latent_delta_l2.mean(),
            "plan_drift": action_delta_l2.sum(),
            "object_interaction_confidence": torch.sigmoid(action_norm.mean()),
            "instruction_attention_score": attention_max.mean() * language_action_proxy.mean(),
            "task_semantic_alignment": language_action_proxy.mean(),
            "latent_oscillation_score": (
                ((latent_delta[:, 1:, :] * latent_delta[:, :-1, :]).sum(dim=-1) < 0).float().mean()
                if latent_delta.shape[1] > 1
                else torch.tensor(0.0, device=lat.device)
            ),
            "video_action_cosine_similarity": video_action_cosine.mean(),
            "video_action_mutual_information": 1.0 - (normalized_latent_entropy.mean() * action_entropy_norm.mean()),
            "score_norm_mean": score_norm.mean(),
            "sampling_stability": 1.0 / (1.0 + score_norm.std(unbiased=False)),
            "latent_success_alignment": (path_efficiency.mean() + latent_cos_adj.mean()) / 2.0,
        }
        return {key: float(value.detach().float().cpu().item()) for key, value in metrics.items()}

    @staticmethod
    def _to_serializable_list(tensor: torch.Tensor) -> list:
        return tensor.detach().float().cpu().tolist()

    def _collect_encoder_hidden_layer_metrics(
        self,
        *,
        hidden_state_grid: torch.Tensor,
    ) -> dict[str, object]:
        hidden = hidden_state_grid.detach().float()
        hidden_norm = F.normalize(hidden, dim=-1, eps=1e-12)
        pooled_t_d = hidden_norm.mean(dim=(2, 3))
        pooled_delta_t_d = pooled_t_d[:, 1:, :] - pooled_t_d[:, :-1, :] if pooled_t_d.shape[1] > 1 else pooled_t_d[:, :0, :]
        pooled_adjacent_cos = (
            F.cosine_similarity(pooled_t_d[:, 1:, :], pooled_t_d[:, :-1, :], dim=-1)
            if pooled_t_d.shape[1] > 1
            else torch.empty((hidden.shape[0], 0), device=hidden.device, dtype=hidden.dtype)
        )
        token_adjacent_cos = (
            F.cosine_similarity(hidden_norm[:, 1:, :, :, :], hidden_norm[:, :-1, :, :, :], dim=-1).mean(dim=(2, 3))
            if hidden_norm.shape[1] > 1
            else torch.empty((hidden.shape[0], 0), device=hidden.device, dtype=hidden.dtype)
        )
        pooled_initial_final_cos = F.cosine_similarity(pooled_t_d[:, 0, :], pooled_t_d[:, -1, :], dim=-1)
        pooled_norms = torch.linalg.norm(pooled_t_d, dim=-1)
        pooled_delta_norms = (
            torch.linalg.norm(pooled_delta_t_d, dim=-1)
            if pooled_delta_t_d.numel()
            else torch.empty((hidden.shape[0], 0), device=hidden.device, dtype=hidden.dtype)
        )

        batch_idx = 0
        return {
            "encoder_hidden_state_shape": list(hidden.shape),
            "encoder_pooled_hw_l2_normalized_t_d": self._to_serializable_list(pooled_t_d[batch_idx]),
            "encoder_pooled_hw_l2_normalized_delta_t_d": self._to_serializable_list(pooled_delta_t_d[batch_idx]),
            "encoder_adjacent_pooled_cosine_t_minus_1": self._to_serializable_list(pooled_adjacent_cos[batch_idx]),
            "encoder_adjacent_token_cosine_mean_t_minus_1": self._to_serializable_list(token_adjacent_cos[batch_idx]),
            "encoder_initial_final_pooled_cosine": float(pooled_initial_final_cos[batch_idx].cpu().item()),
            "encoder_pooled_norm_mean": float(pooled_norms.mean().cpu().item()),
            "encoder_pooled_norm_std": float(pooled_norms.std(unbiased=False).cpu().item()),
            "encoder_delta_norm_mean": (
                float(pooled_delta_norms.mean().cpu().item()) if pooled_delta_norms.numel() else 0.0
            ),
            "encoder_delta_norm_std": (
                float(pooled_delta_norms.std(unbiased=False).cpu().item()) if pooled_delta_norms.numel() else 0.0
            ),
            "encoder_adjacent_pooled_cosine_mean": (
                float(pooled_adjacent_cos.mean().cpu().item()) if pooled_adjacent_cos.numel() else 1.0
            ),
            "encoder_adjacent_token_cosine_mean": (
                float(token_adjacent_cos.mean().cpu().item()) if token_adjacent_cos.numel() else 1.0
            ),
        }

    def _collect_encoder_hidden_diagnostics(
        self,
        *,
        hidden_states_by_layer: dict[int, torch.Tensor],
        video_sigma: torch.Tensor,
    ) -> dict[str, object]:
        layer_metrics = {
            int(layer_idx): self._collect_encoder_hidden_layer_metrics(hidden_state_grid=hidden_state_grid)
            for layer_idx, hidden_state_grid in sorted(hidden_states_by_layer.items())
        }
        return {
            "diagnostics_mode": self.diagnostics_mode,
            "encoder_decoder_xattn_layer_idx": int(self.world2action_pipeline.config.xattn_layer_idx),
            "encoder_capture_layer_indices": list(layer_metrics.keys()),
            "encoder_context_sigma": self._to_serializable_list(video_sigma),
            "layer_metrics": layer_metrics,
        }

    def _collect_decoder_hidden_layer_metrics(
        self,
        *,
        hidden_state_seq: torch.Tensor,
    ) -> dict[str, object]:
        hidden = hidden_state_seq.detach().float()
        hidden_norm = F.normalize(hidden, dim=-1, eps=1e-12)
        delta_t_d = hidden_norm[:, 1:, :] - hidden_norm[:, :-1, :] if hidden_norm.shape[1] > 1 else hidden_norm[:, :0, :]
        adjacent_cos = (
            F.cosine_similarity(hidden_norm[:, 1:, :], hidden_norm[:, :-1, :], dim=-1)
            if hidden_norm.shape[1] > 1
            else torch.empty((hidden.shape[0], 0), device=hidden.device, dtype=hidden.dtype)
        )
        initial_final_cos = F.cosine_similarity(hidden_norm[:, 0, :], hidden_norm[:, -1, :], dim=-1)
        norms = torch.linalg.norm(hidden_norm, dim=-1)
        delta_norms = (
            torch.linalg.norm(delta_t_d, dim=-1)
            if delta_t_d.numel()
            else torch.empty((hidden.shape[0], 0), device=hidden.device, dtype=hidden.dtype)
        )

        batch_idx = 0
        return {
            "decoder_hidden_state_shape": list(hidden.shape),
            "decoder_l2_normalized_t_d": self._to_serializable_list(hidden_norm[batch_idx]),
            "decoder_l2_normalized_delta_t_d": self._to_serializable_list(delta_t_d[batch_idx]),
            "decoder_adjacent_cosine_t_minus_1": self._to_serializable_list(adjacent_cos[batch_idx]),
            "decoder_initial_final_cosine": float(initial_final_cos[batch_idx].cpu().item()),
            "decoder_norm_mean": float(norms.mean().cpu().item()),
            "decoder_norm_std": float(norms.std(unbiased=False).cpu().item()),
            "decoder_delta_norm_mean": float(delta_norms.mean().cpu().item()) if delta_norms.numel() else 0.0,
            "decoder_delta_norm_std": float(delta_norms.std(unbiased=False).cpu().item()) if delta_norms.numel() else 0.0,
            "decoder_adjacent_cosine_mean": float(adjacent_cos.mean().cpu().item()) if adjacent_cos.numel() else 1.0,
        }

    def _collect_decoder_hidden_diagnostics(
        self,
        *,
        decoder_hidden_states_by_block: dict[int, torch.Tensor],
        obs_token_count: int,
        executed_action_count: int,
    ) -> dict[str, object]:
        layer_metrics: dict[int, dict[str, object]] = {}
        for block_idx, hidden_state in sorted(decoder_hidden_states_by_block.items()):
            action_hidden = hidden_state[:, obs_token_count:, :].detach().float()
            if action_hidden.shape[1] == 0:
                continue
            prefix_len = min(int(executed_action_count), int(action_hidden.shape[1]))
            prefix_hidden = action_hidden[:, :prefix_len, :] if prefix_len > 0 else action_hidden[:, :1, :]
            layer_metrics[int(block_idx)] = {
                "full_chunk": {
                    "action_subset": "full_chunk",
                    "action_count": int(action_hidden.shape[1]),
                    **self._collect_decoder_hidden_layer_metrics(hidden_state_seq=action_hidden),
                },
                "first_k_actions": {
                    "action_subset": "first_k_actions",
                    "action_count": int(prefix_len),
                    **self._collect_decoder_hidden_layer_metrics(hidden_state_seq=prefix_hidden),
                },
            }
        return {
            "diagnostics_mode": self.diagnostics_mode,
            "decoder_capture_block_indices": list(layer_metrics.keys()),
            "executed_action_count": int(executed_action_count),
            "layer_metrics": layer_metrics,
        }

    def _build_replay_payload(
        self,
        *,
        hidden_states_by_layer: dict[int, torch.Tensor],
        state_B_HO_O: torch.Tensor,
        video_sigma: torch.Tensor,
        actions: torch.Tensor,
        decoder_hidden_states_by_block: dict[int, torch.Tensor] | None,
        prompt: str,
        seed: int,
        num_sampling_step: int,
        stop_after_step: int | None,
        use_cuda_graphs: bool,
    ) -> dict[str, object]:
        decoder_layer_idx = int(self.world2action_pipeline.config.xattn_layer_idx)
        decoder_hidden_grid = hidden_states_by_layer[decoder_layer_idx].detach().cpu().to(dtype=torch.bfloat16)
        payload = {
            "prompt": prompt,
            "seed": int(seed),
            "num_sampling_step": int(num_sampling_step),
            "stop_after_step": int(stop_after_step) if stop_after_step is not None else None,
            "use_cuda_graphs": bool(use_cuda_graphs),
            "decoder_xattn_layer_idx": decoder_layer_idx,
            "representation_layer_indices": list(self.representation_layer_indices),
            "encoder_hidden_grid_B_T_H_W_D": decoder_hidden_grid,
            "encoder_hidden_grid_shape": list(decoder_hidden_grid.shape),
            "context_timesteps_B_1": video_sigma.unsqueeze(1).detach().cpu().to(dtype=torch.float32),
            "video_sigma_B": video_sigma.detach().cpu().to(dtype=torch.float32),
            "state_B_HO_O": state_B_HO_O.detach().cpu().to(dtype=torch.float32),
            "pred_actions_B_HA_A": actions.detach().cpu().to(dtype=torch.float32),
            "decoder_capture_block_indices": list(sorted((decoder_hidden_states_by_block or {}).keys())),
            "decoder_hidden_states_by_block": {
                int(block_idx): hidden_state.detach().cpu().to(dtype=torch.bfloat16)
                for block_idx, hidden_state in sorted((decoder_hidden_states_by_block or {}).items())
            },
        }
        if self.replay_payload_mode == "none":
            return {}
        if self.replay_payload_mode == "decoder_only":
            payload.pop("encoder_hidden_grid_B_T_H_W_D", None)
            payload.pop("encoder_hidden_grid_shape", None)
            payload.pop("context_timesteps_B_1", None)
            payload.pop("video_sigma_B", None)
            return payload
        return payload

    def _collect_diagnostics(
        self,
        *,
        hidden_states_by_layer: dict[int, torch.Tensor],
        decoder_hidden_states_by_block: dict[int, torch.Tensor],
        obs_token_count: int,
        executed_action_count: int,
        flat_crossattn_emb: torch.Tensor,
        actions: torch.Tensor,
        prompt_embedding: torch.Tensor | None,
        video_sigma: torch.Tensor,
    ) -> dict[str, object]:
        if self.diagnostics_mode == "encoder_hidden":
            return self._collect_encoder_hidden_diagnostics(
                hidden_states_by_layer=hidden_states_by_layer,
                video_sigma=video_sigma,
            )
        if self.diagnostics_mode == "decoder_hidden":
            return self._collect_decoder_hidden_diagnostics(
                decoder_hidden_states_by_block=decoder_hidden_states_by_block,
                obs_token_count=obs_token_count,
                executed_action_count=executed_action_count,
            )
        if self.diagnostics_mode == "all":
            default_metrics = self._collect_default_diagnostics(
                flat_crossattn_emb=flat_crossattn_emb, actions=actions,
                prompt_embedding=prompt_embedding, video_sigma=video_sigma,
            )
            encoder_metrics = self._collect_encoder_hidden_diagnostics(
                hidden_states_by_layer=hidden_states_by_layer, video_sigma=video_sigma,
            )
            decoder_metrics = self._collect_decoder_hidden_diagnostics(
                decoder_hidden_states_by_block=decoder_hidden_states_by_block,
                obs_token_count=obs_token_count, executed_action_count=executed_action_count,
            )
            return {**default_metrics, "encoder_hidden": encoder_metrics, "decoder_hidden": decoder_metrics}
        return self._collect_default_diagnostics(
            flat_crossattn_emb=flat_crossattn_emb,
            actions=actions,
            prompt_embedding=prompt_embedding,
            video_sigma=video_sigma,
        )

    def apply_fsdp(self, dp_mesh: DeviceMesh) -> None:
        self.video2world_pipeline.apply_fsdp(dp_mesh)
        self.world2action_pipeline.apply_fsdp(dp_mesh)

    def apply_cp(self) -> None:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(
        self,
        input_vid: torch.Tensor,
        state_B_HO_O: torch.Tensor,
        prompt: str,
        prompt_embedding: torch.Tensor | None = None,
        num_sampling_step: int = 35,
        stop_after_step: int | None = None,
        seed: int = 0,
        use_cuda_graphs: bool = False,
    ) -> torch.Tensor:
        if self.video2world_pipeline.text_guardrail_runner is not None:
            from cosmos_predict2.auxiliary.guardrail.common import presets as guardrail_presets

            if not guardrail_presets.run_text_guardrail(prompt, self.video2world_pipeline.text_guardrail_runner):
                msg = "Text guardrail error on prompt."
                raise RuntimeError(msg)

        T = input_vid.shape[2]
        assert T in {1, 5}

        decoder_layer_idx = int(self.world2action_pipeline.config.xattn_layer_idx)
        capture_layers = sorted({decoder_layer_idx, *self.representation_layer_indices})
        hidden_state_payload, video_sigma = self.video2world_pipeline.generate_video(
            vid_input=input_vid,
            num_latent_conditional_frames=1 if T == 1 else 2,
            prompt=prompt,
            prompt_embedding=prompt_embedding,
            negative_prompt="",
            guidance=0.0,
            num_sampling_step=num_sampling_step,
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
            return_context_at_step=stop_after_step,
            hidden_state_layer_idx=capture_layers,
        )
        if isinstance(hidden_state_payload, dict):
            hidden_states_by_layer = {int(layer_idx): tensor for layer_idx, tensor in hidden_state_payload.items()}
        else:
            hidden_states_by_layer = {capture_layers[0]: hidden_state_payload}
        crossattn_grid = hidden_states_by_layer[decoder_layer_idx]
        hidden_state_shape = crossattn_grid.shape
        crossattn_emb = crossattn_grid.reshape(hidden_state_shape[0], -1, hidden_state_shape[-1])

        world2action_out = self.world2action_pipeline(
            state_B_HO_O=state_B_HO_O,
            crossattn_emb=crossattn_emb,
            context_timesteps_B_1=video_sigma.unsqueeze(1),
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
            return_hidden_states=bool(self.decoder_capture_block_indices),
        )
        decoder_hidden_states_by_block: dict[int, torch.Tensor] = {}
        if self.decoder_capture_block_indices:
            actions, decoder_hidden_state_list = world2action_out
            for requested_block_idx in self.decoder_capture_block_indices:
                _, hidden_state = self._resolve_decoder_block_request(
                    decoder_hidden_state_list,
                    int(requested_block_idx),
                )
                decoder_hidden_states_by_block[int(requested_block_idx)] = hidden_state
        else:
            actions = world2action_out
        self.last_decoder_hidden_states_by_block = {
            int(block_idx): hidden_state.detach().cpu()
            for block_idx, hidden_state in decoder_hidden_states_by_block.items()
        }
        self.last_diagnostics = self._collect_diagnostics(
            hidden_states_by_layer=hidden_states_by_layer,
            decoder_hidden_states_by_block=decoder_hidden_states_by_block,
            obs_token_count=int(state_B_HO_O.shape[1]),
            executed_action_count=int(actions.shape[1]),
            flat_crossattn_emb=crossattn_emb,
            actions=actions,
            prompt_embedding=prompt_embedding,
            video_sigma=video_sigma,
        )
        self.last_replay_payload = self._build_replay_payload(
            hidden_states_by_layer=hidden_states_by_layer,
            state_B_HO_O=state_B_HO_O,
            video_sigma=video_sigma,
            actions=actions,
            decoder_hidden_states_by_block=decoder_hidden_states_by_block,
            prompt=prompt,
            seed=seed,
            num_sampling_step=num_sampling_step,
            stop_after_step=stop_after_step,
            use_cuda_graphs=use_cuda_graphs,
        )
        return actions
