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
    ) -> None:
        super().__init__()

        self.video2world_pipeline = video2world_pipeline
        self.world2action_pipeline = world2action_pipeline
        if diagnostics_mode not in {"default", "encoder_hidden"}:
            raise ValueError(f"Unsupported diagnostics_mode={diagnostics_mode!r}")
        self.diagnostics_mode = diagnostics_mode
        self.last_diagnostics: dict[str, object] = {}

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

    def _collect_encoder_hidden_diagnostics(
        self,
        *,
        hidden_state_grid: torch.Tensor,
        video_sigma: torch.Tensor,
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
            "diagnostics_mode": self.diagnostics_mode,
            "encoder_xattn_layer_idx": int(self.world2action_pipeline.config.xattn_layer_idx),
            "encoder_hidden_state_shape": list(hidden.shape),
            "encoder_context_sigma": self._to_serializable_list(video_sigma),
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

    def _collect_diagnostics(
        self,
        *,
        hidden_state_grid: torch.Tensor,
        flat_crossattn_emb: torch.Tensor,
        actions: torch.Tensor,
        prompt_embedding: torch.Tensor | None,
        video_sigma: torch.Tensor,
    ) -> dict[str, object]:
        if self.diagnostics_mode == "encoder_hidden":
            return self._collect_encoder_hidden_diagnostics(
                hidden_state_grid=hidden_state_grid,
                video_sigma=video_sigma,
            )
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

        crossattn_grid, video_sigma = self.video2world_pipeline.generate_video(
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
            hidden_state_layer_idx=self.world2action_pipeline.config.xattn_layer_idx,
        )
        hidden_state_shape = crossattn_grid.shape
        crossattn_emb = crossattn_grid.reshape(hidden_state_shape[0], -1, hidden_state_shape[-1])

        actions = self.world2action_pipeline(
            state_B_HO_O=state_B_HO_O,
            crossattn_emb=crossattn_emb,
            context_timesteps_B_1=video_sigma.unsqueeze(1),
            seed=seed,
            use_cuda_graphs=use_cuda_graphs,
        )
        self.last_diagnostics = self._collect_diagnostics(
            hidden_state_grid=crossattn_grid,
            flat_crossattn_emb=crossattn_emb,
            actions=actions,
            prompt_embedding=prompt_embedding,
            video_sigma=video_sigma,
        )
        return actions
