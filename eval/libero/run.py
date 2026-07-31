"""Run LIBERO evaluation with the Video Action Model (VAM) policy."""

from __future__ import annotations

import json
import os
import pathlib
import pickle
import random
import csv
import math
import time
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import imageio
import numpy as np
import torch
import tqdm
import tyro

_orig_torch_load = torch.load


def _torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load

from einops import rearrange
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from scipy.spatial.transform import Rotation

from cosmos_predict2.configs.config import make_config
from cosmos_predict2.data.action.utils import extract_normalization_types
from cosmos_predict2.pipelines.video2world import Video2WorldPipeline
from cosmos_predict2.pipelines.video2world2action import Video2World2ActionPipeline
from cosmos_predict2.pipelines.world2action import World2ActionPipeline
from imaginaire.lazy_config import instantiate
from imaginaire.utils.config_helper import override

from consensus_medoid import consensus_medoid_costs
from decoder_metric_select import (
    DECODER_LINEAR_COMBO_V1_NAME,
    DECODER_METRIC_SELECT_LAYER,
    DECODER_METRIC_SELECT_NAME,
    DECODER_METRIC_SELECT_REDUCE,
    DECODER_METRIC_SELECT_SUBSET,
    decoder_linear_combo_v1_scores,
    decoder_metric_selection_score,
)

LIBERO_SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_90": 600,
}


def get_max_steps_for_suite(task_suite_name: str) -> int:
    if task_suite_name in LIBERO_SUITE_MAX_STEPS:
        return LIBERO_SUITE_MAX_STEPS[task_suite_name]
    for base_suite in sorted(LIBERO_SUITE_MAX_STEPS, key=len, reverse=True):
        if task_suite_name == base_suite or task_suite_name.startswith(f"{base_suite}_"):
            return LIBERO_SUITE_MAX_STEPS[base_suite]
    raise ValueError(f"Task suite {task_suite_name} not available.")

CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
LIBERO_CONTROL_FREQ_HZ = 20
FAILURE_METRICS_MAX_TIMESTEPS = 120

DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

REGEN_FEATURES = [
    "chunk_action_delta_norm_mean", "flow_prediction_error",
    "action_chunk_temporal_consistency", "plan_drift",
    "action_entropy", "video_action_mutual_information",
    "mlp_activation_mean", "task_semantic_alignment",
    "video_latent_variance_mean", "representation_drift_rate",
    "decoder_hidden_norm_mean", "object_interaction_confidence",
    "video_latent_norm_std", "sampling_stability",
    "chunk_action_variance", "video_latent_entropy",
    "decoder_hidden_norm_std", "mlp_sparsity",
    "video_action_alignment_score", "video_action_cosine_similarity",
    "goal_latent_distance", "latent_success_alignment",
    "residual_stream_norm", "chunk_action_delta_norm_max",
    "attention_sparsity", "video_latent_delta_l2_std",
    "action_chunk_smoothness", "action_uncertainty",
    "instruction_attention_score", "chunk_gripper_switches",
    "latent_oscillation_score", "video_latent_cosine_initial_final",
    "video_latent_norm_mean", "score_norm_mean",
    "latent_path_efficiency", "query_latency_sec",
]

ACTION_REGEN_FEATURES = [
    "action_offset_in_chunk",
    "model_action_delta_norm",
    "model_action_norm",
    "model_gripper_command",
    "model_gripper_sign",
    "model_rotation6d_norm",
    "model_translation_norm",
]


def set_seed_everywhere(seed: int) -> None:
    """Sets the random seed for Python, NumPy, and PyTorch functions."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_video2world2action_pipeline(
    experiment_name: str,
    video_model_path: str,
    action_model_path: str,
    dataset_statistics_path: pathlib.Path,
    dtype: torch.dtype = torch.bfloat16,
    use_text_encoder: bool = True,
    diagnostics_mode: str = "default",
    representation_layer_indices: list[int] | None = None,
    decoder_capture_block_indices: list[int] | None = None,
    replay_payload_mode: str = "full",
) -> Video2World2ActionPipeline:
    """Instantiate the video-to-world-to-action pipeline and load normalizer statistics."""
    config = make_config()
    config = override(config, ["--", f"experiment={experiment_name}"])

    # all libero task descriptions have been verified to be unproblematic
    config.model.config.video_pipe_config.guardrail_config.enabled = False

    video2world_pipe = Video2WorldPipeline.from_config(
        config=config.model.config.video_pipe_config,
        dit_path=video_model_path,
        device="cuda",
        torch_dtype=dtype,
        load_ema_to_reg=False,
        use_text_encoder=use_text_encoder,
    )

    world2action_pipe = World2ActionPipeline.from_config(
        config.model.config.pipe_config,
        dit_path=action_model_path,
        device="cuda",
        dtype=dtype,
    )

    data_config = instantiate(config.data_config)

    with dataset_statistics_path.open("rb") as stats_file:
        stats = json.load(stats_file)
    world2action_pipe.normalizer.build_from_stats(
        stats,
        normalization_types=extract_normalization_types(data_config.policy_io.policy_io),
        concat_groups=data_config.policy_io.concat_groups,
        device="cuda",
        dtype=dtype,
    )

    return Video2World2ActionPipeline(
        video2world_pipe,
        world2action_pipe,
        diagnostics_mode=diagnostics_mode,
        representation_layer_indices=representation_layer_indices,
        decoder_capture_block_indices=decoder_capture_block_indices,
        replay_payload_mode=replay_payload_mode,
    ).cuda()


class VAMInference:
    """Helper class that maintains temporal context and queries the VAM policy."""

    def __init__(
        self,
        experiment_name: str,
        video_model_path: str,
        action_model_path: str,
        dataset_statistics_path: pathlib.Path,
        img_horizon: int,
        lowdim_horizon: int,
        stop_video_denoising_step: int,
        num_execute_actions: int,
        rollout_dir: pathlib.Path,
        t5_embeddings_path: pathlib.Path | None = None,
        use_cuda_graphs: bool = True,
        regen_model_path: pathlib.Path | None = None,
        regen_threshold: float = 0.38,
        regen_max_attempts: int = 0,
        regen_strategy: str = "none",
        regen_num_candidates: int = 3,
        decoder_metric_select_layer: int = DECODER_METRIC_SELECT_LAYER,
        decoder_metric_select_name: str = DECODER_METRIC_SELECT_NAME,
        decoder_metric_select_action_subset: str = DECODER_METRIC_SELECT_SUBSET,
        decoder_metric_select_reduce: str = DECODER_METRIC_SELECT_REDUCE,
        consensus_medoid_horizon: int | None = None,
        consensus_medoid_temporal_discount: float = 0.9,
        consensus_medoid_translation_weight: float = 1.0,
        consensus_medoid_rotation_weight: float = 0.5,
        consensus_medoid_gripper_weight: float = 0.25,
        consensus_medoid_continuity_weight: float = 0.25,
        consensus_medoid_smoothness_weight: float = 0.10,
        consensus_medoid_gripper_switch_weight: float = 0.10,
        action_regen_model_path: pathlib.Path | None = None,
        action_conf_threshold: float = 0.62,
        action_min_execute_actions: int = 3,
        action_max_execute_actions: int = 8,
        seed: int = 0,
        diagnostics_mode: str = "default",
        representation_layer_indices: list[int] | None = None,
        decoder_capture_block_indices: list[int] | None = None,
        replay_payload_mode: str = "full",
    ):
        self._t5_embeddings = self._load_t5_embeddings(t5_embeddings_path)
        self.diagnostics_mode = diagnostics_mode
        self.model = load_video2world2action_pipeline(
            experiment_name,
            video_model_path,
            action_model_path,
            dataset_statistics_path,
            use_text_encoder=t5_embeddings_path is None,
            diagnostics_mode=diagnostics_mode,
            representation_layer_indices=representation_layer_indices,
            decoder_capture_block_indices=decoder_capture_block_indices,
            replay_payload_mode=replay_payload_mode,
        )
        self._image_horizon = img_horizon
        self._lowdim_horizon = lowdim_horizon
        self.stop_video_denoising_step = stop_video_denoising_step
        self.num_execute_actions = num_execute_actions
        self.num_sampling_steps = 35
        self.rollout_dir = rollout_dir
        self.use_cuda_graphs = use_cuda_graphs
        self.seed = seed
        self._query_counter = 0
        self.regen_threshold = regen_threshold
        self.regen_max_attempts = regen_max_attempts
        self.regen_strategy = regen_strategy
        self.regen_num_candidates = regen_num_candidates
        self.decoder_metric_select_layer = decoder_metric_select_layer
        self.decoder_metric_select_name = decoder_metric_select_name
        self.decoder_metric_select_action_subset = decoder_metric_select_action_subset
        self.decoder_metric_select_reduce = decoder_metric_select_reduce
        self.consensus_medoid_horizon = int(consensus_medoid_horizon or num_execute_actions)
        self.consensus_medoid_temporal_discount = float(consensus_medoid_temporal_discount)
        self.consensus_medoid_translation_weight = float(consensus_medoid_translation_weight)
        self.consensus_medoid_rotation_weight = float(consensus_medoid_rotation_weight)
        self.consensus_medoid_gripper_weight = float(consensus_medoid_gripper_weight)
        self.consensus_medoid_continuity_weight = float(consensus_medoid_continuity_weight)
        self.consensus_medoid_smoothness_weight = float(consensus_medoid_smoothness_weight)
        self.consensus_medoid_gripper_switch_weight = float(consensus_medoid_gripper_switch_weight)
        self.regen_model, self.regen_features = self._load_regen_model(regen_model_path)
        self.action_regen_model = self._load_regen_model(action_regen_model_path)[0]
        self.action_conf_threshold = action_conf_threshold
        self.action_min_execute_actions = action_min_execute_actions
        self.action_max_execute_actions = action_max_execute_actions
        self.decoder_capture_block_indices = sorted(int(idx) for idx in (decoder_capture_block_indices or []))
        uses_chunk_selection = self.regen_strategy in {
            "catboost_select",
            "hybrid_select_action_dynamic",
            "decoder_metric_select",
            "consensus_medoid",
        }
        uses_catboost = self.regen_strategy in {"catboost_select", "hybrid_select_action_dynamic"}
        uses_action_dynamic = self.regen_strategy in {"action_dynamic", "hybrid_select_action_dynamic"}
        if uses_catboost and self.regen_model is None:
            raise ValueError(
                "regen_strategy in {'catboost_select','hybrid_select_action_dynamic'} requires --regen_model_path."
            )
        if uses_chunk_selection and self.regen_num_candidates < 2:
            raise ValueError("regen_num_candidates must be >= 2 for chunk selection strategies.")
        if uses_action_dynamic and self.action_regen_model is None:
            raise ValueError(
                "regen_strategy in {'action_dynamic','hybrid_select_action_dynamic'} "
                "requires --action_regen_model_path."
            )
        if self.diagnostics_mode == "decoder_hidden" and not self.decoder_capture_block_indices:
            raise ValueError("diagnostics_mode='decoder_hidden' requires decoder_capture_block_indices.")
        if self.regen_strategy == "decoder_metric_select":
            if self.diagnostics_mode != "decoder_hidden":
                raise ValueError("regen_strategy='decoder_metric_select' requires diagnostics_mode='decoder_hidden'.")
            if self.decoder_metric_select_layer not in self.decoder_capture_block_indices:
                raise ValueError(
                    "regen_strategy='decoder_metric_select' requires "
                    f"decoder_capture_block_indices to include {self.decoder_metric_select_layer}."
                )
            if self.decoder_metric_select_reduce not in {"min", "max"}:
                raise ValueError("decoder_metric_select_reduce must be either 'min' or 'max'.")
        if self.regen_strategy == "consensus_medoid":
            if self.consensus_medoid_horizon < 1:
                raise ValueError("consensus_medoid_horizon must be >= 1.")
            if not 0.0 < self.consensus_medoid_temporal_discount <= 1.0:
                raise ValueError("consensus_medoid_temporal_discount must be in (0, 1].")
        if self.action_min_execute_actions < 1:
            raise ValueError("action_min_execute_actions must be >= 1.")
        if self.action_max_execute_actions < self.action_min_execute_actions:
            raise ValueError("action_max_execute_actions must be >= action_min_execute_actions.")
        self.last_query_latency_sec: float | None = None
        self.last_query_actions: np.ndarray | None = None
        self.last_query_diagnostics: dict[str, float] | None = None
        self.last_query_regen_probability: float | None = None
        self.last_query_regen_attempts: int = 0
        self.last_query_was_regenerated: bool = False
        self.last_query_candidate_probs: list[float] = []
        self.last_query_candidate_chunk_metrics: list[dict[str, object]] = []
        self.last_query_selected_chunk_metrics: dict[str, object] | None = None
        self.last_query_selected_candidate_idx: int = 0
        self.last_query_action_success_probs: list[float] = []
        self.last_query_execute_horizon: int = 0
        self.last_query_representation_metrics: dict[str, object] | None = None
        self.last_raw_action: np.ndarray | None = None
        self.last_chunk_id: int | None = None
        self.last_action_offset_in_chunk: int | None = None
        self.last_step_used_query = False
        self._chunk_counter = -1
        self.reset(task_description="")

    @staticmethod
    def _load_t5_embeddings(t5_embeddings_path: pathlib.Path | None) -> dict[str, torch.Tensor] | None:
        if t5_embeddings_path is None:
            return None
        with pathlib.Path(t5_embeddings_path).open("rb") as f:
            embeddings = pickle.load(f)
        if not isinstance(embeddings, dict):
            raise TypeError(f"Expected dict in {t5_embeddings_path}, got {type(embeddings)!r}.")
        return embeddings

    @staticmethod
    def _load_regen_model(regen_model_path: pathlib.Path | None) -> tuple[object | None, list[str]]:
        if regen_model_path is None:
            return None, list(REGEN_FEATURES)
        from catboost import CatBoostClassifier

        model = CatBoostClassifier()
        model.load_model(str(regen_model_path))
        features_path = pathlib.Path(regen_model_path).parent / "catboost_chunk_regen_features.json"
        if features_path.is_file():
            with features_path.open(encoding="utf-8") as f:
                features = json.load(f)["features"]
        else:
            features = list(REGEN_FEATURES)
        return model, features

    def reset(self, task_description: str) -> None:
        """Reset internal state for a new task/episode."""
        self.task_description = task_description
        self._image_history: deque[np.ndarray] = deque(maxlen=(self._image_horizon - 1) * 4 + 1)
        self._lowdim_history: deque[np.ndarray] = deque(maxlen=self._lowdim_horizon)
        self.action_buffer: np.ndarray | None = None
        self.action_buffer_idx = 0
        self._execute_horizon = 0
        self.last_query_latency_sec = None
        self.last_query_actions = None
        self.last_query_diagnostics = None
        self.last_query_regen_probability = None
        self.last_query_regen_attempts = 0
        self.last_query_was_regenerated = False
        self.last_query_candidate_probs = []
        self.last_query_candidate_chunk_metrics = []
        self.last_query_selected_chunk_metrics = None
        self.last_query_selected_candidate_idx = 0
        self.last_query_action_success_probs = []
        self.last_query_execute_horizon = 0
        self.last_query_representation_metrics = None
        self.last_raw_action = None
        self.last_chunk_id = None
        self.last_action_offset_in_chunk = None
        self.last_step_used_query = False
        self._chunk_counter = -1

    def step(
        self,
        image: np.ndarray,
        task_description: str,
        obs: dict,
    ) -> np.ndarray:
        """Return the next action for the given observation."""
        if image.dtype != np.uint8:
            raise ValueError(f"Expected image dtype uint8, received {image.dtype}.")

        if task_description != self.task_description:
            self.reset(task_description)

        processed_image = self._process_image(image)
        self._add_image_to_history(processed_image)

        state_vec = self._state_from_observation(obs)
        self._add_lowdim_to_history(state_vec)
        self.last_step_used_query = False

        if self.action_buffer is None:
            self._query_policy(task_description)
            self.last_step_used_query = True

        current_action = self.action_buffer[self.action_buffer_idx]
        self.last_raw_action = current_action.copy()
        self.last_chunk_id = self._chunk_counter
        self.last_action_offset_in_chunk = self.action_buffer_idx
        self.action_buffer_idx += 1
        if self.action_buffer_idx >= self._execute_horizon:
            self.action_buffer = None

        return self._convert_action(current_action)

    def _sample_chunk(
        self,
        *,
        input_vid: torch.Tensor,
        state_tensor: torch.Tensor,
        task_description: str,
        sample_seed: int,
    ) -> tuple[np.ndarray, dict[str, float], dict[str, object], float]:
        with torch.no_grad():
            pred_actions = self.model(
                input_vid=input_vid,
                state_B_HO_O=state_tensor,
                prompt=task_description,
                prompt_embedding=self._prompt_embedding(task_description),
                num_sampling_step=self.num_sampling_steps,
                stop_after_step=self.stop_video_denoising_step,
                seed=sample_seed,
                use_cuda_graphs=self.use_cuda_graphs,
            )
        actions_np = pred_actions[0].float().cpu().numpy()
        diagnostics = getattr(self.model, "last_diagnostics", {}) or {}
        chunk_metrics = _chunk_action_metrics(actions_np)
        chunk_metrics.update(_clean_model_metrics(diagnostics))
        probability = self._regen_probability(chunk_metrics)
        return actions_np, diagnostics, chunk_metrics, probability

    def _query_policy(self, task_description: str) -> None:
        """Query the model and cache the planned action sequence."""
        images = np.concatenate(list(self._image_history)[::4], axis=1)  # downsample from 20 fps to 5
        lowdims = np.stack(list(self._lowdim_history), axis=0)

        input_vid = torch.from_numpy(images[None]).cuda().to(dtype=torch.bfloat16)
        state_tensor = torch.from_numpy(lowdims[None]).cuda().to(dtype=torch.bfloat16)

        start_time = time.perf_counter()
        base_seed = self.seed + self._query_counter * 100
        accepted_actions = None
        accepted_diagnostics = None
        accepted_probability = None
        attempts_used = 0
        candidate_probs: list[float] = []
        candidate_chunk_metrics: list[dict[str, object]] = []
        selected_candidate_idx = 0
        accepted_chunk_metrics: dict[str, object] | None = None

        if self.regen_strategy in {
            "catboost_select",
            "hybrid_select_action_dynamic",
            "decoder_metric_select",
            "consensus_medoid",
        }:
            candidates: list[tuple[np.ndarray, dict[str, float], dict[str, object], float]] = []
            candidate_scores: list[float] = []
            candidate_diagnostics_list: list[dict[str, object] | None] = []
            for candidate_idx in range(self.regen_num_candidates):
                actions_np, diagnostics, chunk_metrics, probability = self._sample_chunk(
                    input_vid=input_vid,
                    state_tensor=state_tensor,
                    task_description=task_description,
                    sample_seed=base_seed + candidate_idx,
                )
                chunk_metrics = dict(chunk_metrics)
                chunk_metrics["query_latency_sec"] = float(time.perf_counter() - start_time)
                chunk_metrics["catboost_failure_probability"] = float(probability)
                chunk_metrics["catboost_success_probability"] = float(1.0 - probability)
                if self.regen_strategy == "decoder_metric_select":
                    chunk_metrics["decoder_metric_selection_layer"] = int(self.decoder_metric_select_layer)
                    chunk_metrics["decoder_metric_selection_name"] = self.decoder_metric_select_name
                    chunk_metrics["decoder_metric_selection_action_subset"] = self.decoder_metric_select_action_subset
                    chunk_metrics["decoder_metric_selection_reduce"] = self.decoder_metric_select_reduce
                elif self.regen_strategy == "consensus_medoid":
                    chunk_metrics["consensus_horizon_config"] = int(self.consensus_medoid_horizon)
                    chunk_metrics["consensus_temporal_discount"] = self.consensus_medoid_temporal_discount
                    chunk_metrics["consensus_translation_weight"] = self.consensus_medoid_translation_weight
                    chunk_metrics["consensus_rotation_weight"] = self.consensus_medoid_rotation_weight
                    chunk_metrics["consensus_gripper_weight"] = self.consensus_medoid_gripper_weight
                    chunk_metrics["consensus_continuity_weight"] = self.consensus_medoid_continuity_weight
                    chunk_metrics["consensus_smoothness_weight"] = self.consensus_medoid_smoothness_weight
                    chunk_metrics["consensus_gripper_switch_weight"] = self.consensus_medoid_gripper_switch_weight
                candidates.append((actions_np, diagnostics, chunk_metrics, probability))
                candidate_diagnostics_list.append(diagnostics)
                candidate_probs.append(probability)
                candidate_chunk_metrics.append(chunk_metrics)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if self.regen_strategy == "consensus_medoid":
                candidate_scores, medoid_details, pairwise_distances = consensus_medoid_costs(
                    [candidate[0] for candidate in candidates],
                    previous_action=self.last_raw_action,
                    horizon=self.consensus_medoid_horizon,
                    temporal_discount=self.consensus_medoid_temporal_discount,
                    translation_weight=self.consensus_medoid_translation_weight,
                    rotation_weight=self.consensus_medoid_rotation_weight,
                    gripper_weight=self.consensus_medoid_gripper_weight,
                    continuity_weight=self.consensus_medoid_continuity_weight,
                    smoothness_weight=self.consensus_medoid_smoothness_weight,
                    gripper_switch_weight=self.consensus_medoid_gripper_switch_weight,
                )
                selected_candidate_idx = int(np.argmin(candidate_scores))
                for chunk_metrics, detail in zip(candidate_chunk_metrics, medoid_details):
                    chunk_metrics.update(detail)
                    chunk_metrics["consensus_pairwise_distance_matrix"] = pairwise_distances
                    chunk_metrics["consensus_selected_by"] = "minimum_cost"
            elif self.regen_strategy == "decoder_metric_select":
                if self.decoder_metric_select_name == DECODER_LINEAR_COMBO_V1_NAME:
                    candidate_scores, combo_details = decoder_linear_combo_v1_scores(
                        candidate_diagnostics_list,
                        layer_idx=self.decoder_metric_select_layer,
                        action_subset=self.decoder_metric_select_action_subset,
                    )
                    for chunk_metrics, score, detail in zip(candidate_chunk_metrics, candidate_scores, combo_details):
                        chunk_metrics["decoder_metric_selection_score"] = float(score)
                        chunk_metrics.update(detail)
                else:
                    for chunk_metrics, diagnostics in zip(candidate_chunk_metrics, candidate_diagnostics_list):
                        selection_score = decoder_metric_selection_score(
                            diagnostics,
                            layer_idx=self.decoder_metric_select_layer,
                            metric_name=self.decoder_metric_select_name,
                            action_subset=self.decoder_metric_select_action_subset,
                        )
                        chunk_metrics["decoder_metric_selection_score"] = float(selection_score)
                        candidate_scores.append(selection_score)
                if self.decoder_metric_select_reduce == "min":
                    selected_candidate_idx = int(np.argmin(candidate_scores))
                else:
                    selected_candidate_idx = int(np.argmax(candidate_scores))
            else:
                selected_candidate_idx = int(np.argmin(candidate_probs))
            accepted_actions, accepted_diagnostics, accepted_metrics, accepted_probability = candidates[selected_candidate_idx]
            accepted_chunk_metrics = dict(accepted_metrics)
            attempts_used = self.regen_num_candidates - 1
        else:
            for attempt in range(max(self.regen_max_attempts, 0) + 1):
                actions_np, diagnostics, chunk_metrics, probability = self._sample_chunk(
                    input_vid=input_vid,
                    state_tensor=state_tensor,
                    task_description=task_description,
                    sample_seed=base_seed + attempt,
                )
                elapsed = time.perf_counter() - start_time
                chunk_metrics["query_latency_sec"] = elapsed

                accepted_actions = actions_np
                accepted_diagnostics = diagnostics
                accepted_probability = probability
                attempts_used = attempt
                accepted_chunk_metrics = dict(chunk_metrics)
                if (
                    self.regen_strategy != "threshold"
                    or self.regen_model is None
                    or probability < self.regen_threshold
                    or attempt >= self.regen_max_attempts
                ):
                    break

        self.last_query_latency_sec = time.perf_counter() - start_time
        self.action_buffer = accepted_actions
        self.last_query_actions = self.action_buffer.copy()
        self.last_query_diagnostics = accepted_diagnostics
        self.last_query_regen_probability = accepted_probability
        self.last_query_regen_attempts = attempts_used
        self.last_query_candidate_probs = candidate_probs
        self.last_query_candidate_chunk_metrics = candidate_chunk_metrics
        self.last_query_selected_candidate_idx = selected_candidate_idx
        self.last_query_selected_chunk_metrics = accepted_chunk_metrics
        self.last_query_action_success_probs = []
        if self.diagnostics_mode == "encoder_hidden":
            self.last_query_representation_metrics = _structured_model_metrics(self.last_query_diagnostics)
        else:
            self.last_query_representation_metrics = None
        if self.regen_strategy in {
            "catboost_select",
            "hybrid_select_action_dynamic",
            "decoder_metric_select",
            "consensus_medoid",
        }:
            self.last_query_was_regenerated = selected_candidate_idx > 0
        else:
            self.last_query_was_regenerated = attempts_used > 0
        if self.regen_strategy in {"action_dynamic", "hybrid_select_action_dynamic"}:
            success_probs = self._action_success_probabilities(self.last_query_actions)
            self.last_query_action_success_probs = [float(p) for p in success_probs]
            self._execute_horizon = self._dynamic_execute_horizon(success_probs)
        else:
            self._execute_horizon = self.num_execute_actions
        self.last_query_execute_horizon = int(self._execute_horizon)
        self.action_buffer_idx = 0
        self._chunk_counter += 1
        self._query_counter += 1

    def _action_success_probabilities(self, planned_actions: np.ndarray) -> np.ndarray:
        if self.action_regen_model is None:
            return np.ones((planned_actions.shape[0],), dtype=np.float32)
        rows: list[list[float]] = []
        previous = self.last_raw_action.copy() if self.last_raw_action is not None else None
        for action_offset, action in enumerate(planned_actions):
            metrics = _action_metrics(action, previous)
            metrics["action_offset_in_chunk"] = int(action_offset)
            row = [float(metrics.get(feature, np.nan)) for feature in ACTION_REGEN_FEATURES]
            rows.append(row)
            previous = action
        fail_prob = self.action_regen_model.predict_proba(rows)[:, 1]
        return 1.0 - fail_prob

    def _dynamic_execute_horizon(self, success_probs: np.ndarray) -> int:
        if success_probs.size == 0:
            return self.action_min_execute_actions
        max_actions = min(self.action_max_execute_actions, int(success_probs.shape[0]))
        horizon = min(self.action_min_execute_actions, max_actions)
        for idx in range(horizon, max_actions):
            if float(success_probs[idx]) >= float(self.action_conf_threshold):
                horizon = idx + 1
            else:
                break
        return int(max(1, horizon))

    def _prompt_embedding(self, task_description: str) -> torch.Tensor | None:
        if self._t5_embeddings is None:
            return None
        candidates = [
            task_description,
            task_description.replace("bowl", "black bowl"),
            task_description.replace("black bowl", "bowl"),
        ]
        for key in candidates:
            if key in self._t5_embeddings:
                return self._t5_embeddings[key].cuda().to(dtype=torch.bfloat16)
        raise KeyError(f"No precomputed T5 embedding for {task_description!r}. Example keys: {list(self._t5_embeddings)[:5]}")

    def _regen_probability(self, metrics: dict[str, object]) -> float:
        if self.regen_model is None:
            return float("nan")
        row = [[float(metrics.get(feature, np.nan)) for feature in self.regen_features]]
        return float(self.regen_model.predict_proba(row)[0, 1])

    def _process_image(self, image: np.ndarray) -> np.ndarray:
        tensor = rearrange(image, "h w c -> c h w")[:, None, :, :]
        return 2.0 * (tensor.astype(np.float32) / 255.0 - 0.5)

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self._image_history.append(image)
        while len(self._image_history) < self._image_history.maxlen:
            self._image_history.append(image.copy())

    def _add_lowdim_to_history(self, lowdim: np.ndarray) -> None:
        self._lowdim_history.append(lowdim)
        while len(self._lowdim_history) < self._lowdim_horizon:
            self._lowdim_history.append(lowdim.copy())

    @staticmethod
    def _state_from_observation(obs: dict[str, np.ndarray]) -> np.ndarray:
        rot_6d = Rotation.from_quat(obs["robot0_eef_quat"]).as_matrix()[:2].reshape((6,))
        return np.concatenate((obs["robot0_eef_pos"], rot_6d, obs["robot0_gripper_qpos"][0][None]), axis=0)

    @staticmethod
    def _matrix_from_6d(orient6: np.ndarray) -> np.ndarray:
        r1 = orient6[:3]
        r2 = orient6[3:]
        r1_norm = r1 / (np.linalg.norm(r1) + 1e-9)
        r2_orth = r2 - np.dot(r2, r1_norm) * r1_norm
        r2_norm = r2_orth / (np.linalg.norm(r2_orth) + 1e-9)
        r3 = np.cross(r1_norm, r2_norm)
        return np.stack([r1_norm, r2_norm, r3], axis=0)

    def _convert_action(self, action: np.ndarray) -> np.ndarray:
        delta_pos = action[:3]
        rot_matrix = self._matrix_from_6d(action[3:9])
        rot_vec = Rotation.from_matrix(rot_matrix).as_rotvec()
        gripper = np.sign(action[9][None])
        return np.concatenate([delta_pos, rot_vec, gripper], axis=0)


def get_libero_env(task) -> tuple[OffScreenRenderEnv, str]:
    """Initializes and returns the LIBERO environment alongside the task description."""
    task_description = task.language.replace("black bowl", "bowl")
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": CAMERA_HEIGHT,
        "camera_widths": CAMERA_WIDTH,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)
    return env, task_description


def get_libero_image(obs: dict[str, np.ndarray]) -> np.ndarray:
    """Extract the agentview image and check that it matches the expected resolution."""
    image = obs["agentview_image"][::-1, ::-1]
    if image.shape != (CAMERA_HEIGHT, CAMERA_WIDTH, 3):
        raise ValueError(f"Unexpected agentview image shape {image.shape}.")
    return image


def save_rollout_video(
    rollout_images: Iterable[np.ndarray],
    idx: int,
    success: bool,
    task_description: str,
    rollout_dir: Path,
) -> Path:
    """Save an MP4 replay of the episode."""
    rollout_dir.mkdir(parents=True, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")
    mp4_path = rollout_dir / f"episode{idx}_{'success' if success else 'failure'}_{processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=20)
    try:
        for img in rollout_images:
            video_writer.append_data(img)
    finally:
        video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    return mp4_path


def _clean_model_metrics(diagnostics: dict[str, float] | None) -> dict[str, float]:
    if not diagnostics:
        return {}
    return {
        key: float(value)
        for key, value in diagnostics.items()
        if isinstance(value, (int, float, np.integer, np.floating)) and math.isfinite(float(value))
    }


def _structured_model_metrics(diagnostics: dict[str, object] | None) -> dict[str, object]:
    if not diagnostics:
        return {}
    result: dict[str, object] = {}
    for key, value in diagnostics.items():
        if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
            result[key] = _jsonable(value)
        elif isinstance(value, (list, tuple, dict, np.ndarray)):
            result[key] = _jsonable(value)
    return result


def _chunk_action_metrics(raw_chunk: np.ndarray | None) -> dict[str, object]:
    chunk = np.asarray(raw_chunk, dtype=np.float64) if raw_chunk is not None else np.empty((0, 10), dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[0] == 0:
        return {
            "predicted_action_count": 0,
            "chunk_action_variance": float("nan"),
            "chunk_action_delta_norm_mean": float("nan"),
            "chunk_action_delta_norm_max": float("nan"),
            "chunk_gripper_switches": 0,
        }
    continuous = chunk[:, :9]
    deltas = np.linalg.norm(np.diff(continuous, axis=0), axis=1) if chunk.shape[0] > 1 else np.empty((0,))
    gripper = chunk[:, 9] if chunk.shape[1] > 9 else np.empty((0,))
    return {
        "predicted_action_count": int(chunk.shape[0]),
        "chunk_action_variance": float(np.var(continuous)),
        "chunk_action_delta_norm_mean": float(deltas.mean()) if deltas.size else 0.0,
        "chunk_action_delta_norm_max": float(deltas.max()) if deltas.size else 0.0,
        "chunk_gripper_switches": int(np.sum(np.diff(np.sign(gripper)) != 0)) if gripper.size > 1 else 0,
    }


def _action_metrics(raw_action: np.ndarray, previous_raw_action: np.ndarray | None) -> dict[str, object]:
    raw = np.asarray(raw_action, dtype=np.float64)
    continuous = raw[:9]
    previous = np.asarray(previous_raw_action, dtype=np.float64) if previous_raw_action is not None else None
    delta_norm = float(np.linalg.norm(continuous - previous[:9])) if previous is not None else float("nan")
    return {
        "model_action": raw.tolist(),
        "model_delta_position": raw[:3].tolist(),
        "model_rotation_6d": raw[3:9].tolist(),
        "model_gripper_command": float(raw[9]) if raw.shape[0] > 9 else float("nan"),
        "model_gripper_sign": int(np.sign(raw[9])) if raw.shape[0] > 9 else 0,
        "model_action_norm": float(np.linalg.norm(continuous)),
        "model_translation_norm": float(np.linalg.norm(raw[:3])),
        "model_rotation6d_norm": float(np.linalg.norm(raw[3:9])),
        "model_action_delta_norm": delta_norm,
    }


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_jsonable(row), allow_nan=True) + "\n")


def _write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: json.dumps(value) if isinstance(value, (list, dict)) else value for key, value in row.items()} for row in rows])


def _flatten_chunk_rows(episode_traces: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in episode_traces:
        meta = episode["meta"]
        for chunk in episode["chunks"]:
            row = {
                "total_episode_idx": meta["total_episode_idx"],
                "task_id": meta["task_id"],
                "episode_idx": meta["episode_idx"],
                "task_description": meta["task_description"],
                "success": meta["success"],
                "termination_reason": meta["termination_reason"],
                "chunk_id": chunk["chunk_id"],
                "inference_step_idx": chunk["inference_step_idx"],
                "query_timestep": chunk["query_timestep"],
                "query_latency_sec": chunk["query_latency_sec"],
                "executed_action_count": len(chunk["actions"]),
                "regen_probability": chunk.get("regen_probability"),
                "regen_attempts": chunk.get("regen_attempts"),
                "was_regenerated": chunk.get("was_regenerated"),
                "regen_strategy": chunk.get("regen_strategy"),
                "num_candidates": chunk.get("num_candidates"),
                "selected_candidate_idx": chunk.get("selected_candidate_idx"),
                "candidate_regen_probabilities": chunk.get("candidate_regen_probabilities"),
                "execute_horizon": chunk.get("execute_horizon"),
                "action_conf_threshold": chunk.get("action_conf_threshold"),
            }
            row.update(chunk["metrics"])
            rows.append(row)
    return rows


def _flatten_candidate_chunk_rows(episode_traces: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in episode_traces:
        meta = episode["meta"]
        for chunk in episode["chunks"]:
            for candidate in chunk.get("candidate_chunk_metrics") or []:
                row = {
                    "total_episode_idx": meta["total_episode_idx"],
                    "task_id": meta["task_id"],
                    "episode_idx": meta["episode_idx"],
                    "task_description": meta["task_description"],
                    "success": meta["success"],
                    "termination_reason": meta["termination_reason"],
                    "chunk_id": chunk["chunk_id"],
                    "inference_step_idx": chunk["inference_step_idx"],
                    "query_timestep": chunk["query_timestep"],
                    "selected_candidate_idx": chunk.get("selected_candidate_idx"),
                    "candidate_idx": candidate["candidate_idx"],
                    "catboost_failure_probability": candidate.get("catboost_failure_probability"),
                    "catboost_success_probability": candidate.get("catboost_success_probability"),
                    "is_selected": int(candidate["candidate_idx"]) == int(chunk.get("selected_candidate_idx", -1)),
                }
                row.update(candidate.get("metrics") or {})
                rows.append(row)
    return rows


def _flatten_representation_rows(episode_traces: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in episode_traces:
        meta = episode["meta"]
        for chunk in episode["chunks"]:
            representation_metrics = chunk.get("representation_metrics") or {}
            if not representation_metrics:
                continue
            row = {
                "total_episode_idx": meta["total_episode_idx"],
                "task_id": meta["task_id"],
                "episode_idx": meta["episode_idx"],
                "task_description": meta["task_description"],
                "success": meta["success"],
                "termination_reason": meta["termination_reason"],
                "chunk_id": chunk["chunk_id"],
                "inference_step_idx": chunk["inference_step_idx"],
                "query_timestep": chunk["query_timestep"],
                "query_latency_sec": chunk["query_latency_sec"],
            }
            row.update(representation_metrics)
            rows.append(row)
    return rows


def _flatten_action_rows(episode_traces: list[dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode in episode_traces:
        meta = episode["meta"]
        for chunk in episode["chunks"]:
            for action in chunk["actions"]:
                row = {
                    "total_episode_idx": meta["total_episode_idx"],
                    "task_id": meta["task_id"],
                    "episode_idx": meta["episode_idx"],
                    "task_description": meta["task_description"],
                    "success": meta["success"],
                    "termination_reason": meta["termination_reason"],
                    "chunk_id": chunk["chunk_id"],
                    "inference_step_idx": chunk["inference_step_idx"],
                    "action_index": action["action_index"],
                    "timestep": action["timestep"],
                    "action_offset_in_chunk": action["action_offset_in_chunk"],
                }
                row.update(action["metrics"])
                rows.append(row)
    return rows


def _trim_failure_episode_metrics(episode_trace: dict[str, object]) -> dict[str, object]:
    meta = dict(episode_trace["meta"])
    if bool(meta.get("success")):
        return episode_trace
    trimmed_chunks = []
    for chunk in episode_trace["chunks"]:
        if int(chunk["query_timestep"]) >= FAILURE_METRICS_MAX_TIMESTEPS:
            continue
        kept_actions = [action for action in chunk["actions"] if int(action["timestep"]) < FAILURE_METRICS_MAX_TIMESTEPS]
        if not kept_actions:
            continue
        new_chunk = dict(chunk)
        new_chunk["actions"] = kept_actions
        trimmed_chunks.append(new_chunk)
    meta["inference_step_count"] = len(trimmed_chunks)
    meta["chunk_count"] = len(trimmed_chunks)
    meta["action_count"] = int(sum(len(chunk["actions"]) for chunk in trimmed_chunks))
    meta["metrics_max_timesteps"] = FAILURE_METRICS_MAX_TIMESTEPS
    return {"meta": meta, "chunks": trimmed_chunks}


def _write_trace_outputs(metrics_dir: Path, episode_traces: list[dict[str, object]]) -> None:
    episode_traces = [_trim_failure_episode_metrics(ep) for ep in episode_traces]
    chunk_rows = _flatten_chunk_rows(episode_traces)
    candidate_rows = _flatten_candidate_chunk_rows(episode_traces)
    representation_rows = _flatten_representation_rows(episode_traces)
    action_rows = _flatten_action_rows(episode_traces)
    _write_jsonl(metrics_dir / "episode_traces.jsonl", episode_traces)
    _write_jsonl(metrics_dir / "chunk_metrics.jsonl", chunk_rows)
    _write_jsonl(metrics_dir / "candidate_chunk_metrics.jsonl", candidate_rows)
    _write_jsonl(metrics_dir / "representation_metrics.jsonl", representation_rows)
    _write_jsonl(metrics_dir / "action_metrics.jsonl", action_rows)
    _write_csv_rows(metrics_dir / "chunk_metrics.csv", chunk_rows)
    _write_csv_rows(metrics_dir / "candidate_chunk_metrics.csv", candidate_rows)
    _write_csv_rows(metrics_dir / "action_metrics.csv", action_rows)
    summary = {
        "num_episodes": len(episode_traces),
        "num_successes": int(sum(bool(ep["meta"]["success"]) for ep in episode_traces)),
        "success_rate": (sum(bool(ep["meta"]["success"]) for ep in episode_traces) / max(len(episode_traces), 1)),
        "total_chunks": int(sum(ep["meta"]["chunk_count"] for ep in episode_traces)),
        "total_actions": int(sum(ep["meta"]["action_count"] for ep in episode_traces)),
        "total_regenerated_chunks": int(sum(int(chunk.get("was_regenerated", False)) for ep in episode_traces for chunk in ep["chunks"])),
    }
    (metrics_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def _termination_reason(success: bool, exception: str | None, step_count: int, max_steps: int) -> str:
    timeout = (not success) and exception is None and step_count >= max_steps
    return "success" if success else "exception" if exception else "timeout" if timeout else "stopped"


def run_episode(
    env: OffScreenRenderEnv,
    policy: VAMInference,
    task_description: str,
    initial_observation: dict[str, np.ndarray],
    max_steps: int,
    num_steps_wait: int,
) -> tuple[bool, list[np.ndarray], dict[str, object]]:
    """Execute a single episode and return success flag along with captured frames."""
    obs = initial_observation
    replay_images: list[np.ndarray] = []
    trace: dict[str, object] = {"chunks": [], "action_count": 0}
    success = False
    exception: str | None = None
    previous_raw_action: np.ndarray | None = None

    try:
        for step_idx in range(max_steps + num_steps_wait):
            if step_idx < num_steps_wait:
                obs, _, done, info = env.step(DUMMY_ACTION)
                if done:
                    success = True
                    break
                continue

            image = get_libero_image(obs)
            replay_images.append(image)

            action = policy.step(image, task_description, obs)

            if policy.last_step_used_query:
                if policy.last_query_selected_chunk_metrics is not None:
                    chunk_metrics = dict(policy.last_query_selected_chunk_metrics)
                else:
                    chunk_metrics = _chunk_action_metrics(policy.last_query_actions)
                    chunk_metrics.update(_clean_model_metrics(policy.last_query_diagnostics))
                    chunk_metrics["query_latency_sec"] = float(policy.last_query_latency_sec or 0.0)
                candidate_rows = [
                    {
                        "candidate_idx": int(idx),
                        "catboost_failure_probability": float(prob),
                        "catboost_success_probability": float(1.0 - prob),
                        "metrics": dict(metrics),
                    }
                    for idx, (prob, metrics) in enumerate(
                        zip(policy.last_query_candidate_probs, policy.last_query_candidate_chunk_metrics)
                    )
                ]
                trace["chunks"].append(
                    {
                        "chunk_id": int(policy.last_chunk_id if policy.last_chunk_id is not None else len(trace["chunks"])),
                        "inference_step_idx": int(len(trace["chunks"])),
                        "query_timestep": int(step_idx - num_steps_wait),
                        "query_latency_sec": float(policy.last_query_latency_sec or 0.0),
                        "regen_probability": policy.last_query_regen_probability,
                        "regen_attempts": int(policy.last_query_regen_attempts),
                        "was_regenerated": bool(policy.last_query_was_regenerated),
                        "regen_strategy": policy.regen_strategy,
                        "num_candidates": (
                            int(policy.regen_num_candidates)
                            if policy.regen_strategy in {"catboost_select", "hybrid_select_action_dynamic"}
                            else int(policy.last_query_regen_attempts) + 1
                        ),
                        "selected_candidate_idx": int(policy.last_query_selected_candidate_idx),
                        "candidate_regen_probabilities": list(policy.last_query_candidate_probs),
                        "candidate_chunk_metrics": candidate_rows,
                        "execute_horizon": int(policy.last_query_execute_horizon),
                        "action_conf_threshold": float(policy.action_conf_threshold),
                        "action_success_probabilities": list(policy.last_query_action_success_probs),
                        "representation_metrics": dict(policy.last_query_representation_metrics or {}),
                        "metrics": chunk_metrics,
                        "actions": [],
                    }
                )

            if not trace["chunks"]:
                raise RuntimeError("No model chunk available for executed action.")
            if policy.last_raw_action is None or policy.last_chunk_id is None or policy.last_action_offset_in_chunk is None:
                raise RuntimeError("Missing raw model action metadata.")

            action_record = {
                "action_index": int(trace["action_count"]),
                "timestep": int(step_idx - num_steps_wait),
                "chunk_id": int(policy.last_chunk_id),
                "action_offset_in_chunk": int(policy.last_action_offset_in_chunk),
                "metrics": _action_metrics(policy.last_raw_action, previous_raw_action),
            }
            trace["chunks"][-1]["actions"].append(action_record)
            trace["action_count"] = int(trace["action_count"]) + 1
            previous_raw_action = policy.last_raw_action.copy()

            obs, _, done, info = env.step(action.tolist())
            if done:
                success = True
                break
    except Exception as exc:
        exception = repr(exc)
        print(f"Episode failed with exception: {exception}")

    trace["exception"] = exception
    return success, replay_images, trace


def eval_vam_libero(
    vam_experiment_name: str,
    vam_video_model_path: str,
    vam_action_model_path: pathlib.Path,
    vam_dataset_statistics_path: pathlib.Path,
    vam_img_horizon: int,
    vam_lowdim_horizon: int,
    vam_stop_video_denoising_step: int,
    vam_num_execute_actions: int,
    task_suite_name: str,
    num_trials_per_task: int = 50,
    eval_rank: int = 0,
    eval_world_size: int = 1,
    num_steps_wait: int = 10,
    seed: int = 0,
    max_eval_episodes: int | None = None,
    rollout_dir: pathlib.Path | None = None,
    metrics_dir: pathlib.Path | None = None,
    t5_embeddings_path: pathlib.Path | None = None,
    use_cuda_graphs: bool = True,
    regen_model_path: pathlib.Path | None = None,
    regen_threshold: float = 0.38,
    regen_max_attempts: int = 0,
    regen_strategy: str = "none",
    regen_num_candidates: int = 3,
    action_regen_model_path: pathlib.Path | None = None,
    action_conf_threshold: float = 0.62,
    action_min_execute_actions: int = 3,
    action_max_execute_actions: int = 8,
    selected_episodes: str = "",
    max_control_steps: int = FAILURE_METRICS_MAX_TIMESTEPS,
    append_metrics: bool = False,
    consensus_medoid_horizon: int | None = None,
    consensus_medoid_temporal_discount: float = 0.9,
    consensus_medoid_translation_weight: float = 1.0,
    consensus_medoid_rotation_weight: float = 0.5,
    consensus_medoid_gripper_weight: float = 0.25,
    consensus_medoid_continuity_weight: float = 0.25,
    consensus_medoid_smoothness_weight: float = 0.10,
    consensus_medoid_gripper_switch_weight: float = 0.10,
    save_rollout_videos: bool = True,
) -> None:
    set_seed_everywhere(seed)

    run_label = (
        f"{vam_action_model_path.stem}_stopafter{vam_stop_video_denoising_step}_execute{vam_num_execute_actions}"
    )
    rollout_dir = rollout_dir or (Path("./results") / run_label / task_suite_name)
    rollout_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = metrics_dir or (rollout_dir / "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    selected_pairs = {
        tuple(int(part) for part in item.split(":"))
        for item in selected_episodes.split(",")
        if item.strip()
    }

    policy = VAMInference(
        vam_experiment_name,
        vam_video_model_path,
        str(vam_action_model_path),
        vam_dataset_statistics_path,
        vam_img_horizon,
        vam_lowdim_horizon,
        vam_stop_video_denoising_step,
        vam_num_execute_actions,
        rollout_dir,
        t5_embeddings_path,
        use_cuda_graphs,
        regen_model_path,
        regen_threshold=regen_threshold,
        regen_max_attempts=regen_max_attempts,
        regen_strategy=regen_strategy,
        regen_num_candidates=regen_num_candidates,
        consensus_medoid_horizon=consensus_medoid_horizon,
        consensus_medoid_temporal_discount=consensus_medoid_temporal_discount,
        consensus_medoid_translation_weight=consensus_medoid_translation_weight,
        consensus_medoid_rotation_weight=consensus_medoid_rotation_weight,
        consensus_medoid_gripper_weight=consensus_medoid_gripper_weight,
        consensus_medoid_continuity_weight=consensus_medoid_continuity_weight,
        consensus_medoid_smoothness_weight=consensus_medoid_smoothness_weight,
        consensus_medoid_gripper_switch_weight=consensus_medoid_gripper_switch_weight,
        action_regen_model_path=action_regen_model_path,
        action_conf_threshold=action_conf_threshold,
        action_min_execute_actions=action_min_execute_actions,
        action_max_execute_actions=action_max_execute_actions,
        seed=seed,
    )

    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"Task suite {task_suite_name} not available.")
    task_suite = benchmark_dict[task_suite_name]()
    num_tasks = task_suite.n_tasks

    max_steps = min(max_control_steps, get_max_steps_for_suite(task_suite_name))
    print(
        f"Using max_control_steps={max_steps} "
        f"(regen_strategy={regen_strategy}, regen_threshold={regen_threshold}, "
        f"regen_max_attempts={regen_max_attempts}, regen_num_candidates={regen_num_candidates}, "
        f"action_conf_threshold={action_conf_threshold}, "
        f"action_min_execute_actions={action_min_execute_actions}, "
        f"action_max_execute_actions={action_max_execute_actions})"
    )

    episode_traces: list[dict[str, object]] = []
    if append_metrics:
        episode_traces = _read_jsonl(metrics_dir / "episode_traces.jsonl")
        print(f"[append] loaded {len(episode_traces)} existing episode traces from {metrics_dir}")
    completed_pairs = {
        (int(ep["meta"]["task_id"]), int(ep["meta"]["episode_idx"]))
        for ep in episode_traces
    }
    total_episodes = len(episode_traces)
    total_successes = int(sum(bool(ep["meta"]["success"]) for ep in episode_traces))
    new_episodes = 0
    stop_eval = False

    for task_id in tqdm.tqdm(range(num_tasks), desc="Tasks"):
        if stop_eval:
            break
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = get_libero_env(task)

        if len(initial_states) == 0:
            raise ValueError(f"No initial states provided for task {task_id}.")

        task_successes = 0
        task_episodes = 0

        try:
            for episode_idx in tqdm.tqdm(range(num_trials_per_task), desc="Episodes", leave=False):
                if max_eval_episodes is not None and new_episodes >= max_eval_episodes:
                    stop_eval = True
                    break
                if selected_pairs and (task_id, episode_idx) not in selected_pairs:
                    continue
                if (task_id, episode_idx) in completed_pairs:
                    continue
                task_episodes += 1
                new_episodes += 1
                video_idx = len(episode_traces) + 1

                if new_episodes % eval_world_size != eval_rank:
                    continue

                env.reset()
                obs = env.set_init_state(initial_states[episode_idx])

                policy.reset(task_description)

                success, replay_images, trace = run_episode(
                    env,
                    policy,
                    task_description,
                    obs,
                    max_steps,
                    num_steps_wait,
                )

                if success:
                    task_successes += 1
                    total_successes += 1

                if save_rollout_videos:
                    save_rollout_video(
                        replay_images,
                        video_idx,
                        success,
                        task_description,
                        rollout_dir,
                    )
                episode_step_count = int(trace["action_count"])
                episode_trace = _trim_failure_episode_metrics(
                    {
                        "meta": {
                            "task_id": int(task_id),
                            "episode_idx": int(episode_idx),
                            "total_episode_idx": int(video_idx),
                            "task_description": task_description,
                            "success": bool(success),
                            "termination_reason": _termination_reason(success, trace["exception"], episode_step_count, max_steps),
                            "exception": trace["exception"] or "",
                            "step_count": episode_step_count,
                            "max_steps": int(max_steps),
                            "num_steps_wait": int(num_steps_wait),
                            "inference_step_count": int(len(trace["chunks"])),
                            "chunk_count": int(len(trace["chunks"])),
                            "action_count": int(trace["action_count"]),
                            "regen_threshold": float(regen_threshold),
                            "regen_max_attempts": int(regen_max_attempts),
                            "regen_strategy": str(regen_strategy),
                            "regen_num_candidates": int(regen_num_candidates),
                            "action_conf_threshold": float(action_conf_threshold),
                            "action_min_execute_actions": int(action_min_execute_actions),
                            "action_max_execute_actions": int(action_max_execute_actions),
                            "max_control_steps": int(max_steps),
                        },
                        "chunks": trace["chunks"],
                    }
                )
                episode_traces.append(episode_trace)
                completed_pairs.add((task_id, episode_idx))
                total_episodes = len(episode_traces)
                _write_trace_outputs(metrics_dir, episode_traces)

                success_rate = total_successes / max(total_episodes, 1)
                print(
                    f"Task {task_id} | Episode {episode_idx + 1} | Success: {success} "
                    f"| Total Success Rate: {success_rate:.3f}\n"
                )
        finally:
            env.close()

        task_success_rate = task_successes / max(task_episodes, 1)
        print(f"Task {task_id} success rate: {task_success_rate:.3f}")

    overall_success_rate = total_successes / max(total_episodes, 1)
    _write_trace_outputs(metrics_dir, episode_traces)
    print(
        f"Completed {total_episodes} episodes | "
        f"Total successes: {total_successes} | "
        f"Overall success rate: {overall_success_rate:.3f}\n"
    )
    print(f"Metrics written to: {metrics_dir}")


if __name__ == "__main__":
    tyro.cli(eval_vam_libero)
