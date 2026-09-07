import os

import numpy as np
from simpler_env.policies.vam.video_action_model import VAMInference

from simpler_env.evaluation.argparse import get_args
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""

    model = VAMInference(
        args.vam_experiment_name,
        args.vam_video_model_path,
        args.vam_action_model_path,
        args.vam_dataset_statistics_path,
        args.vam_img_horizon,
        args.vam_lowdim_horizon,
        args.vam_stop_video_denoising_step,
        args.vam_num_execute_actions,
        is_hil=False,
        prompt_embeddings_path=args.vam_prompt_embeddings_path,
        consensus_medoid_only=args.vam_consensus_medoid_only,
        consensus_num_candidates=args.vam_consensus_num_candidates,
        consensus_rank_fusion=args.vam_consensus_rank_fusion,
        latent_medoid_strategy=args.vam_latent_medoid_strategy,
        diagnostics_mode=args.vam_diagnostics_mode,
        representation_layer_indices=args.vam_representation_layer_indices,
        decoder_capture_block_indices=args.vam_decoder_capture_block_indices,
    )
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
