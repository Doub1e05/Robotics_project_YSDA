# A100 paired baseline then consensus-only: LIBERO-PLUS Object

Use this protocol to measure consensus-only against baseline. The tmux runner executes baseline first and consensus-only second with the same checkpoint, seed, task order, rollout limit, and control horizon. It uses summary-only mode: one summary.json per arm, no videos or action/chunk traces.

Hardware requirement

Use an Ampere-or-newer GPU: A100, A4000, RTX 30xx/40xx, compute capability 8.0 or higher. T4 and V100 use an exact but very slow math-SDPA fallback and are suitable only for smoke tests.

1. Clone and verify hardware

git clone git@github.com:Doub1e05/Robotics_project_YSDA.git
cd Robotics_project_YSDA
git pull --ff-only origin main
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader

The compute capability must be 8.0 or higher. Reserve about 15 GB for Object checkpoints and at least 20 GB for LIBERO-PLUS assets and runtime files.

2. Install project environment

On a fresh host run:

bash scripts/migration/bootstrap_host.sh
# Open a new shell if bootstrap installed uv or conda.
bash scripts/migration/setup_python_environments.sh

Verify the exact Python interpreter used by eval:

model/.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available()
major, minor = torch.cuda.get_device_capability()
assert major >= 8, f'Need Ampere+; found SM{major}{minor}'
print(torch.cuda.get_device_name(0), f'SM{major}{minor}', torch.__version__)
PY

Stop and fix the driver/PyTorch stack if this fails.

3. Download official Object checkpoints

Weights are not in Git. Authenticate to Hugging Face if necessary, then run:

hf auth login
model/.venv/bin/python model/scripts/download_checkpoints.py --models libero_object_one --checkpoint-dir model/checkpoints

Verify these non-empty files:

model/checkpoints/video_backbone/v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000008260_fused.pt
model/checkpoints/action_decoder/w2a_libero_object_one_v2w_libero_object_agentview_lora_rank256_lr1.778e-04_bsz32_iter_000029997.pt
model/checkpoints/dataset_statistics/libero_object_one.json
model/checkpoints/libero_t5_embeddings.pkl

4. Restore LIBERO-PLUS and non-Git assets

LIBERO-plus is external upstream code and assets. Clone it when absent, then download assets:

if [ ! -d LIBERO-plus/.git ]; then
  git clone https://github.com/sylvestf/LIBERO-plus.git LIBERO-plus
fi
bash scripts/migration/download_libero_plus_assets.sh

Verify that LIBERO-plus/libero/libero/assets has scenes, new_objects, and textures. Do not install another conflicting libero package into model/.venv: the runner provides LIBERO-PLUS with PYTHONPATH.

5. Mandatory one-episode tmux preflight

SESSION=libero_object_pair_preflight OUT_ROOT="$PWD/eval_outputs/libero_object_plus/paired_preflight_$(date +%Y%m%d_%H%M%S)" MAX_EVAL_EPISODES=1 NUM_TRIALS_PER_TASK=1 MAX_CONTROL_STEPS=220 bash scripts/eval/launch_mimic_libero_plus_object_paired_tmux.sh
tmux attach -t libero_object_pair_preflight

After it exits, check:

cat "$OUT_ROOT/baseline/metrics/summary.json"
cat "$OUT_ROOT/consensus_only/metrics/summary.json"

Zero success in one episode is not automatically a setup failure. Exceptions, OOMs, or missing summaries are failures.

6. Full sequential paired run

This standard comparison is 100 episodes: ten initial states for the first ten fixed LIBERO-PLUS Object variants. Baseline runs fully before consensus-only. If baseline exits non-zero, set -e prevents an invalid unpaired consensus run.

cd ~/Robotics_project_YSDA
SESSION=libero_object_pair_full_100ep OUT_ROOT="$PWD/eval_outputs/libero_object_plus/paired_full_100ep_seed0_$(date +%Y%m%d_%H%M%S)" SEED=0 MAX_EVAL_EPISODES=100 NUM_TRIALS_PER_TASK=10 MAX_CONTROL_STEPS=220 bash scripts/eval/launch_mimic_libero_plus_object_paired_tmux.sh
tmux attach -t libero_object_pair_full_100ep

Monitor detached:

tmux capture-pane -pt libero_object_pair_full_100ep:0 -S -60
nvidia-smi

Final result:

cat "$OUT_ROOT/comparison_summary.json"

It contains baseline and consensus-only success counts/rates plus deltas. Logs are baseline/run.log and consensus_only/run.log.

Scope and reporting

Do not compare runs with different seed, checkpoint, task selection, control horizon, or trials per task. Consensus candidates are generated sequentially: K=3 does not mean three resident model copies.

LIBERO-PLUS libero_object has 2,518 OOD perturbation variants. The 100-episode protocol is a documented fixed slice, not all of LIBERO-PLUS Object. A full robustness study needs a documented perturbation-balanced subset. The Object policy is trained for ordinary LIBERO Object, so report this as OOD LIBERO-PLUS robustness evaluation, not an in-distribution score.
