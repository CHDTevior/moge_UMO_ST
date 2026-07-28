#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/mnt/afs/conda_path/envs/codeflow/bin/python}"
NUM_GPUS="${NUM_GPUS:-2}"
MASTER_PORT="${MASTER_PORT:-29531}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/dataset/HumanML3D_272}"
VQ_RUN_DIR="${VQ_RUN_DIR:-${REPO_ROOT}/checkpoints/vq/humanml3d_272_pscf_nooverlap_w96_n1024_d128_ddp2_b32_ep200_seed3407_20260624_230655/model}"
VQ_TAG="${VQ_TAG:-fid}"
if [[ "${VQ_TAG}" == "top3" ]]; then
  DEFAULT_VQ_CHECKPOINT="${VQ_RUN_DIR}/net_best_top3.tar"
elif [[ "${VQ_TAG}" == "fid" ]]; then
  DEFAULT_VQ_CHECKPOINT="${VQ_RUN_DIR}/net_best_fid.tar"
else
  DEFAULT_VQ_CHECKPOINT="${VQ_RUN_DIR}/${VQ_TAG}"
fi

VQ_CHECKPOINT="${VQ_CHECKPOINT:-${DEFAULT_VQ_CHECKPOINT}}"
VQ_PARTITION="${VQ_PARTITION:-${REPO_ROOT}/configs/humanml3d_272_skeleton_partition_pscf_nooverlap.json}"
NUM_CODES="${NUM_CODES:-1024}"
CLIP_PATH="${CLIP_PATH:-${REPO_ROOT}/checkpoints/clip/ViT-B-32.pt}"
HYMOTION_ROOT="${HYMOTION_ROOT:-/mnt/afs/HY-Motion-1.0}"
EVALUATOR_CKPT="${EVALUATOR_CKPT:-${REPO_ROOT}/checkpoints/evaluators/motionstreamer/Evaluator_272/epoch=99_state_dict.pt}"
DISTILBERT_PATH="${DISTILBERT_PATH:-${REPO_ROOT}/checkpoints/evaluators/distilbert-base-uncased}"

RUN_NAME="${RUN_NAME:-codeflow_t2m_272_rvq1024_${VQ_TAG}_2gpu_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/checkpoints/t2m/${RUN_NAME}}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node "${NUM_GPUS}" \
  --master_port "${MASTER_PORT}" \
  train_codeflow_part_structured.py \
  --name "${RUN_NAME}" \
  --output_dir "${OUT_DIR}" \
  --dataset_name t2m \
  --motion_dim 272 \
  --motion_fps 30 \
  --data_root "${DATA_ROOT}" \
  --kv_root "${REPO_ROOT}" \
  --vq_backend kv_part \
  --vq_checkpoint "${VQ_CHECKPOINT}" \
  --vq_partition "${VQ_PARTITION}" \
  --mean_path "${DATA_ROOT}/Mean.npy" \
  --std_path "${DATA_ROOT}/Std.npy" \
  --clip_path "${CLIP_PATH}" \
  --representation part_structured \
  --coupling_mode frame_grouped \
  --code_dim 128 \
  --num_parts 6 \
  --num_codes "${NUM_CODES}" \
  --part_hidden_dim "${PART_HIDDEN_DIM:-192}" \
  --hidden_size "${HIDDEN_SIZE:-1152}" \
  --num_heads "${NUM_HEADS:-12}" \
  --depth_double "${DEPTH_DOUBLE:-6}" \
  --depth_single "${DEPTH_SINGLE:-12}" \
  --mlp_ratio 4.0 \
  --dropout 0.05 \
  --motion_length 300 \
  --unit_length 4 \
  --batch_size "${BATCH_SIZE:-16}" \
  --num_workers "${NUM_WORKERS:-4}" \
  --max_epoch "${MAX_EPOCH:-600}" \
  --lr "${LR:-0.0001}" \
  --lr_scheduler half_cosine \
  --eta_min_ratio 0.01 \
  --warmup_steps 2000 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --amp \
  --amp_dtype bf16 \
  --seed "${SEED:-42}" \
  --cond_drop_prob 0.1 \
  --disable_self_condition \
  --time_schedule uniform \
  --latent_norm_mode codebook \
  --terminal_mode tied_logits \
  --terminal_tau_mode codebook_nn \
  --terminal_loss_weight 0.0 \
  --clean_loss_weight 0.0 \
  --best_checkpoint_limit 3 \
  --full_eval_backend motionstreamer272 \
  --full_eval_every_epoch "${FULL_EVAL_EVERY_EPOCH:-25}" \
  --full_eval_start_epoch "${FULL_EVAL_START_EPOCH:-25}" \
  --full_eval_batch_size 32 \
  --full_eval_num_workers "${FULL_EVAL_NUM_WORKERS:-4}" \
  --full_eval_metric_set fid_top3 \
  --full_eval_steps "${FULL_EVAL_STEPS:-96}" \
  --full_eval_cond_scale "${FULL_EVAL_COND_SCALE:-6.0}" \
  --full_eval_repeat_times "${FULL_EVAL_REPEAT_TIMES:-1}" \
  --full_eval_seed "${FULL_EVAL_SEED:-42}" \
  --full_eval_max_samples "${FULL_EVAL_MAX_SAMPLES:-0}" \
  --hymotion_root "${HYMOTION_ROOT}" \
  --motionstreamer_evaluator_checkpoint "${EVALUATOR_CKPT}" \
  --motionstreamer_distilbert_path "${DISTILBERT_PATH}" \
  "$@"
