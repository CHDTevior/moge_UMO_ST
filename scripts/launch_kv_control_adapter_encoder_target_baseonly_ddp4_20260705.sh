#!/usr/bin/env bash
set -euo pipefail

cd /mnt/afs/mogeflow-control

RUN=kv_control_adapter_encoder_target_baseonly_ddp4_20260705
OUT=checkpoints/t2m/${RUN}
RESUME_CKPT="${OUT}/model/latest.pt"
BASE_CKPT=checkpoints/t2m/codeflow_t2m_272_rvq1024_bestfid_4gpu_bz16_eval25_20260627_180755/model/best_fid.pt

mkdir -p "${OUT}" logs

CKPT_ARGS=()
if [[ -f "${RESUME_CKPT}" ]]; then
  CKPT_ARGS=(--resume "${RESUME_CKPT}")
else
  CKPT_ARGS=(--init_checkpoint "${BASE_CKPT}" --init_checkpoint_use_ema)
fi

exec env CUDA_VISIBLE_DEVICES=4,5,6,7 PYTHONUNBUFFERED=1 \
  /mnt/afs/conda_path/envs/codeflow/bin/python -m torch.distributed.run \
  --nproc_per_node 4 \
  --master_port 29653 \
  train_codeflow_part_structured.py \
  --name "${RUN}" \
  --output_dir "${OUT}" \
  --dataset_name t2m \
  --motion_dim 272 \
  --motion_fps 30 \
  --data_root dataset/HumanML3D_272 \
  --kv_root . \
  --vq_backend kv_part \
  --vq_checkpoint checkpoints/vq/humanml3d_272_pscf_nooverlap_w96_n1024_d128_ddp2_b32_ep200_seed3407_20260624_230655/model/net_best_fid.tar \
  --vq_partition configs/humanml3d_272_skeleton_partition_pscf_nooverlap.json \
  --mean_path dataset/HumanML3D_272/Mean.npy \
  --std_path dataset/HumanML3D_272/Std.npy \
  --clip_path /mnt/afs/MoGeFlow_WAM/checkpoints/clip/ViT-B-32.pt \
  --representation part_structured \
  --coupling_mode frame_grouped \
  --code_dim 128 \
  --num_parts 6 \
  --num_codes 1024 \
  --part_hidden_dim 192 \
  --hidden_size 1152 \
  --num_heads 12 \
  --depth_double 6 \
  --depth_single 12 \
  --mlp_ratio 4.0 \
  --dropout 0.05 \
  --motion_length 300 \
  --unit_length 4 \
  --batch_size 32 \
  --num_workers 8 \
  --max_epoch 4000 \
  --max_steps 1464000 \
  --lr 0.0001 \
  --lr_scheduler half_cosine \
  --eta_min_ratio 0.01 \
  --warmup_steps 2000 \
  --weight_decay 0.01 \
  --grad_clip 1.0 \
  --amp \
  --amp_dtype bf16 \
  --seed 3407 \
  --cond_drop_prob 0.1 \
  --disable_self_condition \
  --time_schedule uniform \
  --latent_norm_mode codebook \
  --terminal_mode tied_logits \
  --terminal_tau_mode codebook_nn \
  --terminal_loss_weight 0.0 \
  --clean_loss_weight 0.0 \
  --flow_loss_weight 0.1 \
  "${CKPT_ARGS[@]}" \
  --enable_kv_control \
  --kv_control_train_adapter_only \
  --kv_control_loss_weight 0.9 \
  --kv_control_loss_type l1 \
  --kv_control_min_keyframes 1 \
  --kv_control_max_keyframes 5 \
  --kv_control_min_joints 1 \
  --kv_control_max_joints 6 \
  --kv_control_dropout_prob 0.1 \
  --kv_control_clean_target encoder \
  --full_eval_every_epoch 0 \
  --simple_eval_every_epoch 0 \
  --inpaint_eval_every_epoch 0 \
  --global_edit_eval_every_epoch 0 \
  --best_checkpoint_limit 3 \
  --log_every 20
