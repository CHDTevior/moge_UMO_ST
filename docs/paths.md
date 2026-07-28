# Paths And Checkpoint Manifest

Date: 2026-07-02 CST

This file records the canonical paths for the `mogeflow-control` continuation workspace.

## Workspace

- Control workspace: `/mnt/afs/mogeflow-control`
- Original workspace: `/mnt/afs/mogeflow-umo`
- Python env: `/mnt/afs/conda_path/envs/codeflow/bin/python`

## Local Symlinks

These are intentionally symlinks to avoid duplicating large files.

| In `mogeflow-control` | Target |
|---|---|
| `dataset/HumanML3D_272` | `/mnt/afs/mogeflow-umo/dataset/HumanML3D_272` |
| `checkpoints/vq/humanml3d_272_pscf_nooverlap_w96_n1024_d128_ddp2_b32_ep200_seed3407_20260624_230655` | `/mnt/afs/mogeflow-umo/checkpoints/vq/humanml3d_272_pscf_nooverlap_w96_n1024_d128_ddp2_b32_ep200_seed3407_20260624_230655` |
| `checkpoints/t2m/codeflow_t2m_272_rvq1024_bestfid_4gpu_bz16_eval25_20260627_180755` | `/mnt/afs/mogeflow-umo/checkpoints/t2m/codeflow_t2m_272_rvq1024_bestfid_4gpu_bz16_eval25_20260627_180755` |
| `checkpoints/t2m/codeflow_t2m_umo4_272_from_t2m_bestfid_rvq1024_fid_4gpu_fixedeval_20260629_033829.optional_t2m_plus_umo4` | `/mnt/afs/mogeflow-umo/checkpoints/t2m/codeflow_t2m_umo4_272_from_t2m_bestfid_rvq1024_fid_4gpu_fixedeval_20260629_033829` |
| `checkpoints/evaluators/motionstreamer` | `/mnt/afs/mogeflow-umo/checkpoints/evaluators/motionstreamer` |
| `checkpoints/evaluators/distilbert-base-uncased` | `/mnt/afs/mogeflow-umo/checkpoints/evaluators/distilbert-base-uncased` |
| `checkpoints/clip/ViT-B-32.pt` | `/mnt/afs/MoGeFlow_WAM/checkpoints/clip/ViT-B-32.pt` |
| `glove` | `/mnt/afs/MoGeFlow_WAM/glove` |

## HumanML3D_272 Data

- Data root: `/mnt/afs/mogeflow-control/dataset/HumanML3D_272`
- Mean: `/mnt/afs/mogeflow-control/dataset/HumanML3D_272/Mean.npy`
- Std: `/mnt/afs/mogeflow-control/dataset/HumanML3D_272/Std.npy`
- Motion vectors: `/mnt/afs/mogeflow-control/dataset/HumanML3D_272/new_joint_vecs`
- Texts: `/mnt/afs/mogeflow-control/dataset/HumanML3D_272/texts`
- Splits: `train.txt`, `val.txt`, `test.txt` under the data root.

## RVQ1024

Run directory:

`/mnt/afs/mogeflow-control/checkpoints/vq/humanml3d_272_pscf_nooverlap_w96_n1024_d128_ddp2_b32_ep200_seed3407_20260624_230655`

Active tokenizer:

`/mnt/afs/mogeflow-control/checkpoints/vq/humanml3d_272_pscf_nooverlap_w96_n1024_d128_ddp2_b32_ep200_seed3407_20260624_230655/model/net_best_fid.tar`

Partition:

`/mnt/afs/mogeflow-control/configs/humanml3d_272_skeleton_partition_pscf_nooverlap.json`

Metrics:

| Selection | Epoch | Step | FID | Top3 | MPJPE cm |
|---|---:|---:|---:|---:|---:|
| Best FID | `13` | `4382` | `6.465086421708747` | `0.8931240657698056` | `5.27915029810193` |
| Best Top3 reference | `29` | `9390` | `20.141420128104187` | `0.8976083707025411` | `4.370401224249526` |

## Pure T2M

Run directory:

`/mnt/afs/mogeflow-control/checkpoints/t2m/codeflow_t2m_272_rvq1024_bestfid_4gpu_bz16_eval25_20260627_180755`

Recommended checkpoint:

`/mnt/afs/mogeflow-control/checkpoints/t2m/codeflow_t2m_272_rvq1024_bestfid_4gpu_bz16_eval25_20260627_180755/model/best_fid.pt`

Semantic checkpoint:

`/mnt/afs/mogeflow-control/checkpoints/t2m/codeflow_t2m_272_rvq1024_bestfid_4gpu_bz16_eval25_20260627_180755/model/best_top3.pt`

Metrics:

| Selection | Epoch | Step | FID | Top3 | Notes |
|---|---:|---:|---:|---:|---|
| `best_fid.pt` | `425` | `155550` | `5.954869671775896` | `0.9411182582879762` | Default pure T2M checkpoint |
| `best_top3.pt` | `200` | `73200` | `11.853734896550463` | `0.9450766947055913` | Use only when maximizing retrieval Top3 |

## Optional T2M+4 Reference

This symlink is present only to preserve history and allow comparisons. It is not the default for `mogeflow-control`.

`/mnt/afs/mogeflow-control/checkpoints/t2m/codeflow_t2m_umo4_272_from_t2m_bestfid_rvq1024_fid_4gpu_fixedeval_20260629_033829.optional_t2m_plus_umo4`

Metrics:

| Selection | Epoch | Step | FID | Top3 |
|---|---:|---:|---:|---:|
| `best_top3.pt` | `300` | `109800` | `5.843138406176422` | `0.9351806036615536` |
| `best_fid.pt` | `525` | `192150` | `5.320683719753561` | `0.9190994557149926` |

## Included Entry Scripts

Training:

- `scripts/launch/train_humanml3d_272_part_vq_ddp.sh`
- `scripts/launch/train_humanml3d_272_t2m_codeflow.sh`
- `train_codeflow_part_structured.py`
- `train_codeflow.py`
- `tools/train_humanml3d_272_part_vq_ddp.py`

Evaluation:

- `eval_codeflow_t2m_motionstreamer272.py`
- `eval_codeflow_t2m.py`
- `eval_codeflow_part_structured_t2m.py`
- `scripts/launch/eval_t2m272_cfg_sweep_worker.sh`

Inference / diagnostics:

- `gen_codeflow_t2m.py`
- `tools/check_codeflow_vq_contract.py`
- `tools/visualize_codeflow_t2m_rvq_decode_compare.py`

## Excluded From Primary Workflow

- MotionFix nb8192 tokenizer.
- MotionFix edit-only launcher.
- UMO/edit launchers and eval scripts as primary control scripts.
- Old visual outputs, run logs, and training caches.
- Large checkpoint/data copies; use the symlinks above.
