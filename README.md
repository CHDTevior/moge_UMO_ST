# moge_UMO_ST

HY273/Kimodo273 raw-space rectified-flow research code for a shared
Text-to-Motion, Kimodo-like Control, and Motion Editing model.

This repository is a research snapshot. It contains model, data-loader,
training, sampling, evaluation, configuration, test, and experiment-document
code. It intentionally does not contain datasets, text caches, model weights,
large generated arrays/media, external repositories, or the interactive demo.
Compact experiment summaries required for scientific review are kept under
`results/`.

## Current status

Status date: 2026-08-03.

| Capability | Implemented in code | Included in FullText-Reaction V1 |
|---|---:|---:|
| Text-to-Motion | yes | yes |
| Kimodo-like Control | yes | no, retained in the prior baseline |
| Motion Editing | yes | yes |
| Actor-conditioned Reaction | yes | yes |
| Ease-in/Ease-out conditioning | yes | no |
| Edit + Control composition | yes | no |

### FullText-Reaction V1

The latest completed experiment uses one shared model for T2M, Motion Editing,
and actor-conditioned Reaction:

```text
0 -> 100K:   T2M = 100%
100K -> 350K: T2M / Edit / Reaction = 30% / 35% / 35%
text:         LLM2Vec sentence token + variable-length contextual tokens
text fusion:  tokens participate in the main MMDiT stream
reaction:     observed actor is an independent source-token block
prediction:   clean x0, velocity-space base objective
training:     DDP8, bf16, 50K archived checkpoints
```

The default unified checkpoint is 300K. The 350K checkpoint is preferred when
Reaction or T2M FID is the only priority. Both remain local and are not stored
in Git:

```text
/mnt/afs/mogeflow-control/outputs/hy273_unified_reaction/
hy273_unified_fulltext_reaction_v1_20260801_0315/model/
step_00300000.pt
step_00350000.pt
```

Training is intentionally stopped at 350K under this recipe: Reaction has
mostly plateaued, while stronger Edit response is accompanied by increasing
jerk. See the final report linked below before extending this run.

### Prior K-Encoder Control baseline

The prior selected base model is **K-Encoder Stage-A 200K**:

```text
conditioning architecture: llm2vec_flux
text global conditioning:   llm2vec_tokens_only
text fusion:                f00 = shared projections + bidirectional attention
prediction:                 clean x0
base loss space:            velocity_mse
training:                   0 -> 200K, pure T2M, DDP4 x 32
global batch:               128
```

Stage-A is followed by:

```text
Stage-BE: 200K -> 250K, T2M / Control / Edit = 60% / 0% / 40%
Stage-BC: 250K -> 400K, T2M / Control / Edit = 10% / 70% / 20%
```

Stage-BE established Motion Editing while replaying T2M. Stage-BC added
Kimodo-like Control and physical 6D Ease conditioning while retaining 20% Edit
replay. Ease is independently present for 25% of T2M and 50% of Control
samples, and is forbidden on Edit samples.

Stage-BC completed at 400K. The checkpoint remains local and is not part of
this repository:

```text
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageBC_ease_t2m10_ctrl70_edit20_ddp8x16_20260728_201555/
model/step_00400000.pt
```

Final evidence status:

```text
Training health: PASS
Control:         established, with an endpoint-position trade-off
Motion Editing:  mixed/inconclusive from 250K to 400K
Ease:            condition path active, not yet reliable
```

The Stage-BC launcher is:

```bash
bash scripts/launch/train_hy273_kencoder_stage_bc_ease_control_ddp8.sh
```

It requires the local 250K Stage-BE checkpoint, unified manifests/stats,
LLM2Vec cache, and Ease stats. It uses DDP8 x 16, saves `latest.pt` every 10K,
archives every 50K, and stops at 400K.

## Key documents

- [FullText-Reaction 200K-350K final evaluation](docs/HY273_UNIFIED_FULLTEXT_REACTION_200K_350K_FINAL_REPORT_CN_20260803.md)
- [Unified 400K final evaluation](docs/HY273_KENCODER_UNIFIED_400K_FINAL_EVALUATION_CN_20260729.md)
- [Compact 400K result bundle](results/hy273_kencoder_unified_400k/README.md)
- [Final GPT-5.6 scientific review](docs/HY273_KENCODER_400K_GPT56_FINAL_REVIEW_CN.md)
- [Current T2M/Edit/Control architecture and training design](docs/CURRENT_T2M_EDIT_CONTROL_DESIGN_CN.md)
- [K-Encoder Stage-A 200K results](docs/KENCODER_STAGE_A_200K_RESULTS_CN.md)
- [Ease/Control implementation plan](docs/HY273_EASE_CONTROL_IMPLEMENTATION_PLAN_CN.md)
- [Stage-BC launch record](docs/HY273_KENCODER_STAGEBC_EASE_LAUNCH_CN_20260728.md)
- [GPT-5.6 scientific review](docs/HY273_EASE_GPT56_REVIEW.md)
- [Text-conditioning redesign](docs/HY273_TEXT_CONDITION_FUSION_REDESIGN_CN_20260726.md)
- [Previous Edit experiments and conclusions](docs/HY273_R13_EDIT_450K_PROGRESS_AND_NEXT_PLAN_CN.md)
- [HY273 versus Kimodo Control training patterns](docs/HY273_KIMODO_CONTROL_PATTERN_TRAINING_COMPARISON_CN.md)
- [Unified data-loader protocol](docs/reference/HY273_multitask_data_dataloader_protocol_CN.md)
- [Full multitask model design](docs/reference/HY273_kimodo_context_multitask_motion_editing_plan_CN.md)
- [Root/body Tensor Information Flow](docs/reference/HY273_redenoise_kimodo_like_tensor_information_flow.md)

## Repository map

```text
train_hy273_multitask.py       shared training entry
train_hy273_unified_actor.py   T2M/Edit/Reaction curriculum trainer
sample_hy273_multitask.py      T2M/Control/Edit ODE sampler and layered CFG
data/                          manifest dataset, paired transforms, task scheduler
models/raw_motion/             HY273 model, constraints, losses, metrics
models/codeflow/dit_blocks.py  Flux/MMDiT blocks and text-fusion modes
configs/                       versioned research configurations
scripts/launch/                launch and evaluation scripts
tools/                         cache, diagnostic, benchmark, and rendering tools
tests/                         focused model/data/training/sampling tests
docs/                          current experiment records
```

## Representation

Every motion is `[T,273]`, 30 FPS, Y-up:

```text
[0:3]     smooth root position
[3:5]     global root heading [cos(theta), sin(theta)]
[5:71]    local joint positions, 22 x 3
[71:203]  global joint rotations, 22 x 6D
[203:269] global joint velocities, 22 x 3
[269:273] four foot-contact channels
```

The current R13 protocol uses one normalized Gaussian rectified-flow state over
all 273 channels, including contact. The model predicts clean `x0`; the base
objective is evaluated in velocity space.

## Data dependencies

Default paths in the configurations point to the original research machine:

```text
HumanML3D K273:
/mnt/afs/mogo_base/datasets/HumanML3D/kimodo273_from_hy201_smplx22

MotionFix K273:
/mnt/afs/mogo_base/datasets/MotionFix/kimodo273_from_hy201_smplx22

Unified manifests/stats:
/mnt/afs/mogo_base/datasets/HY273_multitask_v1

LLM2Vec cache:
/mnt/afs/mogo_base/datasets/HY273_multitask_v1/
llm2vec_llama3_8b_profile_v1
```

The upstream K273 conversion and data-semantics repository is:

<https://github.com/CHDTevior/HY201_to_K273>

## Stage-A reference commands

The completed K-Encoder Stage-A was launched with:

```bash
TREATMENT=kencoder \
RUN_NAME=hy273_kencoder_stageA_ddp4x32_20260727_0832 \
bash scripts/launch/train_hy273_llm2vec_stage_a_ddp4.sh
```

Its text/visual diagnostic entry is:

```bash
bash scripts/launch/eval_hy273_llm2vec_kencoder_200k.sh
```

These commands require the local manifests, normalization statistics, LLM2Vec
cache, and CUDA environment. No large asset is downloaded automatically.
