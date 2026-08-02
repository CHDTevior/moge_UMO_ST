# HY273 K-Encoder 统一模型 400K 结果包

日期：2026-07-29

本目录包含 Stage-A、Stage-BE、Stage-C 完整训练后用于科研审核的小型结果文件。模型权重、逐 case `.npy`、GIF、原始训练 `metrics.jsonl` 和数据集未上传。

最终 checkpoint 仅保存在训练机：

```text
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageBC_ease_t2m10_ctrl70_edit20_ddp8x16_20260728_201555/
model/step_00400000.pt
```

## 训练阶段

| 阶段 | Step | T2M / Control / Edit |
|---|---:|---:|
| Stage A | 0K -> 200K | 100 / 0 / 0 |
| Stage BE | 200K -> 250K | 60 / 0 / 40 |
| Stage C | 250K -> 400K | 10 / 70 / 20 |

Stage C 中 Ease 覆盖率为 T2M 的 25%、Control 的 50%、Edit 的 0%，总体期望覆盖率为 37.5%。

## 文件索引

### Control

```text
control/protocol_manifest.json
control/full_test_summary.json
control/paired_bootstrap_10k_seed20260729.json
```

`full_test_summary.json` 对应 HumanML3D test 的 4,042 motions、with-text/notext 两种 regime、共 8,084 cases 和 19 种 control subtype。主协议为 EMA、ODE32、text CFG 2.0、control CFG 2.0、sampling seed 3407。

`paired_bootstrap_10k_seed20260729.json` 比较本模型 400K 与旧 R13 control-best 400K。统计单位为 `case_key`，采用 10,000 次双侧 paired percentile bootstrap，区间只覆盖 case variation，不覆盖 training seed、sampling seed 或 checkpoint selection。

对应复算脚本：

```text
tools/summarize_hy273_control_paired_bootstrap.py
```

### Ease

```text
ease/no_control_000708_seed3407.json
ease/no_control_003800_seed3407.json
ease/no_control_005818_seed3407.json
ease/hands_feet_posrot_003800_seed3407.json
```

四组 sweep 固定 checkpoint、文本、初始噪声和 control，仅改变一个 half 的 Ease scale：`[0, 0.25, 0.5, 0.75, 1.0]`。

### Motion Editing

```text
edit/parent250k_vs_final400k_selection.json
edit/parent250k_vs_final400k_summary.json
edit/cfg_sweep_000038_summary.json
edit/cfg_sweep_requested_summary.json
```

其中 parent/final 对照覆盖 Stage-BE 250K 与 Stage-C 400K；CFG sweep 只用于有限样例的推理敏感性检查，不代表全 test distribution 的最优 CFG。

### T2M

```text
t2m/paraphrase_ema_comparison.json
t2m/paraphrase_raw_comparison.json
t2m/visual16_quality.json
```

这些是固定 prompt panel 的探索性 probe，不是完整 T2M benchmark，不能扩展为 FID、R-precision 或总体语义质量结论。

### Run

```text
run/config_resolved.json
```

该文件是 Stage-C 实际 resolved config。完整 launch CLI 和 runtime override 见最终报告。

## 结论边界

```text
Control:
  已建立可用能力；
  with-text full-pose、rotation、contact 与足部动力学优于旧 R13；
  endpoint position 从 5.53 cm 回退到 6.30 cm。

Motion Editing:
  250K 与 400K 各有优劣，当前证据不足以排序。

Ease:
  条件路径已激活；
  响应弱、样本依赖，hard-control 组合尚未建立；
  不能宣称淡入淡出能力训练成功。
```

完整分析见：

```text
docs/HY273_KENCODER_UNIFIED_400K_FINAL_EVALUATION_CN_20260729.md
docs/HY273_KENCODER_400K_GPT56_FINAL_REVIEW_CN.md
```
