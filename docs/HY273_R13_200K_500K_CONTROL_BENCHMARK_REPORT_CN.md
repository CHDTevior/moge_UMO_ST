# HY273 R13 200K-500K 分阶段控制 Benchmark 报告

## 1. 结论摘要

本报告比较同一次 R13 unified-273 contact-flow 训练在 200K、250K、400K、450K、500K 的 Kimodo-like 控制能力。

- **400K 是本次训练的纯控制最优 checkpoint。** 250K 已学会大部分控制，250K 到 400K 又显著改善末端、全姿态、接触一致性和足滑。
- 400K 后加入高比例 MotionFix editing，控制能力没有灾难性遗忘，但出现了明确的**带文本控制回退**：500K 的末端位置误差增加 0.408 cm，全姿态误差增加 0.353 cm，contact accuracy 下降 0.427 个百分点。
- 对最重要的足滑指标，带文本时 400K 到 500K 的 contact-foot velocity 从 0.1000 增至 0.1122 m/s，per-motion max foot velocity 从 1.0530 增至 1.3087 m/s，foot-skate ratio 从 26.780% 增至 27.264%。配对 bootstrap 的 95% CI 均不跨 0，是真实回退，不是汇总噪声。
- 不带文本时，500K 的 root/endpoint 空间控制仍略优于 400K，但 contact consistency 和足部速度同样变差。这说明主要问题不是 overwrite 失效，而是 editing 联训改变了 contact/foot dynamics；带文本分支受到的影响更强。
- 当前建议：**纯 Kimodo-like control 使用 400K；统一 T2M/control/edit 模型暂保留 500K，同时把 450K 作为能力折中候选。** 最终统一 checkpoint 还需要结合固定 T2M/edit 可视化判断，不能只根据控制 benchmark 决定。

## 2. 评测协议

```text
Run: hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507
Split: HumanML3D test, 4,042 motions
Control subtypes: 19
Text regimes: withtext + notext
Cases/checkpoint: 8,084
Weights: EMA
Sampler: ODE32
Text CFG: 2.0
Observation CFG: 2.0
Seed: 3407
Primary output: generated_raw (raw_pre_exact_clamp)
```

五个 checkpoint 使用相同 test motion、control compiler、case assignment 和初始噪声，因此阶段差异可做逐 case 配对比较。本报告不包含 FID、R-precision 或 Top-3。

## 3. 阶段定义

| Checkpoint | 刚完成的阶段 | T2M / Control / Edit |
|---|---|---:|
| 200K | Stage A: T2M pretrain | 100 / 0 / 0 |
| 250K | Stage B1: control bootstrap | 10 / 90 / 0 |
| 400K | Stage B2: joint adaptation | 18 / 80 / 2 |
| 450K | Stage C1: unified edit v2 | 30 / 40 / 30 |
| 500K | Stage C2: unified edit40 | 30 / 30 / 40 |

## 4. 总体原始指标

下面的 `all` 行按 evaluator 规则只在具有对应 metric 的 case 上做 per-motion mean。距离单位是 cm，旋转是 degree，足部速度是 m/s。除 accuracy/consistency/F1 外均越低越好。

### 4.1 With Text

| Step | Mix | Root cm | Root@10cm % | EE cm | EE rot | Full cm | Contact acc % | Contact F1 % | Contact BCE | FK gap cm | Contact consistency % | Contact skate m/s | Max skate m/s | Skate ratio % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200K | 100/0/0 | 49.930 | 41.356 | 59.710 | 24.324 | 50.438 | 99.399 | 99.250 | 0.08308 | 0.315 | 84.837 | 0.1569 | 1.3173 | 33.700 |
| 250K | 10/90/0 | 5.647 | 91.062 | 12.534 | 6.636 | 4.932 | 99.904 | 99.797 | 0.01332 | 0.234 | 89.201 | 0.1093 | 1.1100 | 28.539 |
| 400K | 18/80/2 | **4.440** | **96.366** | **5.532** | **5.016** | **2.995** | **99.835** | **99.618** | **0.02279** | 0.202 | **90.452** | **0.1000** | **1.0530** | **26.780** |
| 450K | 30/40/30 | 4.400 | 96.755 | 5.700 | 4.991 | 3.085 | 99.573 | 99.269 | 0.05892 | 0.175 | 89.969 | 0.1081 | 1.2100 | 26.840 |
| 500K | 30/30/40 | 4.247 | 95.568 | 5.940 | 5.122 | 3.348 | 99.408 | 99.202 | 0.08174 | **0.170** | 89.497 | 0.1122 | 1.3087 | 27.264 |

粗体用于突出 400K 与 500K 的主要权衡，不表示所有指标都具有同一优先级。

### 4.2 Without Text

| Step | Mix | Root cm | Root@10cm % | EE cm | EE rot | Full cm | Contact acc % | Contact F1 % | Contact BCE | FK gap cm | Contact consistency % | Contact skate m/s | Max skate m/s | Skate ratio % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200K | 100/0/0 | 73.067 | 31.629 | 76.444 | 26.736 | 68.321 | 99.874 | 99.878 | 0.01735 | 0.304 | 84.730 | 0.1709 | 1.3891 | 34.967 |
| 250K | 10/90/0 | 4.766 | 96.094 | 11.034 | 5.602 | 4.158 | 99.977 | 99.975 | 0.00313 | 0.190 | 91.520 | 0.0934 | 0.8793 | 26.217 |
| 400K | 18/80/2 | 4.228 | 97.628 | 4.083 | 4.016 | 2.263 | 99.973 | 99.941 | 0.00369 | 0.165 | 92.742 | 0.0816 | 0.7687 | 25.832 |
| 450K | 30/40/30 | 3.885 | 98.999 | **3.736** | 4.061 | **2.096** | 99.952 | 99.870 | 0.00664 | 0.172 | 92.516 | 0.0858 | 0.8617 | 26.241 |
| 500K | 30/30/40 | **3.775** | 98.630 | 3.854 | 4.120 | 2.219 | 99.950 | 99.930 | 0.00695 | 0.179 | 92.272 | 0.0885 | 0.9036 | **25.481** |

## 5. 400K 到 500K 配对统计

使用相同 case 和相同初始噪声做 paired bootstrap，5,000 次重采样。`diff = 500K - 400K`。

### 5.1 With Text

| Metric | N | 400K -> 500K | Difference | 95% CI | 判断 |
|---|---:|---:|---:|---:|---|
| Root error | 2,340 | 4.440 -> 4.247 cm | -0.192 cm | [-0.837, +0.183] | 无明确变化 |
| Endpoint position | 1,702 | 5.532 -> 5.940 cm | +0.408 cm | [+0.264, +0.548] | 回退 |
| Endpoint rotation | 1,702 | 5.016 -> 5.122 deg | +0.107 deg | [-0.162, +0.315] | 无明确变化 |
| Full-pose keyframe | 1,489 | 2.995 -> 3.348 cm | +0.353 cm | [+0.247, +0.455] | 回退 |
| Contact accuracy | 1,273 | 99.835 -> 99.408% | -0.427 pp | [-0.513, -0.349] | 回退 |
| Contact BCE | 1,273 | 0.02279 -> 0.08174 | +0.05895 | [+0.04833, +0.07006] | 明显回退 |
| FK representation gap | 1,273 | 0.202 -> 0.170 cm | -0.031 cm | [-0.034, -0.029] | 改善 |
| Contact consistency | 4,042 | 90.452 -> 89.497% | -0.955 pp | [-1.142, -0.769] | 回退 |
| Contact-foot velocity | 4,042 | 0.1000 -> 0.1122 m/s | +0.0121 | [+0.0083, +0.0166] | 回退 |
| Per-motion max foot velocity | 4,042 | 1.0530 -> 1.3087 m/s | +0.2556 | [+0.2250, +0.2865] | 明显回退 |
| Foot-skate ratio | 4,042 | 26.780 -> 27.264% | +0.484 pp | [+0.231, +0.721] | 回退 |

### 5.2 Without Text

| Metric | N | 400K -> 500K | Difference | 95% CI | 判断 |
|---|---:|---:|---:|---:|---|
| Root error | 2,340 | 4.228 -> 3.775 cm | -0.453 cm | [-1.066, -0.114] | 改善 |
| Endpoint position | 1,702 | 4.083 -> 3.854 cm | -0.229 cm | [-0.386, -0.041] | 改善 |
| Endpoint rotation | 1,702 | 4.016 -> 4.120 deg | +0.104 deg | [+0.044, +0.164] | 小幅回退 |
| Contact accuracy | 1,273 | 99.973 -> 99.950% | -0.024 pp | [-0.047, -0.004] | 很小但可测 |
| Contact consistency | 4,042 | 92.742 -> 92.272% | -0.469 pp | [-0.589, -0.352] | 回退 |
| Contact-foot velocity | 4,042 | 0.0816 -> 0.0885 m/s | +0.0070 | [+0.0054, +0.0086] | 回退 |
| Per-motion max foot velocity | 4,042 | 0.7687 -> 0.9036 m/s | +0.1349 | [+0.1152, +0.1550] | 回退 |
| Foot-skate ratio | 4,042 | 25.832 -> 25.481% | -0.351 pp | [-0.752, +0.058] | 无明确变化 |

## 6. 500K With-Text 分控制类型结果

| Subtype | N | Root cm | Root@10cm % | EE cm | Rot deg | Full cm | Contact acc % | Contact BCE | Skate ratio % | Contact skate m/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| path_2dpos | 213 | 3.927 | 95.993 |  |  |  |  |  | 26.891 | 0.0787 |
| path_2dposrot | 213 | 3.920 | 97.317 |  |  |  |  |  | 24.711 | 0.0799 |
| waypoint_2dpos | 213 | 5.253 | 91.550 |  |  |  |  |  | 27.292 | 0.0844 |
| waypoint_2dposrot | 213 | 4.826 | 92.314 |  |  |  |  |  | 23.736 | 0.0724 |
| inbetweening | 213 |  |  |  |  | 3.344 |  |  | 19.692 | 0.0739 |
| random | 213 |  |  |  |  | 3.632 |  |  | 27.524 | 0.1180 |
| feet_posrot | 213 |  |  | 5.835 | 8.681 |  |  |  | 30.630 | 0.1514 |
| hands_posrot | 213 |  |  | 8.247 | 3.709 |  |  |  | 23.823 | 0.0823 |
| hands_feet_posrot | 213 |  |  | 6.144 | 5.795 |  |  |  | 29.675 | 0.1384 |
| root_ee_hands_feet_posrot_fullbody | 213 | 3.968 | 97.054 | 4.738 | 5.094 | 3.215 |  |  | 36.224 | 0.1786 |
| root_ee_hands_posrot | 213 | 4.153 | 96.171 | 7.213 | 3.665 |  |  |  | 25.590 | 0.0887 |
| root_ee_hands_posrot_fullbody | 213 | 4.130 | 96.462 | 5.125 | 3.366 | 3.081 |  |  | 31.975 | 0.1467 |
| root_path_fullbody | 213 | 4.107 | 95.378 |  |  | 3.747 |  |  | 31.584 | 0.1400 |
| contact_only_sparse | 213 |  |  |  |  |  | 99.236 | 0.10553 | 20.196 | 0.0686 |
| root_sparse_contact | 212 | 4.821 | 93.894 |  |  |  | 99.040 | 0.13266 | 24.958 | 0.0980 |
| root_dense_contact | 212 | 3.856 | 97.652 |  |  |  | 99.421 | 0.08002 | 22.032 | 0.1178 |
| endpoints_contact | 212 |  |  | 5.792 | 5.612 |  | 99.536 | 0.06409 | 28.908 | 0.1342 |
| fullpose_contact | 212 |  |  |  |  | 3.367 | 99.454 | 0.07539 | 30.978 | 0.1304 |
| mixed_contact | 212 | 3.759 | 97.474 | 4.417 | 5.058 | 3.051 | 99.764 | 0.03261 | 31.610 | 0.1491 |

## 7. Foot-Skate 专项判断

同一 test set 的 GT 基线为：contact-foot velocity 0.0416 m/s、per-motion max foot velocity 0.1325 m/s、foot-skate ratio 21.740%。

- 400K with-text 已分别达到 0.1000 m/s、1.0530 m/s、26.780%，说明控制模型仍有可见的足部滑动余量。
- 500K with-text 进一步退化到 0.1122 m/s、1.3087 m/s、27.264%。尤其 max velocity 增加 24.3%，表示少数帧的严重滑动/跳变增加，这比平均 skate ratio 的 0.484 pp 更值得重视。
- `feet_posrot`、`root + EE + fullbody`、`mixed_contact` 是 500K 最困难的几类；它们的 skate ratio 约为 30.6%-36.2%。

因此，按照“足滑最不能接受”的优先级，不能把 500K 描述为控制能力全面优于 400K。

## 8. Overwrite 与误差解释

主表报告 `generated_raw`，即模型完成逐 ODE step overwrite 后的原始输出。额外的 `diagnostic_exact_clamp` 会在物理 K273 空间再次覆盖受控通道：

- contact 与受控 rotation channel 可以达到精确相等；
- root metric 仍有约 3.2 cm 的表示下限，因为 metric 使用 rotation-FK root，而 observation 是 smooth-root position；
- full-pose/endpoint position 也不会自动变为 0，因为这些控制覆盖 position channel，而 benchmark 用 global-rotation FK 关节测量；
- 强行 final clamp 会增大 position/rotation 不一致，并可能恶化足滑，所以它只作为诊断，不作为主结果。

这说明当前非零控制误差不是简单的“没有 overwrite”，而主要来自模型输出在冗余 K273 position/rotation/contact 表示之间的一致性，以及受控帧周围自然过渡的质量。

## 9. 下一步实验含义

1. 400K 到 500K 的问题集中在 text-conditioned control、contact calibration 和 foot dynamics，不需要推翻 unified-273 contact-flow 结构。
2. 下一轮应该以 400K 控制能力为 retention target，联合训练 editing 时提高 control/contact replay 的有效约束，尤其是带文本 control batch，而不是只看 binary contact accuracy。
3. loss/采样评估应重点监控 contact BCE/Brier、contact-foot velocity、per-motion max velocity 和 skate ratio；仅看 contact accuracy 会掩盖概率饱和度与足部动力学回退。
4. 在决定 450K 或 500K 作为统一模型前，还需用相同 source/instruction/noise 做 400K/450K/500K 的 pure-edit 与 T2M 固定样本可视化。

## 10. 结果位置

```text
outputs/hy273_multitask/
  hy273_r13_contactflow_controlled_staged_ddp8_20260720_040507/
    evaluation/stage_benchmarks_ode32_cfg2_obs2_seed3407/
      step_00200000/summary.json
      step_00250000/summary.json
      step_00400000/summary.json
      step_00450000/summary.json
      step_00500000/summary.json
```

