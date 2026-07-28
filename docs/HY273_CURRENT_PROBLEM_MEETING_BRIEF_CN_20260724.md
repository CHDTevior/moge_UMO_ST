# HY273 统一运动模型当前问题与实验进展

更新时间：2026-07-24  
用途：教师会议简报

## 1. 一句话结论

当前统一模型的主要瓶颈不是 T2M 或 Kimodo-like Control，而是：

> Motion Editing 在全量训练上仍容易依赖 source motion、弱化 relative
> instruction；平均编辑误差已经改善，但速度、节奏、重复次数等动态编辑仍不稳定。

目前最合理的机制假设是：

```text
高带宽逐帧 source 条件
    + MotionFix 中 target 通常接近 source
    + 以逐帧重建为主的训练目标
    -> 复制 source 是容易而且低损失的解
    -> 文本只提供较弱的增量方向
    -> 提高 edit CFG 只会放大已经学错的静态方向
```

## 2. 当前哪些能力是正常的

### T2M

- 仍能生成连续动作。
- 400K 后采用 `T2M/Control/Edit = 30/40/30` 联合 replay，没有发现明显 T2M 遗忘。
- Positive 450K 和 Temporal 450K 的足滑与 jerk 代理指标反而优于 Parent 400K。

### Kimodo-like Control

- root/path、end-effector、full-pose 和 contact 控制的主要几何指标保持。
- 8,084-case Control benchmark 中，主要几何指标都在相对 Parent 5% 非退化范围内。
- with-text controlled-contact Brier 有校准退化，但 Accuracy/F1 仍约为 `0.996/0.994`，
  不是控制能力整体崩溃。

### 训练系统

- 两套数据均为 K273、30 FPS，并使用同一组联合 normalization stats。
- MotionFix source/target 使用配对 crop、配对 root-origin shift 和共享 yaw augmentation。
- 当前 DDP、loss、采样和 resume 没发现 NaN、OOM 或条件串线。

## 3. 主要问题的具体表现

### 3.1 Instruction sensitivity 不够稳定

同一个 source motion 下替换 correct、sibling 或 empty instruction 时，Parent
400K 的输出差异较弱：

| 模型 | ODE32 Correct MSE | Correct-vs-empty gap | Correct assignment |
|---|---:|---:|---:|
| Parent 400K | 1.0581 | -0.0277 | 62.1% |
| Positive 450K | 1.0197 | +0.1252 | 75.9% |
| Temporal 450K | 1.0046 | +0.1412 | 69.0% |

Parent 的 gap 为负，表示空 instruction 平均比正确 instruction 更接近 target。
Positive-only 训练使该关系在完整 ODE 采样中反转，说明文本通路可以被增强；
但 Temporal 版本的平均 MSE 更好、assignment 反而降低，说明低重建误差不等于
正确理解编辑指令。

### 3.2 动态编辑失败

预先指定的 stress case：

```text
move feet faster and punch once faster
```

| 动作 | Contact occupancy | Foot speed (m/s) | Strong-foot peaks |
|---|---:|---:|---:|
| Source | 0.210 | 0.585 | 7 |
| Target | 0.079 | 1.557 | 12 |
| Parent 400K | 0.402 | 0.679 | 6 |
| Positive 450K | 0.844 | 0.131 | 7 |
| Temporal 450K | 0.892 | 0.122 | 5 |

模型进入了近静态、高接触占用状态。把 `edit_cfg` 从 2.0 提高到 3.0 后脚速
进一步下降，说明 CFG 放大的不是正确的 “faster” 方向。

## 4. 怀疑原因及证据强度

### 原因 A：source-copy shortcut

状态：高置信度。

- source 是逐帧、273D、与 target 对齐的高带宽条件。
- 文本需要通过 token/pooled attention 提供低带宽语义增量。
- 原 additive 结构把 source context 直接加到 target motion token，复制路径很短。
- 给定 source 后，标准重建 loss 没有强迫模型证明“必须使用 instruction”。

### 原因 B：MotionFix 中 instruction 的条件可辨识性不足

状态：高置信度。

MotionFix train：

```text
pairs:                         5,387
unique sources:                4,055
有 same-source sibling 的 pair: 2,576 / 47.8%
强 sibling pair:               1,666 / 30.9%
```

也就是说，大量 source 在训练中只对应一条 instruction-target。对于这些样本，
网络可能通过 source 和数据先验预测近似 target，而不需要区分 relative instruction。

在 `Edit=30%` 的联合训练下，强 sibling 的 top-20% changed-region 监督只约占
全部训练坐标 exposure 的 `1.89%`。这是 supervision coverage 的实际天花板。

### 原因 C：当前 loss 与动态语义不完全对齐

状态：高置信度。

- clean-x0、joint velocity 和 FK loss 主要优化逐帧/局部重建。
- “更快”可能表示周期数、事件发生时间、局部肢体速度或 time warp。
- 单纯降低 position/velocity MSE 不保证学会这些指令级时序关系。
- R14 Temporal loss 改善了平均 MSE、jerk 和 foot skate，却没有修复
  `feet faster`，是直接证据。

### 原因 D：additive source 融合带来不合适的结构偏置

状态：中等置信度，R15 正在验证。

这不是“网络完全没有能力”。16 个 same-source 训练组的 tiny-overfit：

| Step | Continuous correct MSE | Assignment |
|---|---:|---:|
| 0 | 3.6935 | 56.25% |
| 100 | 0.1160 | 100% |
| 500 | 0.0135 | 100% |
| 2,000 | 0.0036 | 100% |

因此 text encoder、文本注入、source path 和共享 backbone 都能表达这类映射。
问题更像是全量数据下的 inductive bias，而不是硬断路。

### 原因 E：统一多任务中的 Edit exposure 不足

状态：可能的次要因素。

- Parent 260K–400K 只有约 2% Edit，400K 的 Edit 能力本来较弱。
- 400K 后改为 30% Edit，instruction assignment 明显改善。
- 但 50K 的 30% Edit 仍没有解决动态语义。

因此增加 Edit 比例有帮助，但不是充分条件。继续只堆训练量未必能解决机制问题。

### 当前不优先怀疑

1. **数据配对或随机 yaw 串线**：配对增强和 manifest 已核对。
2. **FPS/normalization 不一致**：两套数据均为 30 FPS，共用联合 stats。
3. **HYText 完全不可用**：tiny-overfit 已证明文本通路可学习。
4. **旧 contact sigmoid feedback bug**：R13 已改为统一 273D Gaussian flow，
   ODE 中没有旧的 sigmoid self-feedback；当前高 contact 是模型学出的错误状态，
   不是旧 sampler 数值反馈。
5. **单纯 CFG 太低**：提高 CFG 会进一步放大错误静态方向。

## 5. 已尝试的解决办法

| 实验 | 修改 | 结果 | 结论 |
|---|---|---|---|
| R13 unified contact flow | contact 并入统一 273D clean flow，移除 sigmoid feedback | 旧 contact 饱和反馈问题被结构性消除 | 已保留 |
| Decomposed Edit CFG | 训练 source+text/source-only/text-only/uncond 等分支 | 能分别调 source 和 edit 强度 | 有效，已保留 |
| Low-t oversampling | 增加低噪声 Edit 训练 | train 改善、held-out 退化 | 不是单纯 timestep coverage 问题 |
| Discrepancy clean-x0 | 强化 source-target discrepancy | matched ODE target MSE 退化约 9.9% | 已停止 |
| Same-source ranking | hinge/softplus correct-vs-sibling | 两个 5K 候选都未通过；fixed-t empty gap 仍为负 | negative/ranking 不是首选 |
| Positive-only 50K | `0.05 target-x0 + 0.02 hard-x0`，不使用 negative | assignment 62.1%→75.9%，gap 转正 | 当前最有效的 Edit 训练改进 |
| Changed-region 50K | same-source top-20% discrepancy 加权 | 有改善，但弱于 Positive-only | 原定义停止 |
| Temporal 50K | Edit-only velocity vector + speed magnitude | 平均 MSE/T2M 平滑度改善，动态语义未改善 | 不继续加同一 loss |
| Higher edit CFG | `2.0→3.0` | faster 样例更静态 | 排除“CFG 不够” |
| R15 token block | source 从逐帧 additive 改为独立 source-context token block | 进行中 | 检验 additive shortcut |

## 6. 当前 R15 实验

R15 是严格 matched A/B：

- 同一个 Parent 400K；
- A：原 additive source；
- B：target-aligned `[source context, SEP, target]` token block；
- 两边都保留 Positive-only loss、decomposed CFG、30/40/30 task mix；
- 同 SamplePlan、noise、timestep、LR、global batch；
- 4 卡 + 4 卡并行；
- 不使用 adapter，不改变 T2M/Control 的 source-free 路径。

405K 的初步 fixed-t 结果：

| 模型 | Correct MSE | Instruction margin | Assignment advantage |
|---|---:|---:|---:|
| Additive 405K | 0.8400 | 0.0624 | 0.3103 |
| Token block 405K | 0.8720 | 0.0168 | 0.3103 |

Token block 在 5K 时尚未胜出，MSE 约退化 3.8%，但仍在 5% guardrail 内。
因为它改变了 source 融合的学习方式，5K 不足以直接判死。

截至 2026-07-24，两组已经稳定完成 425K：

```text
Additive 425K last logged loss:    0.001463
Token-block 425K last logged loss: 0.001372
```

425K matched exact-`t=0` 中期评估已经完成。主要的
target-disjoint、asset-nonoverlap 子集包含 29 个 same-source group：

| 指标 | Additive 425K | Token block 425K | Token−Additive 95% CI |
|---|---:|---:|---:|
| Correct MSE | 0.8432 | 0.8598 | `[-0.0279, +0.0725]` |
| Instruction margin | 0.0535 | 0.0387 | `[-0.0765, +0.0330]` |
| Assignment advantage | 0.3103 | 0.5862 | `[-0.0690, +0.6207]` |
| Correct-vs-empty gap | -0.0231 | +0.1458 | `[+0.0774, +0.2820]` |

解读：

- Token block 的 target fidelity 只退化约 2%，仍在 5% 范围内。
- 预注册主指标 instruction margin 尚未胜出，CI 跨 0。
- Assignment point estimate 明显提高，但 CI 仍跨 0。
- Correct-vs-empty gap 显著转正，说明 token block 已开始使模型依赖
  instruction，而不是把有无 instruction 当成近似等价输入。

这是鼓励性的中间信号，但还不是结构优越性的最终证据。两组按原计划继续到
450K，再统一评估 Edit、T2M、Control 和可视化。

## 7. 明天会议建议讨论的核心决策

### 决策 1：如果 R15 到 450K 仍不优于 additive，下一步走数据还是结构

当前建议优先走数据侧：

- 为同一个 source 程序化构造 opposite/different instruction siblings；
- speed 使用正确 time warp 后重新提取 K273；
- reach/direction 使用 IK 或可控变换；
- 明确构造 `same source + different instruction -> different target`。

这样直接提高“文本在给定 source 后仍然必要”的训练覆盖率。

### 决策 2：是否建立正式 Dynamic Edit benchmark

建议人工清洗：

- faster/slower；
- repetition count；
- event onset/timing；
- direction/amplitude；
- 明确受影响肢体。

评估应使用速度倍率、周期数、事件 onset、target retrieval 和 source preservation，
不能只看全身 MSE。

### 决策 3：统一模型的训练路线

目前 `30% T2M + 40% Control + 30% Edit` 已证明可以避免明显能力遗忘，建议保留
统一 backbone 和联合 replay，不转向 adapter。下一步应解决 Edit 的条件可辨识性
和 source-text fusion，而不是拆成三个独立模型。

## 8. 当前最保守的模型选择

- 原始几何 Control 基线：Parent 400K。
- 当前 Edit instruction assignment 最好：Positive 450K。
- 当前 T2M 平滑度代理最好：Temporal 450K。
- R15 尚未得出最终结论。

当前没有一个 checkpoint 可以严谨地称为所有子任务上的统一最优模型。
