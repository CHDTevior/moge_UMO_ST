# K-Encoder Stage-A 200K 结果审阅

## 当前状态

- Stage-A 已完整训练到 200K。
- Stage-B 已按要求停止；未保存正式 Stage-B checkpoint。
- `latest.pt` 仍对应 Stage-A 200K。
- `smoke_step_00200004.pt` 只用于验证八卡恢复，不用于本次 Stage-A 结果判断。

Stage-A checkpoint：

```text
/mnt/afs/mogeflow-control/outputs/hy273_text_fusion/
hy273_kencoder_stageA_ddp4x32_20260727_0832/model/step_00200000.pt
```

## 建议查看顺序

1. `01_fixed16_cfg2_gifs`
   - 16 条固定困难/复合文本，CFG=2.0。
   - 重点看 `sample_03/06/07/12/13/15.gif`。
2. `02_matched_text_cfg2_gifs`
   - 同 seed、同噪声、同长度，只改变文本。
3. `04_matched_text_cfg1p5_gifs`
   - 当前 paraphrase 一致性最好的 CFG 档位。
4. `03_matched_text_cfg1p0_gifs` 和 `05_matched_text_cfg3p0_gifs`
   - 用于判断 guidance 过弱或过强。
5. `08_stageA_150k_cfg2_gifs`
   - 与 200K 做同协议视觉对比。
6. `06_cfg2_timeline_sheets` 和 `07_cfg_timeline_sheets`
   - 每张图按时间抽取 12 帧，便于快速看动作顺序。

## Matched-text GIF 顺序

每个 seed 连续五条：

```text
sample_00/05/10: a person is break dancing.
sample_01/06/11: a person is breaking dance.
sample_02/07/12: a person breakdances.
sample_03/08/13: empty text
sample_04/09/14: a person is walking slowly.
```

三个 seed 分别为：

```text
3407, 12345, 20260725
```

## 已观察到的结果

### 文本语义

- 三种 breakdance 表达都生成高能量舞蹈。
- empty 和 slow-walk 主要生成普通行走，与 breakdance 有明显区别。
- `breakdances` 与 canonical 的同 seed 输出在 200K 比 150K 更接近。
- `breaking dance` 仍能路由到舞蹈，但跨 seed 离散度比 `breakdances` 大。
- 当前动作更像站立式 breakdance/高能量舞蹈，不能声称已稳定生成典型地板动作。

### 复合动作

固定 16 条中可以看到：

- 走路 -> 下蹲捡物 -> 起身恢复走路；
- 持续弯腰拾物；
- 下蹲快速行走 -> 站直；
- 行走、转向、返回；
- 抬腿上楼梯式动作。

复杂描述不是全部坍缩成同一个 walk prior，但部分细粒度手部语义仍不够明确。

## 关键数值

公共 CFG=2.0，单位为 matched-seed joint MPJPE：

| 指标 | 150K | 200K | 解释 |
|---|---:|---:|---|
| canonical vs empty | 0.922 m | 0.739 m | 文本改变输出的总幅度变小 |
| `breakdances` vs canonical | 0.114 m | 0.061 m | 该 OOD 同义表达改善 |
| `breaking dance` vs canonical | 0.191 m | 0.239 m | 该表达略退化 |
| `breakdances` routing advantage | 0.777 m | 0.695 m | 仍为正，但幅度下降 |
| `breaking dance` routing advantage | 0.712 m | 0.466 m | 仍为正，但幅度下降 |

200K fixed-t `t=0.1`：

```text
correct breakdance MSE: 0.2226
empty-text MSE:         0.2062
```

因此文本通路确定有效，但 correct text 在这个单一 GT/timestep 探针上仍没有优于
empty；不能把本轮结果解释成“文本目标对齐问题已经完全解决”。

固定 16 条物理质量：

| 指标 | 150K | 200K |
|---|---:|---:|
| mean FK jerk | 58.79 | 64.35 |
| median FK jerk | 37.85 | 46.47 |
| mean max foot skate velocity | 0.201 | 0.252 |
| median max foot skate velocity | 0.208 | 0.245 |
| median contact consistency | 0.981 | 0.975 |

200K 的语义组织仍在，但平滑性/滑脚相对 150K 有小幅退化，不能忽略。

## CFG 判断

- CFG=1.5：本轮 paraphrase 一致性整体最好。
- CFG=2.0：语义区分清楚，是公共比较档位。
- CFG=3.0：动作更激烈，但 paraphrase 一致性下降，出现过度 guidance 倾向。

当前建议先查看 CFG=1.5 与 CFG=2.0，不建议把 CFG=3.0 作为默认值。
