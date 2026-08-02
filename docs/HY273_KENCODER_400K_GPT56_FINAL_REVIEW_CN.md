STATUS: GO

唯一 blocker 已解除：两端 8,084 cases 完全对齐；estimand、10,000 resamples、bootstrap seed `20260729`、row-seed 派生、双侧 95% percentile CI、case 重采样及跨指标独立 indices 均已归档。按脚本内存复算的 24 行结果与 JSON 逐项完全一致，报告表格的 N、点估计、单位换算及 CI 均受产物支持。

证据边界：Control 是全量 4,042 motions × 2 regimes、固定 checkpoint/协议、单一生成 sampling seed `3407` 的证据；CI 仅覆盖 case variation，不覆盖 sampling seed、training seed 或 checkpoint selection。Ease 仅有 3 个无控制样例和 1 个 hard-control 样例的固定噪声 sweep，只支持“路径已激活但响应弱且样本依赖”，不支持可靠 Ease 能力或与 hard control 普遍可组合。