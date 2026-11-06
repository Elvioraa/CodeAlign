# CodeAlign 双数据集检查记录

本文件保留 2026-10-27 适配检查的证据边界，并于 2026-11-06 更新配置与恢复规则。原始适配基线为 `75d5ecd359d36a64a3af3d64c4fe32e72ed344bc`。历史检查结果不能代替当前提交的测试结果，也不能证明真实数据训练或 AP 已通过。

## 当前检查规则

- `prepare`、运行器和 checker 读取当前阶段 YAML；参考 YAML 只用于人工追溯，不参与配置或源码版本匹配。
- assignment 校验合法场景、CAV、模态、评价 ego 和成员前缀，并保留插入顺序。首次训练不要求匹配任何历史文件摘要。
- 续训保留数据集和实际样本身份检查，验证模型 state_dict 的键、shape 和 dtype；阶段间加载保留已完成父阶段、checkpoint 完整性和冻结参数所有权检查。
- Batch Size、Epoch、LR、Optimizer、Scheduler、num_workers、eval_freq、save_freq、seed 以当前 YAML 为准。详细恢复规则见 [配置与续训说明](CODEALIGN_CONFIG_AND_RESUME.md)。
- 运行记录只用于追溯实际使用的参数和数据，不作为限制正常超参数调整的新门禁。

## 历史本地证据（2026-10-27）

| 检查 | 结果与边界 |
|---|---|
| 两数据集阶段 YAML | 24 份配置通过当时的静态结构检查 |
| Assignment 场景数 | OPV2V 训练/评价 68/16，V2XSet 58/19；不等于真实帧数 |
| 模态映射 | m1 PointPillar、m2 EfficientNet、m3 SECOND、m4 Resnet101 |
| 输入差异 | OPV2V-H m3 读取 32 线扩展；V2XSet m3 使用原始 PCD |
| Codebook/FCF | 64-channel feature、16-code、single segment/level，0.8 m 物理栅格 center crop/pad |
| 训练隔离 | 按 assignment 插入序 mod 4；各源场景集合互斥，各阶段沿用对应 shard |
| CAV1–4 | 同 test split，GT 在输入 cutoff 前构建；不足 n 个成员不丢帧 |
| CPU 测试 | 当时 `tests/test_codealign_protocol.py` 为 11 passed |
| 原生 Quantizer/FCF | CPU 合成 fixture 验证前后向、冻结解码器下梯度、模块归属与 BN 冻结，不代表完整感知网络训练 |
| Resume | CPU fixture 验证 optimizer、scheduler 和 RNG/generator 状态恢复；未验证服务器真实 checkpoint |
| AP | 复用原函数，导出 TP/FP/scores/GT、双精度 JSON 和 8 位 CSV；未运行真实 AP |

测试中发现原生 `codebook.decode(indices)` 的 `_dequantizer._ix` 未初始化。正式路径使用 `get_feature(one_hot)`，当时测试对照显式 embedding lookup；未修改也未声称验证通过未使用的 decode API。

## 当前验证入口

```text
python -B -m pytest -q tests/test_codealign*.py -p no:cacheprovider
python -B -m opencood.tools.check_codealign_doma
```

PowerShell 不展开传给 pytest 的通配符时，应先用 `Get-ChildItem tests/test_codealign*.py` 收集完整路径，再作为参数传入。当前提交的完整测试结果以实际终端输出和交付报告为准；`docs/codealign_protocol_checks.json` 由静态检查器生成，不代表真实数据或 GPU 验证。

## 尚未验证的边界

1. 本机没有真实数据，不能证明真实 split、scene、frame 数量或 sensor 可用性；缺少数据时检查器应直接报告缺失。
2. 完整感知模型依赖 spconv 等环境；CPU fixture 不能证明正式模型的显存、收敛或推理 AP。
3. OPV2V-H 的 `_32.pcd`、depth、visibility/HDF5 需要真实数据检查；V2XSet 原始 PCD + SECOND 已明确，不是待确认线数问题。
4. 最终 range、comm_range、GT、NMS 和 AP 调用仍需对照 DOMA 实际运行记录；当前参考值不是已确认的服务器最终实验快照。
5. 通信范围内成员应满足累计 m1→m2→m3→m4 前缀；推理不通过重排、替代或删帧修饰违规人口。
6. 每阶段单 GPU，仍不支持 DDP。当前 OPV2V-H Batch Size 是用户指定配置，不是实测显存保证。
7. 原生 Quantizer 含随机采样；恢复 RNG 不承诺跨 PyTorch/CUDA/硬件的位级结果一致。
8. 不能把官方只有模型的 `net_epoch*.pth` 重命名为本运行器的完整 checkpoint。兼容性测试针对 CodeAlign 阶段状态格式，仍严格验证模型结构、数据身份与依赖。

本次工作限于本地仓库与 GitHub 功能分支，无服务器操作或 GPU 训练。官方模型、损失、数据划分和评价数学定义不因这次配置管理调整而变化。
