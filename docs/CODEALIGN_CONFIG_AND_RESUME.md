# CodeAlign 当前配置与续训规则

2026-11-06：运行依据是实验目录中当前阶段的 `config.yaml`。首次准备从对应数据集模板复制缺失文件；重复准备保留已有配置。参考 YAML 仅供人工审查，不限制运行配置。模型结构、损失、数据划分和评价定义未在本次修改中调整。

## OPV2V-H Batch Size

模板目录：`opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/`。以下 11 份配置只改 `train_params.batch_size`，其余内容保持原样。

| Stage | 修改前 | 修改后 |
|---|---:|---:|
| single/m1 | 1 | 4 |
| single/m2 | 1 | 2 |
| single/m3 | 1 | 8 |
| single/m4 | 1 | 4 |
| code_space/m1 | 1 | 8 |
| code_space/m2 | 1 | 4 |
| code_space/m3 | 1 | 8 |
| code_space/m4 | 1 | 4 |
| fcf/m2_to_m1 | 1 | 2 |
| fcf/m3_to_m1 | 1 | 2 |
| fcf/m4_to_m1 | 1 | 2 |

本次全仓库含隐藏和忽略路径搜索只发现 24 份阶段模板，没有本地实验 `config.yaml` 或 `opencood/logs` 目录，因此没有可同步的 OPV2V-H 实验配置，且未提前创建阶段输出目录。以后首次 `prepare` 使用以上模板。V2XSet 的阶段 YAML、assignment、已有实验日志、checkpoint 和结果不变。

## 超参数恢复

Batch Size、总 Epoch（`train_params.epoches`）、LR、Optimizer、Scheduler、num_workers、eval_freq、save_freq 和顶层 seed 都可修改后使用 `train --resume`。

- LR 改动以当前 YAML 为准。StepLR、MultiStepLR、ExponentialLR、CosineAnnealingLR 和 ReduceLROnPlateau 保留兼容进度并调整学习率基准；其他调度器（例如 LinearLR、OneCycleLR、CyclicLR）在 LR 改变时按当前 YAML 重新开始调度，避免沿用旧的缓存边界。warmup、周期上下限等初始化规则按新配置生效，记录中分别列出配置的 Optimizer LR 和实际有效 LR。
- Optimizer 类型和参数均不变（允许单独调整 LR）时恢复动量/矩等状态；类型或参数改变时安全重新初始化，避免把不兼容的历史状态装入新配置，例如 Adam 新启用 AMSGrad。
- Scheduler 配置改变，或 Optimizer 需要重新初始化时，按当前 YAML 重新初始化 Scheduler，从新调度初始进度开始。恢复策略和有效学习率会明确记录。
- Batch Size、总 Epoch、workers 和 eval/save frequency 直接取当前 YAML，不受历史配置文件格式或版本约束。
- seed 未变时恢复原 RNG/generator 状态；seed 变化时按当前值重新设置全局及 DataLoader RNG。
- 较旧的阶段 checkpoint 没有嵌入 config 时，无法可靠判断旧 Optimizer/Scheduler 类型和 seed：恢复兼容模型与 epoch，重新初始化 Optimizer/Scheduler，并采用当前 seed。该降级策略明确记录，不要求恢复旧配置或源码快照。

模型 state_dict 的键、shape、dtype，数据集和样本身份，已完成父阶段依赖以及 checkpoint 文件完整性仍需通过检查。历史配置或源码版本差异不会独立阻断恢复。已完成阶段及已有 checkpoint 的防覆盖保护保留；这不是覆盖已完成实验的入口。

## 记录与证据边界

各阶段独立保存配置、训练日志和 checkpoint。每次启动/续训在 `run_history/<run_id>/` 中记录实际 YAML、参数变更和恢复策略，新的 TensorBoard 输出写入其中的 `tensorboard/`；保留旧 TensorBoard、日志及有效权重。错误退出保留异常信息。没有新增文件锁或调度锁。

本地没有真实旧 V2XSet checkpoint；旧格式兼容性由 CPU fixture 模拟已有状态格式验证，不能宣称实际服务器 checkpoint 已加载成功。完整 CPU 测试结果见本次交付报告。没有连接服务器或运行 GPU 训练。

## 本次交付核验

起点：`feature/codealign-flexible-hparams`，HEAD `282f9ac97a9d3ae53a0811eb51504837161f0412`，工作区干净。该提交尚未删除历史恢复门禁，OPV2V-H 的 11 份模板 Batch Size 当时均为 1。本次修改是在此基础上的新工作。

2026-11-06 本地验证：

- 全部 4 份 CodeAlign CPU 测试文件：**100 passed in 48.02s**（assignment/prepare/checker/generator 35，超参数/状态恢复 39，原协议契约 11，运行器中断续训/完整性 15）。
- 覆盖首次训练、LR/Batch Size/Epoch/seed/频率变更后的真实运行器 CPU 续训、workers 参数接收、Scheduler/Optimizer 变更、LinearLR/OneCycleLR/CyclicLR 后续学习率、AMSGrad/centered 下一步更新，以及旧格式 checkpoint 和 input audit。
- 负例覆盖样本/成员变化、模型 key/shape/dtype 不兼容、数据集/阶段错位、父权重和最终装配文件完整性，以及已有权重/完成阶段防覆盖。
- 24 份 YAML 双数据集静态检查通过；5 个工具和4个测试文件 Python 编译通过；`git diff --check` 通过。
- OPV2V-H 11 份模板与起点逐份比较，仅 Batch Size 变化；V2XSet 12 份阶段 YAML 不变，官方模型、损失、数据加载/数据工具、assignment JSON 以及 OPV2V-H final 配置无 Git 差异。
- 工具、模板、文档及脚本中的旧配置/源码/AST 门禁关键词无残留。旧字段只存在于两份兼容性测试的模拟历史数据中，用于证明它们不再限制恢复。

测试使用独立的临时目录，没有准备真实实验输出目录。Windows 默认共享 pytest 临时目录有权限限制，最终使用新的独立 `--basetemp` 完成全量测试。

## 修改文件清单

- 工具：`opencood/tools/codealign_protocol.py`、`codealign_state.py`、`run_codealign_doma.py`、`check_codealign_doma.py`、`generate_codealign_doma_configs.py`。
- 配置：上表11个阶段各自的 `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/<stage>/config.yaml`；删除两个数据集模板中废弃的协议约束记录。
- 测试：更新 `tests/test_codealign_flexible_hparams.py`；新增 `tests/test_codealign_assignment_validation.py`、`tests/test_codealign_runner_resume.py`；删除不再适用的旧 assignment 约束测试。`tests/test_codealign_protocol.py` 原样保留并通过。
- 文档：本文件、`docs/DOMA_OPV2V_V2XSET_ADAPTATION.md`、`docs/DOMA_OPV2V_V2XSET_PROTOCOL_CHECK.md`、`docs/codealign_protocol_checks.json`；删除废弃的 `docs/protocol_sources/code_implementation_audit.json`。
- 命令打印脚本：`scripts/codealign_doma_commands.sh`，审查实际实验 YAML 的数据检查入口。
