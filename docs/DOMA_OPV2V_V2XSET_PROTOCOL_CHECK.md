# CodeAlign 双数据集协议检查记录

检查日期：2026-10-27（Asia/Shanghai）。基线：`75d5ecd359d36a64a3af3d64c4fe32e72ed344bc`。全部修改限于 CodeAlign；未提交、未推送、未部署、未运行正式训练。

## 已完成的检查

| 检查 | 结果与边界 |
|---|---|
| 两数据集 YAML | 38/38 新增 YAML 可解析，其中 24 份阶段配置，另含锁/manifest/参考副本；模型/数据/父阶段/预算来源校验通过 |
| 协议锁与 manifest | 各两份，路径与 DAG 完整；最终服务器协议仍为 SERVER_VALIDATION_REQUIRED |
| DOMA source copies | 10 份 YAML 保留来源与 SHA256；OPV2V 为保存运行快照，V2XSet 为本地参考 |
| Assignment | OPV2V 68/16 scenes，V2XSet 58/19 scenes；与本地 DOMA 语义及顺序相同，raw/ordered hash 锁定 |
| 模态映射 | m1 PP、m2 EfficientNet、m3 SECOND、m4 Resnet101；V2XSet 禁止 beam rewrite |
| 相机与 SECOND 接口 | RGB 尺寸/BEV grid/depth、voxel .1 与 XYZI 4 维配置同源校验通过 |
| 原生代码审查 | 15 个关键符号中 14 个 AST 相同；GT 多矩阵差异分支在本协议不执行 |
| 实际 GT 调用 | DOMA ordered entry 使用 visualize=True；新入口保留该标志，所有成员 GT 来自 LiDAR 标签 |
| Codebook/FCF 形状 | 64-channel feature，16-code，single segment/level；0.8 m 物理栅格 center crop/pad |
| 训练隔离 | 按 assignment 插入序 mod 4；源场景集合互斥，所有阶段绑定同源 shard |
| CAV1–4 | 参数真实传给 Dataset；同 test/in-order assignment；GT 在输入 cutoff 前构建，样本不足 n 个不丢帧 |
| AP | 复用原函数；原始 TP/FP/scores/GT + 双精度 JSON + 8 位 CSV；汇总检查样本和 GT hash |
| Resume | CPU fixture 验证 optimizer、scheduler、RNG/generator 与下一次更新完全恢复；不是正式模型训练续训证明 |
| Python | 11 个新增或修改 Python 文件编译检查通过 |
| `.gitignore` | 数据/权重/logs 被忽略；两份文档、source evidence 和命令打印模板可纳入版本控制 |
| `git diff --check` | 通过 |

CPU 测试命令：

```text
python -B -m pytest -q tests/test_codealign_protocol.py -p no:cacheprovider
11 passed
python -B -m opencood.tools.check_codealign_doma --output docs/codealign_protocol_checks.json
OPV2V: PASS_STATIC (12 stage configs)
V2XSet: PASS_STATIC (12 stage configs)
```

11 项测试覆盖：配置及错误模态拒绝、分组与 assignment 顺序、V2XSet 原始 PCD/OPV2V 32 线路径、prepare 不覆盖、BEV 物理补零、真实原生 Quantizer 前后向及冻结解码器后的 FCF 梯度、完整状态恢复、缺失父权重拒绝、不同评估样本拒绝汇总；另含完整最终 tensor 所有权/冻结父权重校验与 BN/梯度冻结范围检查。FCF 梯度 fixture 的输入是合成 logits，不代表本地运行了完整 ConvNeXt Translator/相机/SECOND 网络。

一次测试使用原生未被主路径调用的 `codebook.decode(indices)` 时发现其 `_dequantizer._ix` 未初始化。测试已按正式 CodeAlign 路径 `get_feature(one_hot)` 对照显式 embedding lookup 验证。未修改、也未声称验证通过这个未使用的原生 decode API。

## 服务器待验证与已知限制

1. 本机缺少 dataset；`--check-data --dataset v2xset` 正确以 `SERVER_REQUIRED: .../dataset/V2XSET/train` 缺失报错，没有伪造 DataLoader 成功。OPV2V 同样缺真实数据。
2. Python 3.12.4 + torch 2.7.0+cpu；缺 spconv、torch_scatter、timm、icecream、tensorboardX，无法构建完整真实感知模型。没有启动完整 GPU 模型、训练或 AP。
3. 真实 split/scene/frame 数量、所有 CAV 的 sensor 可用性、原始/32 线 PCD 与 depth/visibility/HDF5 需服务器全量 audit。工具提供 `--frames-per-scene N`，使用正 N 时只能报告采样检查；默认 0 扫全部帧。
4. V2XSet 原始 PCD + SECOND 已由用户明确，**不作为未决线数问题**；需验证的是本仓库实际读入和 full forward，不要求补 `_32.pcd`。
5. 最终 range/comm_range/GT/NMS/AP 调用必须对照 DOMA 实际服务器记录。当前锁保存了本地来源和候选数值，没有宣称它就是已验证的最终运行标准。
6. 保留现有 DOMA 的通信范围内成员 prefix；真实帧少于 n 个成员，或某个槽位在通信范围外时，不能保证每帧恰有 n 种模态。必须审查 ledger；非 m1→m2→m3→m4 累积前缀会使推理整体报错。不可改 ego、偷偷重排、过滤帧或用测试 AP 选模型。
7. 一阶段一 GPU；DDP 明确拒绝，未实现/验证分布式 sampler。两 GPU 用于互不依赖的阶段进程并行。
8. 默认 batch=1 不是显存保证。大范围 SECOND、4 cameras/LSS、FCF/codebook 和最终多成员融合的峰值显存必须实测。
9. 原生 ego Quantizer 包含随机采样；本适配恢复 RNG，不承诺跨 PyTorch/CUDA 版本或不同硬件位级相同。
10. 所有官方模型、损失和评估函数仍保留原始版本。新训练入口用自己的状态格式，不能把官方 `net_epoch*.pth` 随便重命名成协议的 `best.pth`；也不能把 DOMA 特有 checkpoint 作为完成阶段。

## 修改清单

下面逐文件清单由当前 Git 工作区生成；新文件尚未 git add，故普通 `git diff --stat` 只统计三个 tracked 文件。全量路径另列，不通过暂存制造统计。

| 相对路径 | 修改内容 | 理由 | 对官方行为的影响 |
|---|---|---|---|
| `.gitignore` | 只放行本次文档/脚本，忽略权重与数据 | 保护实验产物 | 仅 Git 可见性 |
| `docs/DOMA_OPV2V_V2XSET_ADAPTATION.md` | 适配/检查/命令与限制报告 | 交付完整审查依据 | 仅文档 |
| `docs/DOMA_OPV2V_V2XSET_PROTOCOL_CHECK.md` | 适配/检查/命令与限制报告 | 交付完整审查依据 | 仅文档 |
| `docs/codealign_protocol_checks.json` | 本地静态验证结果 | 可审查机器记录 | 无运行影响 |
| `docs/protocol_sources/code_implementation_audit.json` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/opv2v/final.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/opv2v/m1.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/opv2v/m2.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/opv2v/m3.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/opv2v/m4.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/v2xset/final.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/v2xset/m1.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/v2xset/m2.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/v2xset/m3.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `docs/protocol_sources/v2xset/m4.yaml` | 原始参考 YAML 副本或 AST 审计记录 | 区分历史运行快照与本地参考 | 仅证据，无运行影响 |
| `opencood/data_utils/datasets/basedataset/opv2v_basedataset.py` | basename 跨平台修正；协议专属 XYZI reader 分支 | Windows 路径与原始 PCD 兼容 | 非协议 Linux 行为保持；Windows 路径修复 |
| `opencood/data_utils/datasets/codealign_protocol_dataset.py` | 分组、场景排除、正确 sensor map 和 GT mode | 真实复用两类 BaseDataset | protocol 专用 |
| `opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py` | protocol key 启用独立 Adaptor | 隔离 seed/编号/beam 策略 | 默认不启用 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/code_space/m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/code_space/m2/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/code_space/m3/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/code_space/m4/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/fcf/m2_to_m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/fcf/m3_to_m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/fcf/m4_to_m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/final_infer/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/manifest.yaml` | 阶段依赖与 checkpoint DAG | 禁止跳阶段/跨数据集加载 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/protocol_lock.yaml` | 有来源和状态的独立参数锁 | 避免猜测正式参数 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/single/m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/single/m2/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/single/m3/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/single/m4/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/code_space/m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/code_space/m2/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/code_space/m3/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/code_space/m4/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/fcf/m2_to_m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/fcf/m3_to_m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/fcf/m4_to_m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/final_infer/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/manifest.yaml` | 阶段依赖与 checkpoint DAG | 禁止跳阶段/跨数据集加载 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/protocol_lock.yaml` | 有来源和状态的独立参数锁 | 避免猜测正式参数 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/single/m1/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/single/m2/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/single/m3/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/single/m4/config.yaml` | 该数据集/阶段完整 YAML | 四模态和原生训练预算可审查 | 新增，官方 YAML 不变 |
| `opencood/models/codealign_doma_protocol.py` | 原生 CodeAlign 模块组装及物理 BEV 接口 | 保留 FCF/Codebook 且支持统一成员 | 独立模型入口 |
| `opencood/tools/check_codealign_doma.py` | 静态、全量/采样数据与显式 CUDA smoke | 区分 PASS_STATIC 与 SERVER_REQUIRED | 独立工具 |
| `opencood/tools/codealign_protocol.py` | 配置/assignment hash、DAG、prepare/prefix 规则 | 冻结可核验协议并拒绝覆盖 | 独立工具 |
| `opencood/tools/codealign_state.py` | 严格初始化/装配和完整恢复 | 分离预训练/选权重/续训 | 独立工具 |
| `opencood/tools/generate_codealign_doma_configs.py` | 显式只读提取 DOMA 来源后生成新配置 | 来源可复查；trainer 不调用 | 独立工具 |
| `opencood/tools/run_codealign_doma.py` | 显式 train/assemble/infer/summarize | 逐阶段日志、正确 AP/数据 provenance | 独立工具 |
| `opencood/utils/codealign_data.py` | PCD 读取与成员映射 | V2XSet 原始 PCD、不进入 beam 分支 | protocol 专用 |
| `scripts/codealign_doma_commands.sh` | 仅打印全部阶段执行命令 | 可审查，不自动启动作业 | 无运行行为 |
| `tests/test_codealign_protocol.py` | CPU 正负契约与原生 Quantizer 梯度测试 | 验证高风险接口/恢复/隔离 | 仅测试 |

## Git 快照

```text
$ git status --short
 M .gitignore
 M opencood/data_utils/datasets/basedataset/opv2v_basedataset.py
 M opencood/data_utils/datasets/intermediate_heter_fusion_dataset.py
?? docs/
?? opencood/data_utils/datasets/codealign_protocol_dataset.py
?? opencood/hypes_yaml/opv2v/doma_protocol_codealign_v2/
?? opencood/hypes_yaml/v2xset/doma_protocol_codealign_v2/
?? opencood/models/codealign_doma_protocol.py
?? opencood/tools/check_codealign_doma.py
?? opencood/tools/codealign_protocol.py
?? opencood/tools/codealign_state.py
?? opencood/tools/generate_codealign_doma_configs.py
?? opencood/tools/run_codealign_doma.py
?? opencood/utils/codealign_data.py
?? scripts/
?? tests/

$ git diff --stat
 .gitignore                                           | 18 ++++++++++++++++++
 .../datasets/basedataset/opv2v_basedataset.py        | 20 ++++++++++++--------
 .../datasets/intermediate_heter_fusion_dataset.py    |  6 +++++-
 3 files changed, 35 insertions(+), 9 deletions(-)
```

新增文件共 52 个；tracked 修改 3 个。未提交、未推送。

