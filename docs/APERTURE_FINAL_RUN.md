# 最后一轮实验：运行说明

入口：`scripts/run_aperture_final.sh`；协议：`docs/APERTURE_FINAL_PROTOCOL.md`。
本轮**不改主方法，不覆盖旧实验，不用 query 来调参**。

## 0. 代码版本与本地资产

本轮代码已通过 GitHub 连接器提交。GPU 机器在 `yuhanlydia/StateAdaptation`
的 `main` 分支执行 `git pull --ff-only origin main`，确认存在
`scripts/run_aperture_final.sh` 后，直接按下面步骤运行；不再需要下载、应用
上一版离线补丁，也不要重复运行 `apply_and_push.sh`。

模型沿用本地 Qwen2.5-VL-7B-Instruct，路径和原数据路径在
`configs/aperture_final_round.json`。检查路径后再首次运行 `plan`；首次计划会锁定
配置与源码签名。模型版本、图像大小、已有 query ID 不得在得到结果后更换。
若模型路径不同，可以提前修改配置中的路径，但不可悄悄换模型。

与旧实验相比，标注总量不变，每类原有8个 support 拿2个留作校准，剩6个做
模型/控制器拟合。BRIGHT 18+6、医学12+4、ManipBench24+8。三个 support seeds
仍是已有0/1/2，不是新数据或完全未见的测试集。生成的新 split 留在新的 prepared
目录；原始 support/query 文件不改。Frozen+bias 使用同样总量的原 support 标签。

需要24GB或以上显存。16GB不可通过换量化或降低分辨率直接冒充本协议。
依赖沿用仓库现有环境；绘图额外需要 matplotlib。**不要为这轮盲目升级
transformers/peft**，先使用已经跑通旧实验的环境。CPU 测试只验证数值/协议/调度
契约，不等于已验证真实VLM显存和CUDA前后向。

## 1. 先检查，再一次运行

```bash
cd StateAdaptation
PYTHONPATH=src python3 -m pytest -q tests_aperture_final
bash scripts/run_aperture_final.sh plan
bash scripts/run_aperture_final.sh prepare
bash scripts/run_aperture_final.sh check
```

`plan` 应显示96个状态、24个比较单元、72个非冻结拟合任务。
`check` 逐项检查模型和图片路径、原/新 manifest 内容、类别预算和组隔离。
少文件、support/query重叠、已锁定文件被修改，都明确停止，不自动另找数据。

先使用一个真实任务单元确认接口和显存。该步骤的正式状态会在全量运行时复用：

```bash
bash scripts/run_aperture_final.sh run 0 --domains hospital_2 --seeds 0
```

这个单元包含Frozen、LoRA1、LoRA4、Aperture及相应校准，不是新超参数搜索。
它会先执行单图真实梯度审计。正式全量入口会补齐配对图像审计。

```bash
# 一张卡
bash scripts/run_aperture_final.sh run 0

# 四张独立24GB+卡，每张卡同时最多一个模型进程
bash scripts/run_aperture_final.sh run 0,1,2,3
```

执行顺序固定：support-only真实模型审计 → 拟合和校准 → 封存LoRA选择与温度 →
query及固定干预 → 完整汇总。任一执行返回非零，后续阶段不启动；当前已启动的
其他卡任务可能会正常结束。数值有限但性能较差不是技术失败，必须保留。

## 2. 中断与恢复

再次执行相同命令，已完整封存且哈希一致的状态会跳过；不完整query逐行恢复。
未经封存的拟合失败可以用同配置重新拟合。源码、配置、数据或模型改变后，不
能复用旧封存状态：需新版本目录并记录原因，禁止手改hash绕过检查。

存在 `RUNNING.lock` 时先确认原进程已退出；不能在仍运行时删锁。
杀死作业树时应停止所有子进程，不能只关外层终端。

可以分组跑，但计划不会被过滤条件缩小：

```bash
bash scripts/run_aperture_final.sh run 0 --domains hawaii-wildfire libya-flood --seeds 0 1 2
bash scripts/run_aperture_final.sh run 0 --domains noto-earthquake turkey-earthquake --seeds 0 1 2
# 医学/操作问答仍需要随后完成；部分完成不会输出可投稿总表。
```

独立阶段仅用于技术恢复：

```bash
bash scripts/run_aperture_final.sh run 0 --stages audit fit
bash scripts/run_aperture_final.sh run 0 --stages evaluate
```

第二条仍要求已有合格审计与全部比较臂的封存拟合/校准，不允许先看query再选T。

## 3. 结果和论文产物

```bash
bash scripts/run_aperture_final.sh summarize
python3 scripts/plot_aperture_final.py
```

主要输出在 `runs/aperture_final_v1/paper_exports/`：

| 文件 | 内容 / 论文用途 |
|---|---|
| `coverage.json` | 计划/完成数量、错误；必须`paper_ready=true` |
| `clean_per_seed.csv` | 所有臂、三个seeds、原始和校准后的五项指标 |
| `clean_mean_sd.csv` | 同一query上的support-seed均值及样本SD |
| `calibration.csv` | T、calibration NLL、选中的LoRA轮数及依据 |
| `paired_contrasts.csv` | 各seed的Aperture与calibration-selected LoRA差值 |
| `table_nll_part1.tex` / `part2` | BRIGHT及医疗/问答的四域小表，避免8域表过宽 |
| `table_macro_f1_part*.tex`, `table_ece_part*.tex` | 相同状态的互补指标；不能只挑获胜指标假称全面更好 |
| `corruptions.csv` | 固定60-query探针：clean、noise、JPEG；raw与同一T |
| `residual_dose.csv` | 不重新训练的0/.5/1残差强度探针 |
| `resources.csv` | 更新步数、样本访问、优化/基底数量、序列化大小、延迟 |
| `figures/` | 基于真实完整输出的PDF/SVG/PNG，无编造的均值或误差条 |
| `full_results.json`, `RESULTS.md` | 可供写作端自动读取的总记录与解释范围 |

`--allow-partial`只输出覆盖率，不生成不完整论文表。所有原始预测、图片路径、
controller和adapter留在被忽略的 `runs/`；不要上传临床/机器人图像、私有权重
或完整原始预测。公开前检查聚合产物不含不应披露的信息。

## 4. 怎么把结果放回论文

新增表名建议为 **Calibration-controlled comparison under a fixed label budget**，
不可把减少拟合样本后的新分数直接替换原support24/16/32主表而不改协议说明。
主文最优先一张raw-vs-temperature NLL小表；对应F1/BA/Brier/ECE保留附录。
噪声和残差dose图是固定设置的机制探针；显著性并不是继续添加seed的门槛。

只运行这份已固定矩阵。跑完后将完整聚合产物交回写作端；根据实测结果决定
结论力度，不让agent自行新增“可能更有利”的数据集、rank、层和alpha。
