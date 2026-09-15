# 尚未完成的实验与投稿前优先级

**论文：A Learned Visual Lens: Low-Rank State Adaptation for Vision–Language Models**

这是一份作者侧执行建议，不是论文的已完成实验。核对基线为 EventTune `3150cae1eecf2ee82e508d32fe7f2e80913f8149`。该提交的 evidence audit 已冻结历史结果；下面区分“已完成”“已有补充代码但没有 GPU 结果”“新建议、尚未实现”。本次工作只写作、制表、绘图和编译，没有运行模型实验，也没有更新远端仓库。

## 先做什么

优先解决三个会改变论文解释的问题：**主要优势能否在强、同条件 LoRA 下复现；增益是否依赖真实图像；正确答案梯度是否被实现和答案前缀影响。** 不需要为让所有机器人数据集赢而继续无止境扫参数。新实验失败同样报告。

## P0-A：主要结果的独立复现与强基线

**当前已有：** Qwen2.5-VL-7B 四事件、一个 support24/event 的固定主表；三个 support seeds 的 rank5 Lens 对 Frozen；并没有完整的 rank16 Lens 与 LoRA 同三种子配对主表。其他 backbones 中部分短预算 LoRA 很弱。

**要检验：** 32.08 对29.86的平均 F1 优势是否依赖一次 support 或较短 LoRA recipe。

**最小矩阵：** 四 BRIGHT 事件 × 三个 support seeds × Frozen、LoRA-r16、Random-KV-r16、Full Lens-r16，共48个概念单元。同一 pretrained checkpoint、support24、query IDs、图像预处理、候选顺序和答案区间。Frozen可在严格相同query/checkpoint下复用，不能仅按方法名导入旧分数。

**基线增强：** 另加固定四-pass LoRA（与旧一-pass LoRA分列），以及用独立开发事件选择的 LoRA学习率/停止步数。Lens使用同样的开发信息或固定旧配置。不能只给Lens多次查看query来选参数。

**选择规则：** 先封存配置，再评估新的未用于选择的query tiles；若只能重用已经观察过的query，结果标为重复性核验，不声称恢复了“未见测试集”。少量support的CV必须在每个训练折重新估计B，禁止全support建B后拿其中一部分作validation。空间group split不可行时不要用逐样本split偷偷替代；改用独立开发事件。

**报告：** 每事件/每seed F1、BA、NLL、每类recall和预测类别计数；paired差值及基于tile/执行/slide组的CI；实际optimizer更新、processed examples、时间和显存。不能以参数量比充当训练算力比。

**状态：** 新的完全matched三seed BRIGHT尚未完成。旧增量包配置默认只有BRIGHT seed0，不能以`--seeds 1 2`凭空生成支持manifest。

## P0-B：“眼镜”究竟使用了图像证据，还是主要改了答案先验？

**原因：** 主表Full Lens的F1提高，但平均BA仍约0.3450。现有结果支持F1 trade-off，却未证明“模型原来看错细节”。

**实验对象：** 先选两事件Hawaii/Libya及Camelyon17，同一support训练出的Frozen/LoRA/Lens，不重新选择好看样本。

**输入条件：** (1)真实图像；(2)域内确定性打乱图像与文本配对；(3)同尺寸中性图像替代；(4)BRIGHT只替换post图像，pre与文本固定。保持分辨率/token预算一致。不把输入移除简单称作“移除能力”。在完整预先指定query上评分，不只筛选Lens-correct案例。

**主统计：** 比较各方法增益是否依赖完整图像：
`D = (F1_Lens(real)-F1_Frozen(real)) - (F1_Lens(control)-F1_Frozen(control))`。
同时报告BA、NLL、候选分数和逐类recall；D的CI按相同query组配对重采样。对照可以改变分布，需完整报告而不是只解释正方向。

**配套低成本基线：** 仅用support拟合全局候选logit偏置的模型；不改视觉状态。固定维度K-1并约束和为0，正则由开发数据确定。若它能解释大部分F1提高，应缩窄内部视觉解释。

**状态：** 新建议；本包没有实现或运行这些模型对照。Figure prompt 06是预留设计，当前稿件不含该图和结果。

## P0-C：目标函数、答案前缀与梯度路径审计

**已查到的实现差异：** BRIGHT对support loss求均值；旧单图任务求和后加同一λ惩罚。InternVL单图span包含`Answer:`公共前缀，而其他路径主要用类别词。前缀在候选精确评分中是公共项，但会贡献非零训练梯度，可能影响B和κ，绝不只是一个排版标签。

**先跑无需完整query的门：**
1. 零控制器前后logits一致；重置恢复；保存/读回一致。
2. 每个选中模块存在非零、有限视觉梯度；未选行直接残差严格为0。
3. checkpoint on/off的同一support梯度和loss在允许数值误差内一致，检查backward重计算期间mask是否仍正确。
4. 实际每步梯度与手工mean/sum目标一致；惩罚、clipping和optimizer更新计数日志完整。
5. 同一完整prompt下比较正确类别词span与`Answer: <label>`完整span，列出二者loss、gradient norm、basis overlap和controller梯度。候选scoring路径保持不变。

**最小正式核验：** Camelyon17、RoboFail各seed0，旧sum目标 vs一致mean目标；旧full-span vs label-only梯度。用独立输出目录，不能把改变目标后的分数覆盖旧报告或称为原配置复现。若发现旧实现bug，版本化重跑受影响block并撤下对应旧值。

**状态：** 旧增量包有mean/sum ablation及identity检查，尚无该包GPU结果；完整label-span/checkpoint梯度对照需实现。

## P1-D：补足最有解释力的机制对照

**当前完成：** rank5第二矩、rank5 centered、rank5 random三basis种子、rank1 mean-gradient；Full/Diagonal、层/rank/steps/support/α/LR sweeps。

**仍缺：** rank-matched covariance1 vs mean1 vs random1；activation-PCA-r16；shuffled-label basis（拟合仍用正确support标签，专门破坏basis的正确性信息）；hidden-state residual；zero-controller；K-only/V-only；pre-only/all-visual/text mask；Guardian Random-KV。

**执行：** 使用旧补充包的相应arm，先在四BRIGHT事件完成一轮固定配置，再重复事先选定的关键contrast。样本和参数预算写清；不能用rank1对rank16的胜出证明几何本身优越。hidden baseline不是完整ReFT复现；若标题/相关工作要直接比较ReFT，应补作者官方方法的匹配协议，不给自定义baseline冒用名字。

**状态：** 这些多数已在此前增量包中编码和CPU测试，但不在当前main的已完成证据里。本次ZIP是论文项目，不重新分发可能过期的训练overlay。

## P1-E：rank16端到端效率，以及统计口径

**当前完成：** rank5 compact Lens在Hawaii300/RTX3090的一次query测量，连同basis的存储。

**需补：** rank16Full/Diagonal与LoRA同卡同分辨率；测basis extraction、controller fit、query三个阶段，至少3次独立计时（GPU同步、明确warm-up和I/O边界）。报告峰值allocated/reserved显存、额外state字节、实际token数与优化步数。不能把rank5延迟贴到rank16F1旁边当同一运行点。

**统计：** 原预测产物若仍在ignored runs中，先导出并核对ID/标签，再做paired group bootstrap。支持集合随机性和query组随机性分层处理；Guardian不同seed的query补集可变，禁止假定整suite query恒同。原有reported CI可保留，但不能声称我们在此重算。

## P2-F：从描述性κ到实际控制器更新

**动机：** `dL/dA=(ZB)^T(M*G)B`，而目前κ只看投影后的平均activation梯度。二者不是同一个量。

**建议：** 在support内部训练/验证拆分上记录真实controller梯度的一致性和一步实际loss变化，加入小步有限差分核验。query-label版本只能作oracle诊断。先不发明ρ×max(κ,0)作为普适成功分数；与真实变化的相关性须在多个独立domain和种子上验证。

**状态：** 本次只加入了数学推导，没有声称新实验验证。这项可作后续深入研究，不必为当前九页稿再扩数百个实验。

## 不用再重跑的内容

已完成的单图三种子主表、既有的四事件rank5 sweep、三种子方向性诊断、query-prior几何重加权都可以保留。RoboFail class-conditional controller的失败结果保留。没有必要继续为ManipBench盲扫128/256steps、换赢的数据集或把QKVO偷偷升格成主方法。

## 给运行agent的短指令

先读当前GitHub的evidence audit；再检查旧`vigor_handoff`补丁是否已经由有权限账号应用。没有push的本地补丁不会被`git pull`取回。只运行P0审计与缺失contrast，使用新的结果根目录，保持历史结果只读；写出真实已完成清单与失败项。不要根据query结果删域、选seed、换基线，或承诺必须胜过LoRA。
