# A Learned Visual Lens

**完整标题：A Learned Visual Lens: Low-Rank State Adaptation for Vision–Language Models**

## Overleaf

上传整个ZIP，将主文档设为 **`main.tex`**，编译器选 **pdfLaTeX**。全部正文、表格、算法、附录和可编辑的方法图均在这一个TeX文件中；没有其他标题版本、摘要版本或ICLR示例论文。

保留的 `iclr2027_conference.sty/.bst`、`natbib.sty`、`fancyhdr.sty` 是原样的官方编译依赖，不是额外论文。`references.bib` 是本稿引用。标准LaTeX宏包由Overleaf提供，无须另行上传字体。

本稿使用官方匿名样式。正文从标题到Conclusion为 **9页**；必需声明、参考文献和附录位于其后。`main.pdf` 可通过 `bash build.sh` 从源文件生成。

## 本次内容

开头由已有实验的参数量—F1反差引出问题；接着提出视觉状态低秩残差，给出形式化定义和性质、实验设置、主结果、跨backbone/跨任务结果、消融、方向性诊断、相关工作、讨论、限制和结论。所有数字来自所核对的历史结果或明确的算术推导，没有填入“预计实验结果”。

所用正式证据仍对应EventTune的 `3150cae1eecf2ee82e508d32fe7f2e80913f8149`。这份稿件是已生成写作包的迁移副本。代码现已迁到 StateAdaptation；新的 P0 GPU 结果尚未填入稿件。

## 图和英文prompt

- Figure1：少量视觉状态系数与LoRA的参数量—F1对比（真实数据）。
- Figure2：support拟合 / query复用的方法图（数学示意，TikZ在main.tex）。
- Figure3：basis来源消融（真实数据，rank5与rank1明确区分）。
- Figure4：RoboFail class-prior混合的oracle诊断（不是新的F1）。
- Figure5：附录module-wise方向一致性（真实三seed均值）。

`figure_prompts/ALL_FIGURE_PROMPTS.md` 包含每张图的详细英文描述，以及每图三种不改变事实的构图方案。推荐A与当前排版最接近。第06项是尚缺的视觉因果对照图，仅提供prompt，不出现在论文中。总共18段独立英文prompt，其中15段对应已有5张图，3段为未来实验设计。

已有数值图由 `plotting/figure_data.json` 与 `plotting/make_figures.py` 生成；不要用文生图模型重新“猜”柱高、统计区间或具体数值。重新生成：

```bash
python3 plotting/make_figures.py
```

## 投稿前必读

`author_notes/EXPERIMENTS_TO_COMPLETE.md`：最值得补的实验、已实现但未运行的代码范围，以及哪些旧实验不用再跑。

`author_notes/WRITING_AND_EVIDENCE.md`：ARIS写作skill来源、图形风格来源、每张表与实验源文件的映射，以及对先前过强解释的明确修正。

原始写作包的编译记录不当作本次迁移的重新验证。数值图和 PDF 可通过 `bash build.sh` 重建；GitHub 中的稿件不含新 GPU 结果。

**完整稿不等于实验审计已经全部完成。** 当前最重要的后续是强matched LoRA复现、真实图像依赖对照，以及loss/答案前缀/梯度路径审计。不要把oracle诊断当成部署性能；不要将rank5时间视作rank16测量。作者须核对AI-use statement、实验权限、匿名性及最终投稿材料。

`author_notes/`含作者工作记录和真实仓库链接，**不应直接作为双盲supplement上传**。提交论文PDF；若准备公开/匿名代码补充材料，另行移除识别信息并验证。

## 本地编译

```bash
bash build.sh
```

数值图重建需要Python、NumPy和Matplotlib；仅在Overleaf编译已附PDF图不需要Python。没有修改官方样式、页边距或正文字号。
