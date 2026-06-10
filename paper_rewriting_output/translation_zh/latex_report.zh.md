# LaTeX 报告（中文）

## 编译状态

**编译：成功** — 会议版（场景：NeurIPS/ICML）。完整流程：pdflatex → bibtex → pdflatex ×2，零警告。

## LaTeX 源文件

- **主文件：** `final_paper/main.tex` — 完整、可编译的 LaTeX 源文件（约 4500 词）
- **参考文献：** `final_paper/references.bib` — 23 条 BibTeX 条目，全部已解析
- **文档类：** `article`（11pt，margin=1in，letter paper）
- **引用格式：** `plainnat`（natbib 作者-年份）
- **章节：** 6（引言、相关工作、方法、实验、讨论、结论）+ 附录
- **图表：** 0（工作流图占位符被注释掉——图文件存在于 `figures/stable-eureka_workflow.png`）
- **表格：** 2（严格审计运行历史、放宽审计运行历史）

## 源文件检查

| 检查项 | 状态 | 备注 |
|---|---|---|
| 所有 `\section{}` 有 `\label{}` | 通过 | 全部 6 节 + 附录已标注 |
| 所有 `\cite{}` 键在 references.bib 中 | 通过 | 全部 23 个引用键已解析，零未定义 |
| 无未转义特殊字符 | 通过 | 检查了文本中的 `&`、`%`、`$`、`_` |
| 表格使用 booktabs | 通过 | 两个表格均使用 `\toprule`、`\midrule`、`\bottomrule` |
| 摘要存在 | 通过 | 约 150 词 |
| 无 `\input`/`\include` 宏 | 通过 | 单文件自包含 |

## 编译日志

| 指标 | 值 |
|---|---|
| 输出页数 | 10 |
| 警告 | 0 |
| 未定义引用 | 0 |
| 溢满/欠满盒 | 0 |
| PDF 大小 | 约 197 KB |
| 编译时间 | <5 秒 |

## 备注

- 使用 `article` 类/1 英寸边距的 10 页大致相当于 NeurIPS/ICML 双栏格式的约 8 页。正式投稿时需采用目标会议的官方 style 文件。
- 工作流图（`figures/stable-eureka_workflow.png`）存在但被注释掉——准备就绪后取消 `\includegraphics` 块的注释并调整图表位置。
- 作者元数据已匿名化用于双盲审稿。
