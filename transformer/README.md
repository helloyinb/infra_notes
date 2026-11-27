# Transformer：从原论文到架构实现

以《Attention Is All You Need》为主线，按照“读论文 → 画数据流 → 推导形状 → 实现模块 → 实验验证”的顺序学习。先掌握原版 Encoder–Decoder，再衔接现代语言模型和推理工程。

## 从这里开始

1. 打开[本地论文 PDF](papers/attention-is-all-you-need.pdf)，先看 Figure 1（第 3 页）。
2. 按[六周学习计划](learning-plan.md)完成第 1 周任务。
3. 在[阅读笔记](notes/reading-notes.md)中记录自己的解释、张量形状和疑问。

计划按每周 6～8 小时编排，以理解和实现架构为目标。阅读章节、预计用时、练习和验收标准都在计划中；根据基础和验收结果调整节奏。

## 论文来源

| 项目 | 内容 |
| --- | --- |
| 标题 | Attention Is All You Need |
| 作者 | Ashish Vaswani 等 |
| 发表 | NeurIPS 2017（论文中写作 NIPS 2017） |
| arXiv 编号 | 1706.03762 |
| 本地版本 | v7，版本日期 2023-08-02 |
| 下载日期 | 2026-11-27 |
| 官方摘要页 | <https://arxiv.org/abs/1706.03762> |
| 固定版本 PDF 来源 | <https://arxiv.org/pdf/1706.03762v7> |
| 本地文件 | [papers/attention-is-all-you-need.pdf](papers/attention-is-all-you-need.pdf) |
| 文件大小 | 2,215,244 字节 |
| 页数 | 15 页 |

下载的是 2017 年原论文的 arXiv v7 修订版，不是 2017 年首次提交的 v1。计划中的页码均对应此本地 PDF，从第一页开始计数，阅读时优先按章节定位。

SHA-256：

```text
bdfaa68d8984f0dc02beaca527b76f207d99b666d31d1da728ee0728182df697
```

## 辅助资料

- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)：遇到实现问题时对照论文和代码。
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)：建立模块与数据流的直观认识。
- [《动手学深度学习》中文版](https://zh.d2l.ai/)：按需补齐矩阵、自动求导、注意力和训练循环。

论文是主线。辅助资料用于解决具体疑问，每解决一个疑问，就回到论文确认对应公式或模块。

## 已整理的解析

- [前四页解析](notes/01-overview.md)：动机、Figure 1、右移与 mask、注意力公式和手算例子。助手解析已完成，学习验收仍待执行。

## 当前进度

- [x] 下载并验证论文 PDF。
- [x] 制定论文学习路线与笔记模板。
- [ ] 完成基础自检。
- [ ] 第 1 周：整体结构与问题背景。
- [ ] 第 2 周：单头注意力与 mask。
- [ ] 第 3 周：多头注意力与交叉注意力。
- [ ] 第 4 周：完整 Encoder–Decoder 前向计算。
- [ ] 第 5 周：训练、生成与实验验证。
- [ ] 第 6 周：论文复盘与推理工程衔接。

后续实践代码放在 `transformer/code/`，实验记录放在 `transformer/experiments/`，实际开始对应任务时再创建。
