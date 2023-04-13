# 作业1： 模仿学习

作业内容PDF：[hw1.pdf](https://rail.eecs.berkeley.edu/deeprlcourse/static/homeworks/hw1.pdf)

框架代码可在该仓库下载：[ Assignments for Berkeley CS 285: Deep Reinforcement Learning (Fall 2022) ](https://github.com/berkeleydeeprlcourse/homework_fall2022)

该项作业要求完成模仿学习的相关实验，包括直接的行为复制和DAgger算法的实现。由于不具备现实指导的条件，因此该作业给予一个专家策略，来做数据的标注。

最后，利用OpenAI Gym上的若干个benchmark 连续控制任务，来比较直接模仿学习和DAgger的表现。

/// details | 关于绘图规范
    type: danger
    open: true
UCB给出了一套在实验报告中绘图的规范：[viz.pdf](http://rail.eecs.berkeley.edu/deeprlcourse/static/misc/viz.pdf)。可参照，利用matplotlib.pyplot和seaborn完成。

> 请注意，这套规范的示例代码中，引用了 `seaborn`库中的 `tsplot`函数，而该函数在较久远的seaborn更新中已被移除，其替代品为 `seaborn.lineplot` 函数。

1. 代码应该在外部文件（如csv或pkl文件）中保存实验的结果，而不是直接出图。这样的话可以对实验反复尝试绘图，直到有一个好的结果。通常记录的信息包括：（1）每次迭代的平均奖励或损失、部分采样的轨迹、有用的二级指标（Bellman偏误或者梯度大小）
2. 绘图应当单独写一个脚本。如果采用不同的超参数或者随机种子运行算法，或者运行了不同的算法，或者运行当前算法的变体，那么最好把所有的实验日志数据加载到一起并绘制到一张图上，记得做好图例和颜色方案。
3. DRL算法，特别是其中的Model-free算法，在不同次的运行中产生的结果会有很大区别，所以多做几组随机种子来做几次实验，将他们的运行轨迹画在一张图上，最好用粗一点的线再画上他们几个的平均表现。也许平均值图和标准差图比较方便，但这种随机实验不见得遵循正态分布，所以把所有的运行图画出来，可以更好的了解随机种子之间的差异。
/// 