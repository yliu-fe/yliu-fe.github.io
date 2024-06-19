---
comments: true
---

# 我的专业相关领域文档

这一部分内容涉及到本人的专业领域，即金融工程相关工作。这一部分可能会写得比较详细，例如计量经济学的笔记就可能分成很多板块分别更新。

这一部分可能包括如下内容：计量经济学（截面、时序、面板）、金融经济学（数理金融）、统计推断和因果推断

此外，可能还会包括的内容：（传统）强化学习

## 写在前面：金融工程思路及该方向的学习路径

金融工程，或者说金融经济学，是金融学这个经济学主要分支中比较数学化的一种，其核心问题在于asset pricing，即对资产的定价。

对此领域较为广泛的学习路径的讨论，请参考[我老同学的博客](https://dingdebin.github.io/)。

### FE as a course

作为一门单独的课程，"金融经济学"首先刻画理性人投资者的两面：对收益的偏好（偏好表示问题与效用函数的定义）、对风险的厌恶（绝对和相对风险厌恶），从而推导出一阶和二阶随机占优——一阶占优强调收益分布，二阶最优强调同收益资产的风险最小化。

接着，我们拿着这个投资者的小泥人，分别从不同的角度出发探讨最常见的金融资产——股票的定价，也就是CAPM和APT这两个经典的资产定价模型。

- 对于前者，我们回到1952年马科维茨（Markowitz）对金融学的创始性研究"Portfolio Selection"，利用前面所学习的二阶随机占优内容，刻画投资者最简单的资产配置方法：均值-方差模型（Mean-Variance Optimization，MVO）。在对MVO的讨论中，我们一步步推出了资产组合有效前沿面方程。
- 对于后者，我们首先讨论两基金分离性质，并将市场定价修正的思路从所有人理性和资产变动，变成了少数人的大头寸套利——市场的均衡将不会给任何人以套利的机会。从APT开始，金融学领域的多因子模型就开始发展起来。

然后，我们将视线逐步从股票变成期权，先从CAPM的思路上走一走，构造一个含有一支股票及其$n-1$个看涨期权的投资组合，利用Arrow-Debreu资产思想，讨论其状态价格，并以此为基础，在一个两期模型中向看涨期权的定价展开攻坚。是的，你从投资学开始就看着你的各个老师用各种形形色色的方法推了无数遍的，BSM公式，被我们用状态价格的方式再一次推导了出来。之后的Greek Letters，金融工程对此点到为止，没有更深的讨论。最后，我们还可以讨论一下多期模型的问题。

此外，从维纳过程和伊藤引理这熟悉的方向，用套利定价的方法再推一次BSM公式，金融经济学这门课程的内容差不多就是这些。

这门课程的先修课程是比较复杂的，主要包含三类：

1. 数学。高数、线性代数、概率论传统三项保底。这门课涉及到了大量的积分、矩阵运算和条件概率分布讨论，但就课程本身而言，数学工具不算复杂，因为不涉及三重积分、曲线曲面积分，也不涉及傅里叶变换等内容，考研数学三的内容基本够用。**随机过程**和**微分方程**要有一定的功底，BSM公式本身就是一个大PDE的解（论Merton为什么能挂名）。还有**最优化**理论的学习经历。此外，推荐有实分析学习经历，因为很多定义涉及到测度论。
2. 经济学。课程大量使用了微观经济学的概念，如效用、偏好、风险厌恶等，也有部分内容来自宏观经济学，如状态价格。
3. 金融专业先修课。其实就是公司财务、投资学、衍生工具这三门，递进的关系。

### FE as a major

金融经济学本身的逻辑链路是”（会计学）-- 公司财务 -- 投资学 -- 金融衍生工具 -- 金融经济学“，尽管主要讨论的是股票和期权，但相关研究同样对固定收益证券、互换、期货有一定的研究，此外，这个领域的相关模型和研究思路也外溢到了其他金融领域，如银行间市场相关研究。

除了上面所说的FE先修要求外，在这个专业从事研究工作要有的额外基础包括：

1. 数学和统计学。这里指的统计学是统计推断理论，此外，要学习统计学衍生出来的计量经济学，作为金融领域研究的常用数据分析工具。
2. 经济学其他课程。如博弈论、信息经济学都常用于金融微观结构领域的讨论。
3. 金融学其他课程。包括风险管理、行为金融学等。投资学中的固定收益证券常作为单独科目，作为现实中规模最大的交易品，请着重学习。
4. 统计分析和科学计算工具。前者常用的是Stata和R，后者常见于Python(+Anaconda)和Matlab。如您决定从事一定的理论研究，可以考虑Mathematica或者Sagemath。

最后，祝您好运。