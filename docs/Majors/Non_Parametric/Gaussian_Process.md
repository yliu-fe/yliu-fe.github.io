---
comments: true
---

# 高斯过程

在讨论行为人离散多期学习时，一种合适的方法是非参数拟合——行为人通过思考（或外界信息）获取对应特定状态的最优动作，并更新自身知识库——对任意状态$\mathbf{s}_t$，估计$a^{*}(\mathbf{s}_t)$的分布，并（通常）选取分布的均值作为下一次遇到$s_t$状态时的动作方案。因此，基于高斯核函数的随机过程Gaussian Process（下称“高斯过程”或GP）是一种合适的非参数模型。

/// details | 参考书目、文献和网站
    type: info

1. 图书Gaussian Process for Machine Learning（Rasmussen and Williams，2006），网址: <https://ieeexplore.ieee.org/book/6267323/>
2. 知乎专栏《高斯世界下的Machine Learning》，作者“蓦风星吟”，网址：<https://www.zhihu.com/column/gpml2016>
3. Alvarez, M. A., Rosasco, L., & Lawrence, N. D. (2012). Kernels for vector-valued functions: A review. Foundations and Trends® in Machine Learning, 4(3), 195-266. <https://arxiv.org/abs/1106.6251> (关于多维输出的高斯过程)
///

## 预备知识：从高斯分布开始

就正态分布嘛。比如对于一维随机变量$x$来说，如果$x \sim \mathcal{N}(\mu,\sigma^2)$，则$x$的概率密度函数为：

$$
PDF(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp \left(-\frac{(x-\mu)^2}{2\sigma^2} \right)\tag{1}
$$

再熟悉不过的分布函数，其中$\mu$是均值，$\sigma^2$是方差，二者共同唯一地决定了$x$的概率分布，或者说笛卡尔坐标系下$x$分布曲线的形状。但显然，很多时候一维分布是不够用的，比如说很多系统中状态空间的维度都是高于一维的，这时候就需要多维的高斯分布了，如果$x_1,...,x_n$互相独立，则其联合概率分布为：

$$
\begin{aligned}
& p\left(x_1, x_2, \ldots, x_n\right)=\prod_{i=1}^n p\left(x_i\right) \\
& =\frac{1}{(2 \pi)^{\frac{n}{2}} \sigma_1 \sigma_2 \ldots \sigma_n} \exp \left(-\frac{1}{2}\left[\frac{\left(x_1-\mu_1\right)^2}{\sigma_1^2}+\frac{\left(x_2-\mu_2\right)^2}{\sigma_2^2}+\ldots+\frac{\left(x_n-\mu_n\right)^2}{\sigma_n^2}\right]\right)
\end{aligned}
$$

也不麻烦，由于上面假定了$x_1,...,x_n$的两两独立，故$\mathbf{x} - \mathbf{\mu} = [x_1-\mu_1,...,x_n - \mu_n]$的协方差矩阵$\Sigma$是一个对角矩阵，即$\Sigma = \text{diag}(\sigma_1^2,...,\sigma_n^2)$，从而有：

$$
\sigma_1 \sigma_2 ... \sigma_n = |\Sigma|^{1/2}
$$

那么联合分布的密度函数可以写成向量的形式：

$$
p(x) = (2\pi)^{-n/2} |\Sigma|^{-1/2} \exp \left(-\frac{1}{2} (\mathbf{x}-\mathbf{\mu})^T \Sigma^{-1} (\mathbf{x}-\mathbf{\mu}) \right)
$$

## 高斯过程

### a. 基本结构

首先引用GPML一书（第2.2章，pp.13）中对高斯过程的描述：

> (Definition 2.1) A Gaussian process is a collection of random variables, any finite number of which have a joint Gaussian distribution.
>
> (定义2.1) 高斯过程是一个随机变量的联合，其中任意有限个随机变量的联合分布都是高斯分布。

这一系列随机变量是出现在一个连续域之中的，而且对于任何的时间、空间，这个变量集的任何子集都服从（多维）高斯分布。高斯分布是由期望和方差来构造的，对应的，高斯过程也完全由期望函数（mean func）和协方差函数（covariance func）来构造。对于实过程$f(\mathbf{x})$，分别定义期望函数$m(\mathbf{x})$和协方差函数$k(\mathbf{x}, \mathbf{x'})$：

$$
m(\mathbf{x}) = E[f(\mathbf{x})]; \quad k(\mathbf{x}, \mathbf{x'}) = E[(f(\mathbf{x})-m(\mathbf{x}))(f(\mathbf{x'})-m(\mathbf{x'}))]
$$

从而写出$f(\mathbf{x})$作为高斯过程的形式：

$$
f(\mathbf{x}) \sim \mathcal{GP} [m(\mathbf{x}), k(\mathbf{x}, \mathbf{x'})]
$$

对于任意的$\mathbf{x}, \mathbf{x'}$，$k(\mathbf{x}, \mathbf{x'})$都是一个实数，且$k(\mathbf{x}, \mathbf{x'}) = k(\mathbf{x'}, \mathbf{x})$，即协方差函数是对称的。协方差矩阵也可以很容易地写出来。注意，这里用$k$来表示协方差函数，是因为在GP中，协方差函数也被称为kernel，即广为人知的“核函数”。Shepard（1978）的工作表明人类的泛化学习服从指数衰减规律，因此这里通常会讨论Squared Exponential（SE，平方化指数函数）形式的核函数，即：

$$
K_{SE}(x,x') = \exp \left(-\frac{||x-x'||^2}{2l^2}\right)
$$

### b. 高斯过程回归

> 高斯过程回归的“学习原理”可以参照<https://zhuanlan.zhihu.com/p/44960851>。尽管存在先验分布，但由于缺乏训练数据，基于先验分布的多次采样得到的$f(x)$估计函数可能在$\mathbf{x}$上完全不同。但是，随着观测数据的学习，贝叶斯学习$f(x)$得到后验分布，基于后验分布的多次采样函数可能就比较收敛（主要在有数据的区间内收敛，而缺乏训练数据的区间则可能依然我行我素）。

