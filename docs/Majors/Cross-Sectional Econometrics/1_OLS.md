---
comments: true
---

# 一、线性模型和最小二乘法

这一部分是初级计量经济学的起始内容。在这一部分中，我们探讨计量经济学的基本假设，并介绍最小二乘法思想及其相关的假设检验。

/// admonition | “假设x”的指代
    type: info

考虑到下文中大量出现了“假设1”之类的指代，因此这里将提前明确各个假设的指代：

- 假设1： 线性模型假设（1.1）
- 假设2： 严格外生性假设（1.2）
- 假设3： 无完美多重共线性假设（1.3）
- 假设4： 条件同方差假设（3.2）
- 假设5： 正态分布假设（3.4）
///

## 基本假设

计量经济学的前三个基本假设分别是：线性性、严格外生性、无完美多重共线性。

### 1.1 线性模型假设

首先是线性模型假设，传统的计量经济学认为经济模型应当服从线性关系，即：

$$
y_i = \beta_1 x_{1i} + \beta_2 x_{2i} + ... + \beta_k x_{ki} + \epsilon_i
$$

其中$y_i$被称作被解释变量(dependent variable)，$\beta$为参数(parameter)，$x_i$为解释变量（有各种称呼，比如regressor, independent variable)，$\epsilon$为误差项（模型必然存在无法解释的误差）。

> 在以后的内容中，默认$x_{1i} = 1$，即截距项。另外，$x_i^2$这种平方项（正幂次项），或者$x_i x_j$这种交叉项，也可以被纳入线性模型中，即看作一个独立的解释变量。

线性模型的基本形式就是“参数$\times$解释变量”并求和。由于直接写代数式太长，可以用向量的方式来简化表示，假设有$k$个参数，则令$X_i = [x_{1i}, x_{2i},...,x_{ki}]^T$，$\vec \beta = [\beta_1, ...,\beta_k]^T$，从而有$y_i = x_i^T \vec \beta + \epsilon_i$。

现假设有$N$个观测值，则令$Y = [y_1, y_2, ..., y_N]^T$，而令完整的样本集$X = [X_1^T, X_2^T,...,X_N^T]^T$ 为：

$$
X = 
    \left[ \begin{array}{cccc}
        x_{11} & x_{21} & ... & x_{k1}  \\
        x_{12} & x_{22} & ... & x_{k2} \\
        ... & ... & ... & ...\\
        x_{1N} & x_{2N} & ... & x_{kN}
    \end{array}\right]
$$

另设$\vec \epsilon = [\epsilon_1, \epsilon_2, ..., \epsilon_N]^T$，则有线性模型的基本表达形式：

$$
Y = X \vec \beta + \vec \epsilon
$$

### 1.2 严格外生性假设

即$E[\epsilon_i | X] = 0$。**给定观测值$X$时，误差项的条件期望为0。**这与期望迭代法则（Law of iterated expectations，LIE)有关。

给定两个随机变量$u,v$并知其联合密度函数$f(u,v)$，求其边际密度函数，例如$u$的边际密度函数为$f(u) = \int f(u,v) dv$；而条件密度函数则需要利用贝叶斯公式：$f(u|v = a) = \frac{f(u,v = a)}{f(v = a)}$，那么就可以讨论条件期望和期望迭代法则了。

**期望迭代法则：** $E_v[E(u|v)] = E(u)$。*即对于所有的$v$取值情况下，$E(u|v)$的期望是$E(u)$。*

> 期望迭代原则经常会用在宏观经济学的动态均衡模型中，利用T期信息集来求解之后的消费决策，如知信息集$I_t$时求解t+2期的消费决策，$E_t[C_{t+2}] = E[C_{t+2}|I_t] = E_t [ E(C_{t+2} | I_{t+1})]$，另一个用途就是接下来要讲的严格外生性。

/// details | 期望迭代法则的证明
    type: success

$$
\begin{align}
    E[u|v] &= \int uf(u|v) du\\
    E_v[E(u|v)] &= \int (\int uf(u|v) du) f(v) dv = \int \int uf(u|v) f(v) du dv\\
    &= \int u (\int f(u,v)dv) du = \int u f(u) du = E(u)
    \end{align}
$$
///

接下来继续讨论严格外生性问题，由于知道了全部观测值$X$，就可以将这些常数$x_{ji}$乘进去：$E[\epsilon_i x_{ji}] = 0$，从而有$E_x [ E(\epsilon_i x_{ji} | x)] = 0$，从而有$E[\epsilon_i x_{ji}] = 0$，而如果$x_{1i} = 1$，那么就有$E[\epsilon_i] = 0$。而$cov(x_{ji}, \epsilon_i) = E[x_{ji} \epsilon_i] - E(x_{ji})E(\epsilon_i) = 0$，从而有$x_{ji}$与$\epsilon_i$正交（独立）。

有了以上两个基本假设，模型（及其参数的估计值）便有了意义，即：

$$
E[y_i|X] = \sum_{j=1}^k \beta_j x_{ji}
$$

或者说偏导数$\partial E(y_i|X)/\partial x_{ji} = \beta_j$。其经济学意义在于，*平均而言（因为式子里是期望），在其他的变量不变的情况下（偏导数的意义），如果$x_{ji}$增大了$\Delta x_{ji}$个单位，那么$y_i$将会提高$\beta_j \Delta x_{ji}$。*

### 1.3 没有完美的多重共线性

假设3的严格表达应该是：*矩阵$X$（其规模为$n\times k$）的秩$rank(X) = k$*。

> 严格来讲，这句话是with probability 1，而不是绝对的。

此外，（1）这句话还暗含了$n \geq k$，即观测点数应大于等于变量数的原则（否则会有无数组解）；（2）观测值的任意一列**不能被其他列线性表出**（各变量不能具有完全的多重共线性，否则模型无法估计这些变量的参数，即变量“不可被识别”。）。

从线性代数的角度，如果存在完美的多重共线性，则$rank(X) \not = k$，因而有$rank(X^T X) < k$，$X^T X$这个正方形矩阵不可逆了，从而无法估计参数。

> 参照$rank(X^T X) = rank(XX^T) = rank(X)$

此外，受限于计算机浮点数精度的限制，可能$X^T X$的行列式趋近于0，导致精度不够，或者其逆矩阵的行列式趋近于无穷，出现无法估计的情况。当这个行列式趋近于0时，其实就表明这个模型中存在比较严重的多重共线性了（如果等于零，则必然是存在完美的多重共线性），我们可以通过对$X^TX$做特征值分解来考察这个模型，如果特征值趋近于0，则就可以知道$X^TX$行列式趋近于0。或者，多重共线性可以通过VIF指标监测，若$VIF > 10$，则可以认为多重共线性存在。至于如何解决多重共线性，筛选变量即可。

## 最小二乘法估计

/// admonition | 变量上边加hat
    type: warning

从现在开始，带着帽子头的变量全都是估计值。由于恁Mathjax不支持 `\bm`也不支持 `\boldsymbol`，我就只能在帽子下面加箭头，太傻了。为了省一省我的工作量，除了下面的式子，后面带帽子的变量就不再直接区分标量和向量了，不过我想应该比较好分开。
///
目标：最小化估出模型的残差平方和，即：

$$
\hat{\vec{\beta}} = \operatorname{arg}\min_{\beta} \sum_{i=1}^N \epsilon_i^2 = \vec \epsilon^T \epsilon = (Y - X \vec \beta)^T (Y - X \vec \beta)
$$

从而，我们的目标是求出$\hat{\vec{\beta}}$，使得：

$$
\frac{\partial (Y - X \vec \beta)^T (Y - X \vec \beta)}{\partial \vec{\beta}} = 0
$$

> 这里要用到的线性代数求导法则：$\frac{\partial \alpha^T \beta}{\partial \beta} = \alpha$, $\frac{\partial \beta^T \alpha}{\partial \beta} = \alpha$, $\frac{\partial (\beta^T \alpha \beta)}{\partial \beta} = 2 \alpha \beta$。这里的$(\alpha, \beta)$都是向量。

对上式左侧展开：

$$
\begin{align}
    \frac{\partial (Y - X \vec \beta)^T (Y - X \vec \beta)}{\partial \vec \beta} & = \frac{\partial Y^T Y}{\partial \beta} - \frac{\partial Y^T X\beta}{\partial \beta} - \frac{\partial \beta^T X^T Y}{\partial \beta} + \frac{\partial \beta^T X^T X\beta}{\partial \beta}\\
    &= 0-X^T Y - X^T Y + 2X^TX \hat \beta
\end{align}
$$

从而有:

$$
\hat \beta = (X^T X)^{-1} X^T Y
$$

得到$\hat \beta$后，将其带回$y = x\beta + \epsilon$，即可求得$\hat y = x \hat \beta$，$\hat y$被称作预测值，而预测值与真实值的差$\hat \epsilon = y - \hat y$被称作残差(residuals)。

/// details | $X^T X$的性质
    type: note
    open: True

$X^T X$具有一些性质：

1. $X^T X$是对称矩阵： $(X^TX)^T = X^TX$；

2. $X^T X$是正定矩阵（特征值均不小于0），对于任何$k$阶向量$c$，有$c^T (X^T X)c \geq 0$，而且如果$\vec c \not = \vec 0$，则前面的二次型必然大于0（即严格正定）；

3. $X^T X$是一个$k\times k$的方阵。
///

/// details | OLS的特殊情形
    type: note

(1) 过原点回归，即截距项为0。为简单，这里设$k = 1$，有$y_i = \beta_i x_{1i} + \epsilon_i$，从而有$X = [x_{11},...,x_{1N}]^T$，$Y = [y_1,...,y_N]^T$，那么就可以得出:

$$
\hat \beta = (X^TX)^{-1}X^TY = \frac{\sum_{i=1}^N x_{1i}Y_i}{\sum_{i=1}^N x_{1i}^2}
$$

进一步，如果$x_{1i}$ = 1，则有$y_i = \beta_i + \epsilon_i$，此时$\beta = \bar y$。

(2) k = 2，且有截距项。则$y_i = \beta_1 + \beta_2 x_{2i} + \epsilon_i$，代入OLS系数表达式，就有著名的线性规划系数方程：

$$
\begin{align}
\hat \beta_2 &= \frac{\sum_{i=1}^N (x_{2i} - \bar x_2) (y_i - \bar y)}{\sum_{i=1}^N (x_{2i} - \bar x_2)^2}\\
\hat \beta_1 &= \bar y - \hat \beta_2 \bar x_2
\end{align}
$$
///

将视线转向残差，可以得到残差的几项性质：

**性质1**： $X^T \hat \epsilon = 0$
/// details | 残差性质1证明
    type: success

$$
\begin{align}
X^T \hat \epsilon &= X^T(Y - \hat Y) = X^T Y - X^T \hat Y\\
&=X^T Y - X^T X \hat \beta = X^T Y - (X^TX) (X^TX)^{-1} X^TY\\
&= X^TY - X^TY = 0
\end{align}
$$
///

由于不存在完美的多重共线性，所有的$X$中各个向量在多维空间中形成了一个超平面，而原始的$Y$亦是多维空间中的一个向量，二者并不一定共面，$Y$在超平面上的投影即为$\hat Y$，而剔除掉这个投影向量后，剩下的那部分（即残差）是垂直于超平面的，这也就是性质1的几何解释。而这个投影对应的投影矩阵(Projection Matrix)为$P = X(X^TX)^{-1}X^T$，这是一个$N \times N$的方阵。而再引入一个Annihilator Matrix $M = I - P$，其中I是N阶单位阵。
> $PY = \hat Y$，$\hat Y$即为$Y$的投影，而P即为投影矩阵。
>
>"Annihilator"在数学中有三种译法，第一种叫“零化子”，常用于环论和泛函分析；第二种译作“消灭矩阵”，即此处所指Annihilator Matrix，用于回归分析；第三种用于解决非齐次常微分方程的“吸纳法”（Annihilator Method）。

/// details | 投影矩阵和Annihilator矩阵的性质
    type: note
    open: True

（1）$P$和$M$都是方阵和对称矩阵。
    
（2）$P$和$M$都是幂等矩阵(idempotent matrix)。 即$P^2 = P$,$M^2 = M$，不难证明。
    
（3）$PX = X$, $MX = 0$，不难证明。
///

## OLS的性质及假设检验
在有限样本下，OLS具有三个性质：无偏性(unbiased)、有效性(efficient)、一致性(consistency)。我们首先推导无偏性，而其他两个性质需要更强的假设，因此我们需要先讨论一些数学结论、补充假设。此外，这一节还会介绍一些假设检验的内容，和假设检验对应的置信区间等性质，然后讨论有效性。一致性属于大样本范畴下的性质，将在其他单元讨论。

### 3.1 无偏性
无偏性即$E[\hat \beta|X] = \beta$，即基于样本估计出来的参数估计值是无偏的。仅依据基本假设1-3，即可证明：
/// details | 无偏性的证明
    type: success

$$
\begin{align}
    \hat \beta &= (X^TX)^{-1}X^TY = (X^TX)^{-1}X^T(X\beta + \epsilon)\\
    &= \beta + (X^TX)^{-1}X^T\epsilon\\
    E(\hat \beta|X) &= E(\beta|X) + E[(X^TX)^{-1}X^T\epsilon|X]\\
    &=\beta + (X^TX)^{-1}X^T E(\epsilon|X) = \beta
\end{align}
$$
///

而在讨论有效性和一致性时，需要考虑扰动项的分布问题，需要补充两个额外的假设：条件同方差假设、正态分布假设。

### 3.2 条件同方差假设

条件同方差假设(conditional homoscedasticity)也可以被称作spherical error variance（球形方差），由于在直角坐标系中每个变量的方差图象都是一个圆。其定义为：

**假设4（条件同方差）**：$E[\epsilon_i^2|X] = \sigma^2$，而对于交叉项，有$E[\epsilon_i \epsilon_j|X] = 0, i \not = j$。

现在先讨论一下向量的方差、协方差问题。如果$y$是一个列向量，那么默认$Var(y) = E\{[y - E(y)][y - E(y)]^T\}$，这是一个$n \times n$的矩阵，如果写开，就是：

$$
\begin{equation}
    Var(y) = \left[\begin{array}{ccc}
        E[(y_1 - E(y_1))^2] & E[(y_1 - E(y_1)) (y_2 - E(y_2)] &... \\
        E[(y_2 - E(y_2) (y_1 - E(y_1))] & E[(y_2 - E(y_2))^2] & ...\\
        ... & ... & ...
    \end{array} \right] = \left[\begin{array}{ccc}
        Var(y_1) & Cov(y_1,y_2)&... \\
        Cov(y_2,y_1) & Var(y_2) & ...\\
        ...&...&...
    \end{array}\right]
\end{equation}
$$

即向量$y$的方差协方差矩阵。对于向量$Ay$($A$是一个矩阵)的方差，则有：

$$
Var(Ay) = E[(Ay - E(Ay))(Ay  -E(Ay)^T] = AVar(y)A^T
$$

如果$w,y$是两个向量，那么有：

$$
Cov(w,y) = E[(w - E(w)) (y - E(y))^T]
$$

但要注意的是$Cov(y,w) \not = Cov(w,y)$，如果说是两个随机变量的协方差，反过来是一样的。但是这里是两个由若干随机变量组成的向量，其维度可能不同，计算时要转置的矩阵不一样，因而不能调换位置。而对于矩阵$A,B$，有$Cov(Aw, By) = A Cov(w,y) B^T$。

那么由以上四个基本假设，可以求出$Var(\hat \beta| X) = \sigma^2 (X^TX)^{-1}$，下面给出证明：

/// details | OLS估计量条件方差性质的证明
    type: success

$$
\hat \beta = (X^TX)^{-1} X^TY = \beta + (X^TX)^{-1}X^T\epsilon
$$

令$A = (X^TX)^{-1}X^T$，则有：

$$
\begin{align}
    Var(\hat \beta|X) &= Var[(X^TX)^{-1}X^T\epsilon|X] = Var(A\epsilon|X) = AVar(\epsilon|X)A^T\\
    \text{由假设4，}Var(\epsilon|X) &= \sigma^2\\
    \text{因而有}Var(A\epsilon|X) &= \sigma^2 AA^T = \sigma^2(X^TX)^{-1}
\end{align}
$$

通过假设4，定义一个新的参数$\sigma^2$，即误差项方差，而有其估计量：

$$
\hat \sigma^2 = \frac{1}{N-k} \sum_{i=1}^N \hat \epsilon_i^2
$$

系数采用$N-k$，使得这个估计值是无偏的，同时$N-k$也被称作自由度。由假设1-4，可以证明这是无偏的估计(这里会用到前面的两个投影矩阵$P$和$M$)：

$$
\begin{align}
    \sum_{i=1}^N \hat \epsilon_i^2 &= \hat \epsilon ^T \hat \epsilon = (MY)^T(MY)\\
    &=[M(X\beta + \epsilon)]^T [M(X\beta + \epsilon)] = \epsilon^T M^T M \epsilon\\
    &= \epsilon^T M \epsilon \text{（这里用到了M的幂等性和对称性）}
\end{align}
$$

因而，有:

$$
\begin{align}
    E[\sum_{i=1}^N \hat \epsilon_i^2|X] &= E[\epsilon^T M \epsilon|X] = E[\sum_{i=1}^N \sum_{j=1}^N m_{ij} \epsilon_i \epsilon_j|X]\\
    \text{由假设4}&=E[\sum_{i=1}^N m_{ii} \sigma^2|X] = (\sum_{i = 1}^N m_{ii}) \sigma^2 = tr(M) \sigma^2 \\
    &= tr(I_N - P) \sigma^2 = [tr(I_N) - tr(P)]\sigma^2 = (N - tr(P))\sigma^2
\end{align}
$$

其中$tr(\cdot)$表示矩阵的迹（trace）运算，矩阵的迹有以下运算：$tr(AB) = tr(BA)$，而$P = X(X^TX)^{-1}X^T$，从而有：

$$
tr(P) = tr(X(X^TX)^{-1}X^T) = tr(X^TX (X^TX)^{-1}) = tr(I_k) = k
$$

从而有$E[\sum_{i=1}^N \hat \epsilon_i^2|X] = (N-k)\sigma^2$，或者$E[\hat \sigma^2|X] = \sigma^2$，得证。同时，这里就是自由度$N-k$的来源。
///

### 3.3 Gauss-Markov定理
这里其实是BLUE的来源。
> BLUE即best linear unbiased estimator，最优线性无偏估计量。

首先，OLS将系数向量估计为$\hat \beta = (X^TX)^{-1}X^TY$，如果将$A = (X^TX)^{-1}X^T$，则$\hat \beta = AY$，因而$\hat \beta$是线性的估计量；其次，而如果有$E(\hat \beta) = \beta$，从而达到了无偏；最后，而如果对于任意的估计量$\tilde \beta = Cy$，有$Var(\tilde \beta|X) \geq Var(\hat \beta|X)$，则$\hat \beta$就是最优的，因而$\hat \beta$即为BLUE。

而Gauss-Markov定理的内容就是，*基于假设1-4，可以证明OLS估计量是BLUE。*下面给出证明：

/// details | Gauss-Markov定理的证明
    type: success

(上面那段话的前两部分，已经证明了其线性性和无偏性，接下来只需要证明其“best”，即具有最小的条件方差。)

设$D \equiv C-A$，则$\tilde \beta = (D+A)Y = DY + \hat \beta$，进一步有：

$$
\begin{align}
    \tilde \beta &= (D+A)Y = DY + \hat \beta = DX\beta + D\epsilon + \hat \beta\\
    \beta &= E(\tilde \beta|X) = DX\beta + DE(\epsilon|X) + E(\hat \beta|X)= DX\beta + \beta\\
    \text{因而有：}DX &= 0
\end{align}
$$

因而有$\tilde \beta = D\epsilon + \hat \beta$，进而有$\tilde \beta -\beta = D\epsilon + \hat \beta - \beta = (D+A)\epsilon$，故有：

$$
\begin{align}
    Var(\tilde \beta|X) &= Var(\tilde \beta - \beta|X) = Var[(D+A)\epsilon|X] = (D+A) \sigma^2 (D+A)^T \\
    &= \sigma^2 (D+A)(D+A)^T = \sigma^2 (DD^T + AD^T + DA^T + AA^T)
\end{align}
$$

其中$DA^T = DX(X^TX)^{-1} = 0$（因为$DX = 0$），同理$AD^T = 0$，从而有：

$$
Var(\tilde \beta|X) = \sigma^2 (DD^T + AD^T + DA^T + AA^T) = \sigma^2(DD^T + AA^T) = Var(\hat \beta|X) + \sigma^2 DD^T
$$

而由于$DD^T$是一个半正定矩阵，因而$Var(\tilde \beta|X) \geq Var(\hat \beta|X)$，故$\hat \beta$具有最小的条件方差，是最优的估计量，因而Gauss-Markov定理得证。

($DD^T$这种形式的矩阵均为半正定矩阵，如果不能一眼看出，代入一个行向量$c$，按半正定矩阵的条件也可以证明。)
///

### 3.4 正态分布假设
这个假设的内容是$f(\epsilon|X) \sim MN(0,\sigma^2 I_N)$。

>多元正态分布，均值为0向量，且服从仅有方差且方差相等的一个协方差矩阵。

从一元正态分布说起，如果$x \sim N(\mu, \sigma^2)$，则其概率密度函数(PDF)为：

$$
f(x) = \frac{1}{\sqrt{2\pi}\sigma}exp(-\frac{(x-\mu)^2}{2\sigma^2})
$$

对应的，对于多元正态分布$X \sim MN(\mu, \Sigma)$，则其PDF为：

$$
f(X) = \frac{1}{(2\pi)^{N/2} \sqrt{det(\Sigma)}} exp[-\frac{1}{2} (X - \mu)^T \Sigma^{-1} (X-\mu)]
$$

接下来讨论多元正态分布的性质，设$X$为$N$维列向量且$X \sim MN(\mu, \Sigma)$，将其分为$X = [X_1, X_2]^T$，其中$X_1,X_2$分别为$N_1,N_2$维列向量，且有$N_1 + N_2 = N$。对应的，将$\mu = [\mu_1, \mu_2]^T$，二者分别为$N_1$和$N_2$维，而$\Sigma$应分为：

$$
    \Sigma = \left[\begin{array}{cc}
        \Sigma_{11} & \Sigma_{12} \\
        \Sigma_{21} & \Sigma_{22} 
    \end{array} \right]
$$

左、上的维度数为$N_1$，右、下为$N_2$。由以上内容，可以推出：

1. $X_1$的边际密度函数$f(X_1) = MN(\mu_1, \Sigma_{11})$；

2. 条件密度函数：$f(X_1|X_2) = MN(\mu_{1|2}, \Sigma_{1|2})$。其中$\mu_{1|2} = \mu_1 + \Sigma_{12}\Sigma_{22}^{-1}(X_2 - \mu_2)$；$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12} \Sigma_{22}^{-1}\sigma_{12}^T$；

3. 如果$[X_1,X_2]^T \sim MN$，则$X_1, X_2$均服从MN，但**逆命题不成立**，这个逆命题被称作Copula问题；

4. 如果$X,Y \sim MN$，则$cov(X,Y) = 0$与“$X$、$Y$互相独立”可以相互推出。对于其他大多数分布来说，独立性可以推出协方差为0，而不能反推。 

由于假设5，且$Y = X\beta + \epsilon$，因而有:

$$
f(Y|X) = MN(X\beta, \sigma^2 I_N)
$$

进一步，由$\hat \beta = (X^TX)^{-1}X^TY = \beta + (X^TX)^{-1}X^T\epsilon$，从而可以推导$\hat \beta$的概率分布：

$$
\begin{align}
    f(\hat \beta|X) &= MN(\beta, (X^TX)^{-1}X^TVar(\epsilon|X) X(X^TX)^{-1}\\
    &= MN(\beta, \sigma^2 (X^TX)^{-1})
\end{align}
$$

由此，我们已经设定了截面数据计量经济学下的所有基本假设。

### 3.5 假设检验1：单一变量显著性检验（T检验）
第一种假设检验是对单一参数的检验，其原假设为$H_0$：$\beta_j = \bar \beta_j$，其中$\bar \beta_j$为任意给定的值。那么由多元正态分布的性质，有：

$$
f(\hat \beta_j|X) \sim N(\beta_j, \sigma^2 (X^TX)^{-1}_{jj})
$$

鉴于这个统计量是一个正态统计量，我们可以采用Z-score方式，即将上面的$\hat \beta_j$标准化：

$$
z_j = \frac{\hat \beta_j - \bar \beta_j}{\sqrt{\sigma^2 (X^TX)^{-1}_{jj}}}
$$

为了检验原假设$H_0$，我们可以选择备择假设,$H_1:\beta_{j} \not = \bar \beta_j$，或者定为$H_1: \beta_j > \bar \beta_j$ or $\beta_j  < \bar \beta_j$。双边的备择假设更为常见，而单边的备择假设选择方向时通常考虑经济学意义。

回到统计量本身，$Z_j$服从标准正态分布，那么对于**双边**备择假设来说，其极端值（无论左右）都表明$H_0$为假。那么，就要确定一个认为$H_0$为假（拒绝原假设）的区间，是一个多大的极端值，从而引出了置信区间(significance level) $\alpha$。$\alpha$通常取值为1\%,5\%,10\%，如果$|Z_j| > Z_{\frac{\alpha}{2}}$(此处指的是Z统计量分布的$\alpha/2$分位数)，则拒绝原假设(reject $H_0$)，否则不拒绝（而不是"接受"！）原假设

> 对应的，如果是单边备择假设情形，则分情况：（1）若$H_1$为"$\beta_j > \bar \beta_j$"，则当$Z_j > Z_\alpha$时，拒绝原假设；（2）反之则在$Z_j < Z_\alpha$时拒绝原假设。假设检验具体是单边还是双边是要看要检验的假设本身的，而并不是由分布的性质来决定。

进一步，因为实际上的分布情况不明，我们只能通过估计出来的$\hat \sigma^2$来对应。所以，对于估计值$\hat \beta$，其对应的统计量我们称为$t_j$，表示为：

$$
t_j = \frac{\hat \beta_j - \bar \beta_j}{\sqrt{\hat \sigma^2 (X^TX)^{-1}_{jj}}}
$$

与$z_j$相比，唯一的变化在于分母上标准差为估计值。$t_j$服从于学生T分布，其自由度为(N-K)，即$t_j \sim t_{N-k}$。

>学生T分布的形式是$t = \frac{MN(0,I_N)}{\sqrt{\chi^2(N-K)/ (N-K)}}$。分子为多元标准正态分布，分母主体是自由度为$N-K$的卡方统计量$\chi^2(N-K)$。

下面给出证明：
/// details | 证明t统计量服从学生t分布
    type: note

$$
\begin{align}
    t_j = \frac{\hat \beta_j - \bar \beta_j}{\sqrt{\hat \sigma^2 (X^TX)^{-1}_{jj}}} = \frac{\hat \beta_j - \bar \beta_j}{\sqrt{\sigma^2 (X^TX)^{-1}_{jj}}} \sqrt{\frac{\sigma^2}{\hat \sigma^2}} = z_j \sqrt{\frac{\sigma^2}{\hat \sigma^2}} 
\end{align}
$$

由上文知$z_j \sim N(0,1)$，接下来证明$\hat \sigma^2 / \sigma^2 \sim \frac{X_{N-K}^2}{N-K}$，首先考察$\hat \sigma^2$的性质：

$$
\hat \sigma^2 = \frac{\hat \epsilon^T \hat \epsilon}{N-K} = \frac{\epsilon^T M \epsilon}{N-K}
$$

所以有:

$$
\frac{\hat \sigma_2}{ \sigma^2} = (\frac{\epsilon}{\sigma})^T M (\frac{\epsilon}{\sigma})/(N-K)
$$

而$\epsilon \sim N(0,\sigma^2 I_N)$，因而$\frac{\epsilon}{\sigma} \sim N(0,I_N)$，故有:

$$
\frac{\hat \sigma_2}{ \sigma^2} = N(0,I_N)^T M N(0,I_N) / (N-K) \sim \chi^2_{N-K}/ (N-K)
$$

(这里运用的定理：如果$X\sim N(0,I_N)$且A为幂等矩阵，则$X^TAX \sim \chi^2_{Rank(A)}$。此外，如果$A$为幂等矩阵，则Rank(A) = Trace(A)。在此之前，我们已经证出了$trace(M) = N-K$。)
///

由此，我们就证出了$t_j$服从学生t分布，接下来证明$t_j$的分子与分母相互独立（这是学生t分布的一个性质）。这个统计量的分子是$\hat \beta_j - \bar \beta_j$，其中我们知道$\hat \beta = \beta + (X^TX)^{-1}X^T\epsilon = \beta + A\epsilon$；而分母是一个$\hat \epsilon$的函数，而$\hat \epsilon = M \epsilon$，且$\epsilon \sim N(0,\sigma^2 I_N)$。因此有：

$$
\begin{align}
    \left[\begin{array}{c}
        \hat \beta - \beta  \\
         \hat \epsilon 
    \end{array} \right] = \left[\begin{array}{c}
         A  \\
         M 
    \end{array}\right] \epsilon = N(0,\sigma^2\left[\begin{array}{cc}
        AA^T & AM^T  \\
        MA^T & MM^T
    \end{array}\right])
\end{align}
$$

其中:
> $AA^T$可由其表达式直接推出，$MM^T = M$是因为$M$是幂等矩阵和对称矩阵，而$AM^T$与$MA^T$则是代入$A$和$M$的表达式解出，其中$M = I_N - X(X^TX)^{-1}X^T$， $A = (X^TX)^{-1}X^T$。

$$
\begin{align}
    AA^T &= (X^TX)^{-1}\\
    MM^T &= M\\
    AM^T &= MA^T = 0
\end{align}
$$

故有：

$$
    \left[\begin{array}{c}
        \hat \beta - \beta  \\
         \hat \epsilon 
    \end{array} \right] = N(0,\left[\begin{array}{cc}
        Var(\hat \beta|X) & 0  \\
        0 & \sigma_2 M
    \end{array}\right])
$$

所以$Cov(\hat \beta, \hat \epsilon) = 0_{K\times N}$，而由上文性质4可知，$\hat \beta$和$\hat \epsilon$互相独立，那么二者的函数亦相互独立，故统计量的分子和分母是互相独立的。


### 3.6 假设检验2：线性关系的假设检验（F分布）

这个假设检验的原假设$H_0$：$R \times \beta = \gamma$。其中$R$是$r \times k$的矩阵，$\beta$是$k \times 1$的向量，$\gamma$是$r \times 1$的向量。这个原假设可以被看作是规定了线性规划问题中的约束，即各系数$\beta$的线性组合约束。这里假设$Rank(R) = r$，使得这里没有多余的线性约束（也就是没有多余的原假设），这隐含了$r \leq k$的条件。此外要注意的是，$R$与$\gamma$中是常值，而$\beta$则包含了所有的未知数（待估参数）。接下来以Cobb-Douglas函数为例，其基本形式是：

$$
Y = AK^{\beta_2}L^{\beta_3}e^\epsilon
$$

取对数：

$$
\ln Y_i = \ln A + \beta_2 \ln K + \beta_3 \ln L + \epsilon_i
$$

其中我们可以令$\beta_1 = \ln A$，则我们可以检验一下$\beta_1 = \bar \beta_1$（全要素生产率是否不以个体为转移），或者$\beta_2 + \beta_3 = 1$（规模效应不变），我们可以同时做检验，将二者同时作为$H_0$。那么有：

$$
\beta = \left[\begin{array}{c}
        \beta_1\\
        \beta_2\\
        \beta_3
    \end{array} \right], R = \left[\begin{array}{ccc}
        1 & 0 & 0 \\
        0 & 1 & 1
    \end{array}\right], \gamma = \left[\begin{array}{c}
        \bar \beta_1 \\
        1
    \end{array}\right]
$$

我们不知道$\beta$的真值，但是我们可以从$R\beta - \gamma$上下手。我们知道$\hat \beta \sim N(\beta, \sigma^2 (X^TX)^{-1})$，那么$\hat \beta - \beta \sim N(0,\sigma^2 (X^TX)^{-1})$，然后同乘以R：

$$
R\hat \beta - \gamma = R(\hat \beta - \beta)  \sim N(0,\sigma^2 R(X^TX)^{-1}R^T)
$$

如果原假设成立，那么左边的等式是成立的。所以检验原假设就是要检验$R\hat \beta - \gamma$是否服从那个上式的分布，我们可以算一下这玩意的平方和：

$$
(R\hat \beta - \gamma)^T [\sigma^2 R(X^TX)^{-1}R^T]^{-1} (R\hat \beta - \gamma) \sim \chi_r^2 
$$

上式的结果是一个标量，而这个标量服从自由度为r的卡方分布。

/// details | 证明上式服从卡方分布
    type: success

接下来证明这是一个卡方分布，这里会用到Cholesky分解法：
> Cholesky 分解在Matlab中使用函数`chol()`。

如果矩阵$A$是正定的对称阵，则$A = (A^{\frac{1}{2}})(A^{\frac{1}{2}})^T$，而$A^{\frac{1}{2}}$是一个下三角矩阵。对于这样的矩阵，其逆矩阵为：$A^{-1} = ((A^{\frac{1}{2}})^T)^{-1}(A^{\frac{1}{2}})^{-1}$，我们令$(A^{\frac{1}{2}})^{-1} = A^{-\frac{1}{2}}$。
///

接下来讨论$R\hat \beta - \gamma$的分布，我们令$A = \sigma^2 R(X^TX)^{-1}R^T$，对其作Cholesky分解，则有：

$$
R\hat \beta - \gamma \sim A^{\frac{1}{2}} N(0,I_r)
$$

将Cholesky分解结果和基本性质带进平方和（那个被证明服从卡方分布的标量），有：

$$
\begin{align}
    (R\hat \beta - \gamma)^T [\sigma^2 R(X^TX)^{-1}R^T]^{-1} (R\hat \beta - \gamma) &= [A^{\frac{1}{2}} N(0,I_r)]^T A^{-1} [A^{\frac{1}{2}} N(0,I_r)]\\
    &= N(0,I_r)^T (A^{\frac{1}{2}})^T (A^{-\frac{1}{2}})^T (A^{-\frac{1}{2}}) A^{\frac{1}{2}} N(0,I_r)\\
    &= N(0,I_r)^T N(0,I_r) \sim \chi_r^2
\end{align}
$$

毕竟卡方分布的实质是正态分布的平方和。但是，这个统计量的缺陷在于，我们需要知道$\sigma^2$的真值，但这不可能。所以下面要用$\hat \sigma^2$代替$\sigma$，从而引入F统计量：

$$
   F \equiv (R\hat \beta - \gamma)^T [\hat \sigma^2 R(X^TX)^{-1}R^T]^{-1} (R\hat \beta - \gamma) / r
$$

当然，这个和中级计量经济学下的F统计量等价：

$$
F_R = \frac{(SSR_R - SSR_U)/r}{SSR_U/(N-K)}
$$

其中$SSR_R$是将原假设作为约束时进行OLS，求得的残差平方和；而$SSR_U$则是不带原假设约束的OLS残差平方和。

在满足假设1-5的情况下，如果原假设成立，则$F\sim F_{r,N-k}$（F统计量服从自由度为r和N-k的F分布）。由于我们的原假设是$R \times \beta = \gamma$，因而备择假设为$H1: R \times \beta \not = \gamma$，从而使得这个假设检验是双边的。但F统计量的本质是估计量（$\hat \beta$）函数的平方项，因而无论是$R\beta > \gamma$还是$R\beta < \gamma$，F统计量都是异常提高，所以说，无论单双边检验，F分布都只看右侧的极端值，也就是：如果$F > F_{r,N-k,\alpha}$，则在$\alpha$置信度的情况下**拒绝**原假设（无论单边/双边检验）。

//// details | 证明：F统计量的两个形态等价
    type: success

（这是我们当初的习题）

/// admonition | 问题要求
    type: question

只用一次无约束OLS求出的F统计量如下式所示：

$$
F \equiv (R\hat \beta - \gamma)^T [\hat \sigma^2 R(X^TX)^{-1}R^T]^{-1} (R\hat \beta - \gamma) / r
$$

而之前中级计量会采用以$H0$为约束的OLS计算F统计量，即：

$$
F_R = \frac{(SSR_R - SSR_U)/r}{SSR_U/(N-K)} \tag{A.2}
$$

而二者可以被证明是等价的，请做出证明。
///

答案可参阅：[StackExchange Math的这个问题](https://math.stackexchange.com/questions/3868127/prove-these-two-f-stats-are-equivalent)，我的自问自答。

**一、对分子的推导**。首先，对于没有任何约束的OLS估计来说，其SSR为：

$$
\begin{align}
    SSR_U &= (Y - X\hat \beta)^T (Y - X\hat \beta) = [Y - X(X^TX)^{-1} X^TY]^T [Y - X(X^TX)^{-1} X^TY]\\
    &= (MY)^T (MY) = [M(X\beta + \epsilon)]^T[M(X\beta + \epsilon)] = \epsilon^T M^TM\epsilon = \epsilon^T M \epsilon \tag{A.4}
\end{align}
$$

接下来讨论带有约束的OLS估计，及其残差平方和。设此问题的估计值为$\bar \beta$，这个问题本质上是一个线性规划：

$$
\begin{align}
    \min_{\beta}\quad& (Y - X\beta)^T (Y - X\beta)\\
    s.t. \quad& R\beta = \gamma
\end{align}
$$

利用Lagrange乘子法，则有Lagrange函数为：

$$
L = (Y - X\beta)^T (Y - X\beta) + \lambda^T (R\beta - \gamma) 
$$

其中$\lambda$为一个$r\times 1$的列向量。这个问题的FOC有两项：

$$
\begin{align}
    \textbf{(1)}\frac{\partial L}{\partial \beta} &= -2X^TY + 2X^TX\bar \beta + R^T \lambda = 0\tag{A.8}\\
    \bar \beta &= (X^TX)^{-1}[-\frac{1}{2}R^T\lambda + X^TY]\\
    \textbf{(2)} \frac{\partial L}{\partial \lambda} &= R\bar\beta - \gamma = 0\\
    \gamma &= R\beta \tag{A.11}
\end{align}
$$

把A.8和A.11联合在一起，写成矩阵形式，有：

$$
\begin{align}
    \left[\begin{array}{cc}
        2X^TX & R^T  \\
        R & 0
    \end{array}\right] \left[\begin{array}{c}
        \bar\beta  \\
        \lambda^T 
    \end{array}\right] &= \left[\begin{array}{c}
        2X^TY  \\
        \gamma 
    \end{array}\right]\\
     \left[\begin{array}{c}
        \bar\beta  \\
        \lambda^T 
    \end{array}\right] &= \left[\begin{array}{cc}
        2X^TX & R^T  \\
        R & 0
    \end{array}\right]^{-1}\left[\begin{array}{c}
        2X^TY  \\
        \gamma 
    \end{array}\right]  
\end{align}
$$

从而解得：

$$
\bar \beta = \hat \beta - (X^TX)^{-1}W(R\hat\beta - \gamma)
$$

其中$W = R^T[R(X^TX)^{-1}R^T]^{-1}$。那么，带约束的OLS的SSR为：

$$
\begin{align}
    SSR_R &= (Y - X\bar \beta)^T (Y - X\bar \beta) \tag{A.15}\\
    Y-X\bar\beta &= MY + X(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}[R(X^TX)^{-1}X^TY - \gamma]
\end{align}
$$

若令$S = X(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^T$,则:

$$
Y - X\bar\beta = (M+S)Y - X(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}\gamma \tag{A.17}
$$

易证矩阵$S$的幂等性。然后可证$(M+S)$是幂等和对称的：

$$
\begin{align}
    &(M+S)(M+S) = M + S + SM + MS\\
    &MS = [I - X(X^TX)^{-1} X^T][X(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^T] = S - S = 0\\
    &SM = [X(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^T][I - X(X^TX)^{-1} X^T] = S - S = 0\\
    &(M+S)(M+S) = M + S + 0 + 0 = M + S
\end{align}
$$

那么，将式A.17代入式A.15，有：

$$
\begin{align}
    SSR_R &= Y^T(M+S)Y - Y^T(M+S)^TX(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}\gamma \tag{A.22}\\
    &-\gamma^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^T(M+S)Y + \gamma^T[R(X^TX)^{-1}R^T]^{-1}\gamma \tag{A.23}
\end{align}
$$

对于上式第三项，有：

$$
\begin{align}
    &\gamma^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^TMY = \gamma^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^T[I-X(X^TX)^{-1}X^T]Y = 0 \tag{A.24}\\
    &\gamma^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^TSY\\
    &= \gamma^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^TX(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^TY\\
    &= \gamma^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^TY \tag{A.27}
\end{align}
$$

对于上式第二项，有:

$$
\begin{align}
    &Y^TM^TX(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}\gamma = 0\tag{A.28}\\
    &Y^TS^TX(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}\gamma = Y^T X(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}\gamma \tag{A.29}
\end{align}
$$

由以上的整理，将A.24,A.27,A.28,A.29带回A.22/23，有：

$$
\begin{align}
    SSR_R - SSR_U &= Y^TSY - \gamma^T[R(X^TX)^{-1}R^T]^{-1}R(X^TX)^{-1}X^TY \\
    &- Y^T X(X^TX)^{-1}R^T[R(X^TX)^{-1}R^T]^{-1}\gamma + \gamma^T[R(X^TX)^{-1}R^T]^{-1}\gamma\\
    &= Y^TSY - \gamma^T[R(X^TX)^{-1}R^T]^{-1}R\hat \beta - \hat\beta^TR^T [R(X^TX)^{-1}R^T]^{-1} \gamma + \gamma^T[R(X^TX)^{-1}R^T]^{-1}\gamma \tag{A.32}
\end{align}
$$

其中由A.4知，$SSR_U = Y^TMY$，而$\hat \beta = (X^TX)^{-1}X^TY$。其中第一项为：

$$
\begin{align}
    Y^TSY &= \hat \beta^T R^T[R(X^TX)^{-1}R^T]^{-1}R\hat\beta
\end{align}
$$

从而A.32可变形为：

$$
SSR_R - SSR_U = (R\hat \beta - \gamma)^T [R(X^TX)^{-1}R^T]^{-1} (R\hat \beta - \gamma)\tag{A.34}
$$

**二、对分母的推导**。再来看A.2的分母$SSR_U/(N-K)$，参照$\hat \sigma^2 = \frac{\epsilon^T M \epsilon}{N-K}$，有：

$$
\frac{SSR_U}{N-K} = \frac{\epsilon^T M\epsilon}{N-K} = \frac{\hat\epsilon^T \hat \epsilon}{N-K} = \hat \sigma^2 \tag{A.35}
$$

将A.34/35带回A.2，有：

$$
\begin{align}
    F_R &= (R\hat \beta - \gamma)^T [R(X^TX)^{-1}R^T]^{-1} (R\hat \beta - \gamma) /(r \hat \sigma^2)\\
    &=(R\hat \beta - \gamma)^T [\sigma^2 R(X^TX)^{-1}R^T]^{-1} (R\hat \beta - \gamma) /r = F
\end{align}
$$

因而$F$与$F_R$等价，原问题得证。
////

### 3.7 假设检验3：非线性关系的假设检验
这种非线性关系的假设检验是非常宽泛的定义。例如检验$\hat \beta_j^2 + \hat \beta_i^2 = 1$这种非线性关系，在接下来讨论大样本情形时再讨论，而在小样本下，这种假设检验难以进行。非线性假设检验中包含了线性关系假设检验，而线性关系假设检验又包含了单一变量的显著性检验。

### 3.8 置信区间和显著性水平
置信区间(confidence interval)的定义是：*指定一个置信度$\alpha$，那么如果$\bar \beta_j$（真值）在置信区间中，我们就不能拒绝原假设*。
>有一种错误的说法，说“真值出现在置信区间中的概率为$1-\alpha$”，错误的原因是真值$\bar\beta_j$是一个客观存在但看不到的值，而不是随机变量（这是贝叶斯学派的解释，但这里是频率学派的内容，也不涉及先验后验问题）。人们通过一次次的观测来估计$\beta_j$，得到了一个个估计值$\bar\beta_j$，从而得到了一个个不同的置信区间，而这些置信区间中包含了真值的概率是$1-\alpha$（这是频率学派的解释）。换句话说，**置信区间是可变的，而真值是不变的**。

以t统计量为例（其实也可以用在Z统计量和F统计量，但是前者不知道$\sigma^2$，后者的置信区间是多维空间中的几何体/面，都比较复杂），若$-t_{N-k,\alpha} \leq t_j \leq t_{N-k,\alpha}$则不能拒绝原假设，而这个条件可以写为：

$$
\begin{align}
    -t_{N-k,\alpha} &\leq \frac{\hat \beta_j - \bar \beta_j}{StdDev(\hat \beta_j)} \leq t_{N-k,\alpha}\\
    \hat \beta_j - t_{N-k,\alpha} StdDev(\hat\beta_j) &\leq \bar \beta_j \leq  \hat \beta_j + t_{N-k,\alpha} StdDev(\hat\beta_j)
\end{align}
$$

上两式中，下面那个表现出来的$\bar \beta_j$的取值范围即为置信区间。而P值(P-value)表示统计量得到极端（或者更加极端）取值的概率。
>Probability of obtaining a value as extreme or more extreme for the test statistic.

如果备择假设是双边的，则P-value取值为：$P = P(t_{N-K} > |t_j|) \times 2$，如果是单边的，就是$P = P(t_{N-K} > t_j)$（以备择$\beta_j > \bar \beta_j$为例）。

### 3.9 MLE
最大似然估计全程需要假设1-5，并且需要$\epsilon$的概率密度，在条件上要更强一些，但估计的结果通常都是有效的。

所谓的“似然函数”（likelihood）是观测值的联合概率密度：

$$
    L(\theta) \equiv f(Y|X,\theta), \theta = \left[\begin{array}{c}
        \beta \\
        \sigma^2
    \end{array}\right]
$$

$\theta$即为模型中需要估计的参数的集合。对于线性模型$Y = X\beta + \epsilon$和正态假设，有$L(\theta) = N(X\beta, \sigma^2 I_N)$，那么有：

$$
L(\theta) = [(2\pi)^{N/2} \det(\sigma^2 I_N)^{1/2}]^{-1} \exp [-\frac{1}{2} (Y- X\beta)^T [\sigma^2 I_N]^{-1} (Y-X\beta)]
$$

那么，使得$L(\theta)$最大的参数组$\bar \theta$即为MLE下的估计值，通常来说，求这个值都是用FOC（一阶条件）来搞（但不排除似然函数的形状很奇怪，从而找到局部最优），而且求FOC时通常采用对数化的似然函数$\mathcal{L}(\theta) = \ln L(\theta)$。在这个线性模型中，对数化的似然函数为：

$$
\mathcal{L}(\theta) = -\frac{N}{2} \ln(2\pi) - \frac{N}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2} (Y-X\beta)^T (Y-X\beta)
$$

求FOC：

$$
\begin{align}
    &\frac{\partial \mathcal{L}(\theta)}{\partial \beta} = 0 \leftrightarrow \frac{d (Y-X\beta)^T (Y-X\beta)}{d\beta} = 0 \leftrightarrow \hat \beta = (X^TX)^{-1}X^TY\\
    &\frac{\partial \mathcal{L}(\theta)}{\partial \sigma^2} = 0 \leftrightarrow \hat \sigma^2 = \frac{(Y-X\beta)^T (Y-X\beta)}{N}
\end{align}
$$

由此可见，MLE估计出的$\beta$和OLS相同，但是$\sigma^2$的估计与OLS不同，我们知道OLS的方差估计量是无偏的，因而MLE的方差估计值是**有偏**的。大样本情况下，MLE满足一致性，但是否满足有效性呢？

### 3.10 OLS和MLE的大样本有效性

我们定义S函数(Score function)，即$S(\theta) = \frac{\partial \mathcal{L}(\theta)}{\partial \theta}$，则引出Cramer-Rao下确界的定义：
/// admonition | 定义： Cramer-Rao下确界
    type: info

给定$\hat \theta$是$\theta$的无偏估计，且方差是有限的，那么基于一般的约束条件（DCT，参见Cramer-Rao下界的Wiki Chapter1.5），有：

$Var(\hat \theta) \geq I(\theta)^{-1}$，其中信息矩阵$I(\theta) = E[S(\theta) S^T(\theta)] = -E[\frac{\partial^2 \mathcal{L}(\theta)}{\partial \theta \partial \theta^T}]$
///

如果一个无偏估计的方差能够达到Cramer-Rao下界，则这个估计量必然是有效的。对于这个线性模型，其Score Function为：

$$
   S(\theta) = \left[\begin{array}{c}
        \frac{\partial \mathcal{L}(\theta)}{\partial \beta} \\
        \frac{\partial \mathcal{L}(\theta)}{\partial \sigma^2} 
    \end{array} \right] = \left[\begin{array}{c}
        -\frac{1}{\sigma^2} X^T (Y-X\beta)  \\
        -\frac{N}{2\sigma^2} + \frac{1}{2\sigma^4}(Y-X\beta)^T (Y-X\beta)
    \end{array}\right]
$$

则其信息矩阵为：

$$
    I(\theta) = E[S(\theta)S^T(\theta)] = \left[\begin{array}{cc}
        \frac{\partial \mathcal{L}(\theta)}{\partial \beta} \frac{\partial \mathcal{L}(\theta)}{\partial \beta^T}& \frac{\partial \mathcal{L}(\theta)}{\partial \beta}\frac{\partial \mathcal{L}(\theta)}{\partial \sigma^2} \\
        \frac{\partial \mathcal{L}(\theta)}{\partial \sigma^2}\frac{\partial \mathcal{L}(\theta)}{\partial \beta^T} & (\frac{\partial \mathcal{L}(\theta)}{\partial \sigma^2})^2
    \end{array}\right]
$$

**（1）对于左上角那一项**，有：

$$
\begin{align}
    \frac{\partial \mathcal{L}(\theta)}{\partial \beta} \frac{\partial \mathcal{L}(\theta)}{\partial \beta^T} &= \frac{1}{\sigma^4} X^T(Y-X\beta)(Y-X\beta)^T X = \frac{1}{\sigma^4}X^T\epsilon\epsilon^TX\\
    E[\frac{\partial \mathcal{L}(\theta)}{\partial \beta} \frac{\partial \mathcal{L}(\theta)}{\partial \beta^T}|X] &= \frac{1}{\sigma^4}X^T E[\epsilon\epsilon^T|X]X = \frac{X^TX}{\sigma^2}
\end{align}
$$

**（2）对于右上角和左下角项**（二者互为转置，仅看右上角项），有：

$$
\begin{align}
    \frac{\partial \mathcal{L}(\theta)}{\partial \beta}\frac{\partial \mathcal{L}(\theta)}{\partial \sigma^2} &= -\frac{1}{\sigma^2} X^T(Y-X\beta) [-\frac{N}{2\sigma^2} + \frac{1}{2\sigma^4}(Y-X\beta)^T (Y-X\beta)] \\
    &= \frac{N}{2\sigma^4}X^T\epsilon - \frac{1}{2\sigma^6} X^T \epsilon \epsilon^T \epsilon
\end{align}
$$

而$E[X^T\epsilon|X] = 0$。而第二项有：

$$
    E[X^T\epsilon \epsilon^T \epsilon|X] = X^T E\left[\begin{array}{c}
        \epsilon_1 \sum \epsilon_i^2\\
        ... \\
        \epsilon_N \sum \epsilon_i^2
    \end{array}\right]
$$

分情况讨论:

- 对于$j\not=i$时，有$E[\epsilon_j \epsilon_i^2|X] = E[\epsilon_j|X]E[\epsilon_i^2|X] = 0$（这里用到了假设2、4）；
- 而$j = i$时，$E[\epsilon_i^3|X] = 0$。

>$\int \epsilon^3 f(\epsilon) d\epsilon = 0$，因为$f(\epsilon)$为偶函数（正态分布），$\epsilon^3$为奇函数，二者相乘为奇函数，那么其积分为0。

因而第二项绝对为0，所以右上角和左下角项为0。

**（3）对于右下角项**，这是一个标量，因此我们可以把它当作一个特定的值，而不去求它了，比如说，我们管它叫something。

> $\mathcal{L}$是一个标量，而求$\sigma^2$（标量）偏导时也是个标量。

所以，信息矩阵$I(\theta)$形式为：

$$
    I(\theta) = \left[\begin{array}{cc}
        \frac{1}{\sigma^2}X^TX & 0 \\
        0 & sth
    \end{array}\right], I^{-1} = \left[\begin{array}{cc}
        \sigma^2(X^TX)^{-1} & 0 \\
        0 & (sth)^{-1}
    \end{array}\right]
$$

而$Var(\hat \beta_{MLE}) = Var(\hat \beta_{OLS}) = \sigma^2(X^TX)^{-1}$，因而二者均达到了Cramer-Rao下界，因而OLS和MLE对$\beta$的估计量都是有效的。
> 如果无偏估计量的方差严格大于CR下界，则必然不是有效估计量。

但是，对于$\sigma^2$的MLE估计量，由于其不满足无偏性，所以**不适用**于Cramer-Rao下确界法。