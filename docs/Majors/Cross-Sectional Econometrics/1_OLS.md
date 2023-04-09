# 一、线性模型和最小二乘法

这一部分是初级计量经济学的起始内容。在这一部分中，我们探讨计量经济学的基本假设，并介绍最小二乘法思想及其相关的假设检验。

## 基本假设

计量经济学的前三个基本假设分别是：线性性、严格外生性、无完美多重共线性。

### 1. 线性模型假设

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

### 2. 严格外生性假设

即$E[\epsilon_i | X] = 0$。**给定观测值$X$时，误差项的条件期望为0。**这与期望迭代法则（Law of iterated expectations，LIE)有关。

给定两个随机变量$u,v$并知其联合密度函数$f(u,v)$，求其边际密度函数，例如$u$的边际密度函数为$f(u) = \int f(u,v) dv$；而条件密度函数则需要利用贝叶斯公式：$f(u|v = a) = \frac{f(u,v = a)}{f(v = a)}$，那么就可以讨论条件期望和期望迭代法则了。

**期望迭代法则：** $E_v[E(u|v)] = E(u)$。*即对于所有的$v$取值情况下，$E(u|v)$的期望是$E(u)$。 *

> 期望迭代原则经常会用在宏观经济学的动态均衡模型中，利用T期信息集来求解之后的消费决策，如知信息集$I_t$时求解t+2期的消费决策，$E_t[C_{t+2}] = E[C_{t+2}|I_t] = E_t [ E(C_{t+2} | I_{t+1})]$，另一个用途就是接下来要讲的严格外生性。

??? success "期望迭代法则的证明"
    

$$
\begin{align}
    E[u|v] &= \int uf(u|v) du\\
    E_v[E(u|v)] &= \int (\int uf(u|v) du) f(v) dv = \int \int uf(u|v) f(v) du dv\\
    &= \int u (\int f(u,v)dv) du = \int u f(u) du = E(u)
    \end{align}
$$

接下来继续讨论严格外生性问题，由于知道了全部观测值$X$，就可以将这些常数$x_{ji}$乘进去：$E[\epsilon_i x_{ji}] = 0$，从而有$E_x [ E(\epsilon_i x_{ji} | x)] = 0$，从而有$E[\epsilon_i x_{ji}] = 0$，而如果$x_{1i} = 1$，那么就有$E[\epsilon_i] = 0$。而$cov(x_{ji}, \epsilon_i) = E[x_{ji} \epsilon_i] - E(x_{ji})E(\epsilon_i) = 0$，从而有$x_{ji}$与$\epsilon_i$正交（独立）。

有了以上两个基本假设，模型（及其参数的估计值）便有了意义，即：

$$
E[y_i|X] = \sum_{j=1}^k \beta_j x_{ji}
$$

或者说偏导数$\partial E(y_i|X)/\partial x_{ji} = \beta_j$。其经济学意义在于，*平均而言（因为式子里是期望），在其他的变量不变的情况下（偏导数的意义），如果$x_{ji}$增大了$\Delta x_{ji}$个单位，那么$y_i$将会提高$\beta_j \Delta x_{ji}$。*

### 3. 没有完美的多重共线性

假设3的严格表达应该是：*矩阵$X$（其规模为$n\times k$）的秩$rank(X) = k$*。

> 严格来讲，这句话是with probability 1，而不是绝对的。

此外，（1）这句话还暗含了$n \geq k$，即观测点数应大于等于变量数的原则（否则会有无数组解）；（2）观测值的任意一列**不能被其他列线性表出**（各变量不能具有完全的多重共线性，否则模型无法估计这些变量的参数，即变量“不可被识别”。）。

从线性代数的角度，如果存在完美的多重共线性，则$rank(X) \not = k$，因而有$rank(X^T X) < k$，$X^T X$这个正方形矩阵不可逆了，从而无法估计参数。

> 参照$rank(X^T X) = rank(XX^T) = rank(X)$

此外，受限于计算机浮点数精度的限制，可能$X^T X$的行列式趋近于0，导致精度不够，或者其逆矩阵的行列式趋近于无穷，出现无法估计的情况。当这个行列式趋近于0时，其实就表明这个模型中存在比较严重的多重共线性了（如果等于零，则必然是存在完美的多重共线性），我们可以通过对$X^TX$做特征值分解来考察这个模型，如果特征值趋近于0，则就可以知道$X^TX$行列式趋近于0。或者，多重共线性可以通过VIF指标监测，若$VIF > 10$，则可以认为多重共线性存在。至于如何解决多重共线性，筛选变量即可。

## 最小二乘法估计

???+ warning "变量上边加hat"
    从现在开始，带着帽子头的变量全都是估计值。由于恁Mathjax不支持 `\bm`也不支持 `\boldsymbol`，我就只能在帽子下面加箭头，太傻了。为了省一省我的工作量，除了下面的式子，后面带帽子的变量就不再直接区分标量和向量了，不过我想应该比较好分开。

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

???+ note "$X^T X$的性质"
    $X^T X$具有一些性质：

    1. $X^T X$是对称矩阵： $(X^TX)^T = X^TX$；

    2. $X^T X$是正定矩阵（特征值均不小于0），对于任何$k$阶向量$c$，有$c^T (X^T X)c \geq 0$，而且如果$\vec c \not = \vec 0$，则前面的二次型必然大于0（即严格正定）；

    3. $X^T X$是一个$k\times k$的方阵。

??? note "OLS的特殊情形"
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

将视线转向残差，可以得到残差的几项性质：

**性质1**： $X^T \hat \epsilon = 0$
??? success "残差性质1证明"
    $$
    \begin{align}
    X^T \hat \epsilon &= X^T(Y - \hat Y) = X^T Y - X^T \hat Y\\
    &=X^T Y - X^T X \hat \beta = X^T Y - (X^TX) (X^TX)^{-1} X^TY\\
    &= X^TY - X^TY = 0
    \end{align}
    $$

由于不存在完美的多重共线性，所有的$X$中各个向量在多维空间中形成了一个超平面，而原始的$Y$亦是多维空间中的一个向量，二者并不一定共面，$Y$在超平面上的投影即为$\hat Y$，而剔除掉这个投影向量后，剩下的那部分（即残差）是垂直于超平面的，这也就是性质1的几何解释。而这个投影对应的投影矩阵(Projection Matrix)为$P = X(X^TX)^{-1}X^T$，这是一个$N \times N$的方阵。而再引入一个Annihilator Matrix $M = I - P$，其中I是N阶单位阵。
> $PY = \hat Y$，$\hat Y$即为$Y$的投影，而P即为投影矩阵。
>
>"Annihilator"在数学中有三种译法，第一种叫“零化子”，常用于环论和泛函分析；第二种译作“消灭矩阵”，即此处所指Annihilator Matrix，用于回归分析；第三种用于解决非齐次常微分方程的“吸纳法”（Annihilator Method）。

???+ note "投影矩阵和Annihilator矩阵的性质"

    （1）$P$和$M$都是方阵和对称矩阵。
    
    （2）$P$和$M$都是幂等矩阵(idempotent matrix)。 即$P^2 = P$,$M^2 = M$，不难证明。
    
    （3）$PX = X$, $MX = 0$，不难证明。