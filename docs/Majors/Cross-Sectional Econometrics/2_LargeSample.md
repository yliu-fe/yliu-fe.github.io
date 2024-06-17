---
comments: true
---

# 大样本理论

当样本量极大($n \to +\infty$)时，会发生什么？

上一单元的内容中，我们给出了五个假设：

- 线性假设：$Y = X\beta + \epsilon$；
- 严格外生性假设： $E[\epsilon|X] = 0$；
- 无完美多重共线性假设： $Rank(X) = K$；
- 同方差假设： $Var[\epsilon|X] = \sigma^2 I_N$；
- 正态性假设： $f(\epsilon|X) \sim N(0,\sigma^2 I_N)$

在大样本理论中，我们将放松正态性假设；进一步，尝试放松同方差假设来研究异方差问题，或者放松严格外生假设，来研究内生性问题（并讨论广义矩估计法）；最后，尝试放松线性假设，考虑非线性模型（与极值估计量问题）。在这一单元中，我们讨论大样本理论和异方差问题。

## 从特殊情况开始

$$
\hat \beta = (X^TX)^{-1}X^TY = \frac{\sum_{i=1}^N x_{1i}Y_i}{\sum_{i=1}^N x_{1i}^2} = \beta + \frac{\sum X_i \epsilon_i}{\sum X_i^2} \tag{2.1} \label{eq: 2.1}
$$

那么，如果$N\to +\infty$，则明显有$\sum_{i=1}^N x_i^2 \to +\infty$，然后讨论分子。先看它的方差：

$$
\begin{align}
    Var[\sum_{i=1}^N X_i \epsilon_i |X] = \sum_i^N Var(X_i\epsilon_i|X_i) = \sum_{i=1}^N X_i^2 Var(\epsilon_i|X) = \sigma^2 \sum_{i=1}^N X_i^2 \to +\infty
\end{align}
$$

在这一单元中，我们很多时候都会和这个式$\ref{eq: 2.1}$中的分式（上的两个求和）打交道，对于分子，我们希望其服从中心极限定理（Central Limit Theorem,CLT），而对于分母则讨论大数定律(Law of Large Number, LLN)。将二者应用于这个分式，则有：

$$
\hat \beta =\beta + \frac{\sum X_i \epsilon_i}{\sum X_i^2} = \beta + \frac{\sum X_i \epsilon_i / N}{\sum X_i^2 / N}
$$

从而应用LLN，进而得出$\hat \beta$依概率收敛于某个$\beta$（一致性）。如果这个真的成立了，那么进一步，我们更希望能让上下依分布收敛，那么就要尝试应用CLT，尝试得到$\sqrt{N}(\hat \beta - \beta) \mathop{\to}\limits^{d} N(0,Var)$的结论。

### 1.1 预备知识：随机变量的收敛性

收敛有几种，比较常见的有依概率收敛（收敛到一个常数）、依分布收敛（到一个随机变量/分布）。假设有一个随机变量序列$\{z_j\}$,比如$\{z_j|\sum_i^j x_i\epsilon_i/j\}$：

**（一）依概率收敛**：对于随机变量序列$\{z_n\}$，如果对于任何$\epsilon > 0$，有$\lim P[|z_N - \alpha| > \epsilon]  = 0$，则称$z_N$依概率收敛于$\alpha$（一个常数），记作$z_N \mathop{\to}\limits^P \alpha$。亦可以称作$\text{plim} z_N = \alpha$，即$z_N$的概率极限为$\alpha$。其几何意义在于，如果样本足够大，$z_N$的分布曲线（PDF）会逐渐变成一条垂直于x轴的直线，只在$z_N = \alpha$处概率为无穷大，其他点为0。对于一个估计量来说，如果在大样本情况下$\hat \beta \mathop{\to}\limits^{P} \beta$，则称其具有**一致性**。依概率收敛通常利用**弱大数定律**证明。

**（二）依分布收敛**：如果$N \to \infty$，$z_N$的CDF收敛到变量$z$的CDF曲线，或者说对于任意连续点$a$，有$P(z_N \leq a) \to P(z \leq a)$，则称$z_N$依分布收敛到$z$（一个随机变量），记作$z_N \mathop{\to}\limits^d z$。有的文献也称$z$是$z_N$的渐进分布/极限分布(asymptotic/limiting distribution)。其几何意义是随着样本量不断变大时，$z_N$的分布曲线（PDF）逐渐与$z$的分布曲线重合。依分布收敛是最弱的收敛形式，如果知道依概率收敛，则可以推出依分布收敛，但无法返回。依分布收敛通常利用**中心极限定理**来证明。

**（三）依概率1收敛/几乎一定收敛**：如果$P[\lim\limits_{N\to \infty} Z_N = \alpha] = 1$，则称$Z_N$依概率1收敛于常数$\alpha$，记作$Z_N \mathop{\to}\limits^{a.s.} \alpha$。这是一个更强的收敛形式，如果依概率1收敛成立，则对应的依概率收敛自然成立。

要证明随机变量序列（向量）依概率收敛于一个向量，最笨的办法就是一个一个证，这是可行的，而且是双向推出的，但可以用大数定律来简化这个证明。对于依分布收敛问题，就不能用那个笨办法，只能**以向量为整体**来证明。

> 要考虑向量里各个随机变量的联合分布。两个正态分布联立起来可能还是多元正态，也可能是各种牛鬼蛇神——这个问题被称作Copula问题，即知道边际分布是没有办法推出联合分布的。

那么怎么证呢？（1）直接讨论联合分布（一般用CLT来进行）；（2）Cramer-Wold定理。

> Cramer-Wold定理将$Z_N \mathop{\to}\limits^d Z_{k\times 1}$问题转化为$a^T Z_N \mathop{\to}\limits^d a^T Z$，其中$a$为任意的$k$阶向量。从而把向量的依概率分布问题转化为单一随机变量的依概率分布问题，但是麻烦的地方在于$a$的任意性。

### 1.2 大数定律和中心极限定理

**Chebyshev不等式**：$P(|x-\mu| > k\sigma) \leq \frac{1}{k^2}$。

/// details | Chebyshev不等式的证明

$$
\begin{align}
    \sigma^2 &= E[(x-\mu)^2]\\
    &= P_1E[(x-\mu)^2|(x-\mu)^2 \leq k^2 \sigma^2] +  P_2E[(x-\mu)^2|(x-\mu)^2 > k^2 \sigma^2] \tag{2.5}\\
    &\geq 0 + k^2 \sigma^2 P(|x-\mu| > k\sigma)\tag{2.6}
\end{align}
$$

> 式2.5-2.6的过程，两个条件期望直接取最小，第一项条件期望的最小为0（故第一项在2.6中为0），第二项条件期望最小为$k^2 \sigma^2$。
///

接下来将$Z_N$的均值、方差代入，即有Chebyshev大数定律。

**Chebyshev弱大数定律**：*如果对于向量$Z_N$，有$\lim\limits_{N \to +\infty} E[Z_N] = M$，且$\lim\limits_{N \to \infty} Var(Z_N) = 0$，则有$z_N \mathop{\to}\limits^P M$。*

对应的，还有强大数定律，即Kolmogorov大数定律（SLLN）：*假设${z_i}$序列独立同分布(i.i.d)，且$E|z_i| < \infty, E(z_i) = \mu$，则$Z_N \mathop{\to}\limits^{a.s.} \mu$*。

> 在Kolmogorov大数定律中，$Z_N$几乎一定(almost surely)收敛于$\mu$，或者称以概率1收敛。

**Linderberg-Levy 中心极限定理**：若向量$\{z_i\}$独立同分布(i.i.d)，令其均值为$\mu$，方差为$\Sigma$，那么有：

$$
\sqrt{N}(\overline Z_N - \mu) = \frac{1}{\sqrt{N}} \sum(z_i - \mu) \mathop{\to}\limits^d MN(0,\Sigma)
$$

这个方差矩阵被称作渐近分布方差(variance of asymptotic distribution)。它的证明需要引入矩母函数(Moment Generating Function)，对于一个随机变量$X$，其矩母函数$M_X(t) = E[e^{tx}]$。但由于很多时候矩母函数会趋近于正无穷，所以现在多使用特征函数来代替：$\phi_X(t) = E[e^{itx}]$，其中$e^{ix} = \cos x + i \sin x \leq 1$。可以证明的是，对于任意的随机变量，其PDF和特征函数是一一对应的。

/// details | Linderberg-Levy中心极限定理的证明
        type: success

我们设$E[X_i] = \mu, Var(X_i) = \sigma^2$，则$Y_i = \frac{X_i - \mu}{\sigma}$的均值为0，方差为1。那么那么尝试证明$\frac{1}{\sqrt{N}} Y_i \mathop{\to}\limits^d N(0,1)$即可。对于左边的式子(设为$Z_N$)，有：

$$
\begin{align}
    \phi_{Z_N} (t) &= E[e^{itZ_N}] = E[e^{it} \frac{1}{\sqrt{N}} Y_i] = \prod E[e^{it Y_i/\sqrt{N}}]\\
    &= \prod \phi_{Y_i} (\frac{t}{\sqrt{N}}) = \phi_{Y_1}^N (\frac{t}{\sqrt{N}}) \tag{2.7}
\end{align}
$$

对于特征函数的定义式做Taylor展开，有$E[e^{itx}] = E[1+itx-\frac{1}{2}t^2 x^2 + o(x^3)]$，在$x = 0$处求值，则$\phi_Y(0) = 1, \phi^{'}_Y (0) = i E(Y) = 0,\phi^{''}_Y(0) = i^2 E(Y^2) = -1$。带回到式2.7，有：

$$
\phi_{Z_N} (t) = \phi_{Y_1}^N (\frac{t}{\sqrt{N}}) = [1-\frac{1}{2}\frac{t^2}{N}]^N = e^{-\frac{t^2}{2}}
$$

而对于$X \sim N(0,1)$有:

$$
\phi_X(t) = E[e^{itx}] = \int e^{itx} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dx = e^{-\frac{t^2}{2}}
$$

从而有$\phi_X(t) = \phi_{Z_N} (t)$即$Z_N \mathop{\to}\limits^d X \sim N(0,1)$，得证。
///

### 1.3 一些其他的结论

如果$f()$函数是连续的，且其形式与样本量N无关，则：

1. 若$Z_N \mathop{\to}\limits^P \alpha$，则$f(Z_N) \mathop{\to}\limits^P f(\alpha)$;
2. 若$Z_N \mathop{\to}\limits^d z$，则$f(Z_N) \mathop{\to}\limits^P f(z)$。

基于这两个结论还可以进一步扩展。如果$X_N \mathop{\to}\limits^P \beta, Y_N \mathop{\to}\limits^P \gamma$，则有：

$$
\left[\begin{array}{c}
    X_N\\
    Y_N
\end{array}\right] \mathop{\to}\limits^P \left[\begin{array}{c}
    \beta\\
    \gamma
\end{array}\right], \begin{array}{cc}
        X_N + Y_N \mathop{\to}\limits^P \beta + \gamma&\\
        X_N \cdot Y_N \mathop{\to}\limits^P \beta \cdot \gamma&\\
        X_N / Y_N \mathop{\to}\limits^P \beta / \gamma&(\gamma \not = 0)\\
        Y_N^{-1} \mathop{\to}\limits^P \gamma^{-1} &(\det(\gamma) \not = 0)
\end{array}
$$

此外，Hayashi Lemma 2.4提到了一个结论(Hayashi, 2000, p.92)：

- 如果$X_N \mathop{\to}\limits^d X$，且$Y_N \mathop{\to}\limits^P \alpha$，则$X_N + Y_N \mathop{\to}\limits^d X + \alpha$；
- 如果$X_N \mathop{\to}\limits^d X$，且$Y_N \mathop{\to}\limits^P 0$，则$Y_N^T X_N \mathop{\to}\limits^P 0$；
- 如果$X_N \mathop{\to}\limits^d X$，且$A_N \mathop{\to}\limits^P A$，则$A_N X_N \mathop{\to}\limits^d Ax$；
- 如果$X_N \mathop{\to}\limits^d X$，且$A_N \mathop{\to}\limits^P A$，则$X_N^T A_N^{-1} X_N \mathop{\to}\limits^d X^T A^{-1} X$。

这四条的意义是，除非强收敛的结果为0（且相乘），否则两个不同收敛的结果进行运算时，其结果会以弱的方式收敛。这一点在矩阵运算中同样成立。

### 1.4 从中心极限定理向外拓展

由于中心极限定理，我们知道$z_N = \sqrt{N} \frac{\bar X - \mu}{\sigma} \sim N(0,1)$，那么$P(z_N \leq t) =\Phi(t) =  \int_{-\infty}^t \frac{1}{\sqrt{2\pi}} \exp(-\frac{x^2}{2}) dx$。如果是个标准正态分布，那么拟合这个正态分布，也许只需要几十个样本。但是有的时候，正态分布可能存在高阶矩的问题（如偏度不为0），我们定义偏度为：

$$
Skew(x_i) = \frac{E[(x_i - \mu)^3]}{\sigma^3}
$$

它是分布的三阶矩，那么$P(z_N \leq t)$不再是简单的CDF就能体现出来的，利用三阶的Taylor展开：

$$
P(z_N \leq t) = \Phi(t) - \frac{Skew(x_i)}{\sqrt{N}} \frac{t^2 - 1}{6} \phi(t)
$$

$\Phi(t)$是CDF，而$\phi(t)$是PDF。在这种情况下，通常需要更多的数据来拟合正态分布（特别是其三阶矩）。

## 大样本下的OLS（一致性）

由于大数定律和中心极限定理的存在，在大样本情况下我们可以放松掉正态性假设。之后的新假设条件包括：

- 线性性假设（对应原假设1，没有变化）；
- 严格外生性假设（对应原假设2，没有变化）；
- $K\times K$矩阵$Q = E[X_i X_i^T]$是奇异矩阵（可逆）；（对应原假设3，针对大样本做出变化）
- 同方差假设（对应原假设4，没有变化）。
- 向量$\{Y_i, X_i\} i.i.d$，且$E[X_i] = \mu_x, Var(X_i) = \Sigma_x, E[X_{ji}^4] < +\infty, j = 1,...,K$；（中心极限定理的条件）

那么我们来看看OLS的估计结果：

$$
\hat \beta = (X^TX)^{-1}X^TY = \beta + (X^TX)^{-1}X^T\epsilon
$$

同除并同乘N，有：

$$
\hat \beta = \beta +  (\frac{X^TX}{N})^{-1}\frac{X^T\epsilon}{N} \tag{2.16}
$$

/// admonition | 对上式前半部分的讨论
        type: note

看第二项的重要内容$X^TX/N$，它的形式是：

$$
\frac{X^TX}{N} = \frac{1}{N} \sum_{i=1}^N X_i X_i^T
$$

我们设$Z_i = \frac{X^TX}{N}$，接下来讨论$Z_i$的问题。我们有很多种方法去证明$Z$依概率收敛于Q。

【方法一：强大数定律道路】由强大数定律可知$\bar Z_N \mathop{\to}\limits^{a.s.} E[Z_i] = E[X_i X_i^T]$，依概率1收敛可推出依概率收敛，即有$\frac{1}{N}X^T X \mathop{\to}\limits^P Q$。

> 这一方法要求$Z_i$独立同分布。假设2给出$X_i$独立同分布，而N是常数，因而$Z_i$独立同分布。

【方法二：弱大数定律道路】或者，我们不用强大数定律，用弱大数定律则要讨论极限问题：

$$
\begin{align}
    \lim_{N\to +\infty} E[\frac{X^TX}{N} ] &= \lim_{N\to +\infty} \frac{1}{N} \sum E[X_i X_i^T] = Q\\
    \lim_{N\to +\infty} Var[\frac{X^TX}{N} ] &= \frac{1}{N^2} Var(\sum_{i=1}^N X_i X_i^T)\\
    &=  \lim_{N\to +\infty} \frac{N}{N^2} Var(X_i X_i^T) =  \lim_{N\to +\infty} \frac{1}{N}Var(X_i X_i^T)
\end{align}
$$

最后的方差项是一个平方项的方差，因此和$X_i$的四阶矩有关，参照新假设第二条，这是一个有限的数，因而这个极限最终趋近于0，那么利用弱大数定律可以证得同样的结果。
///

/// admonition | 对上式后半部分的讨论
        type: note

再看那一项的后半部分$\frac{X^T \epsilon}{N}$:

$$
\frac{X^T \epsilon}{N} = \frac{1}{N} \sum X_i \epsilon_i \tag{2.21}
$$

若设$z_i = X_i \epsilon_i$则有：

$$
E[z_i] = E[X_i \epsilon_i] = E_{X_i} [E(X_i\epsilon_i|X_i)] = E_{X_i} [X_i E(\epsilon_i|X_i)] = 0 \tag{2.22}
$$

证明的方式依然有两种：

【方法一：强大数定律道路】要求$z_i$独立同分布，所以由大数定律，由$\frac{X^T \epsilon}{N} \mathop{\to}\limits^{a.s.} E[X_i \epsilon_i] = 0$，因而可以降级证明出依概率收敛。

> 由假设2，$X$独立同分布，而$\epsilon$由假设4知亦独立同分布，因而总体独立同分布。

【方法二：弱大数定律道路】同样，我们可以用弱大数定律来证明，但需要讨论极限性质：

$$
\lim_{N\to \infty} Var(\frac{1}{N}X^T \epsilon) = \lim_{N\to \infty} \frac{1}{N} Var(X_i \epsilon_i)
$$

为了证明$Var(X_i \epsilon_i)$为有限的，我们看一下这一项的性质，这是一个$K\times 1$的向量，其每一项的方差$Var(X_{ji} \epsilon_i) = E[(X_{ji} \epsilon_i)^2] - E^2(X_{ji} \epsilon_i) = E[(X_{ji} \epsilon_i)^2] = E_x [X_{ji}^2 E(\epsilon^2|X_{ji})] = \sigma^2 E[X_{ji}^2]$，这个必然是有限的。所以式2.21必然为0，即有$\frac{X^T \epsilon}{N}\mathop{\to}\limits^P 0_{K\times 1}$。
///

综上所述，对于式2.16的第二项，由于前半部分依概率收敛到$Q^{-1}$，而后者依概率收敛到0，因此第二项依概率收敛到0。那么，由式2.16可以推出$\hat \beta \mathop{\to}\limits^{P} \beta$，即OLS具有一致性。

## 大样本下的估计量性质

讨论大样本下$\hat \beta$的性质需要用到中心极限定理。在开始讨论中心极限定理之前，要先讨论一下$Var(X_i \epsilon_i)$的性质，与上一段拆开讨论不同，我们直接看总体的性质

$$
\begin{align}
    Var(X_i \epsilon_i) &= E[(X_i \epsilon_i)(X_i\epsilon_i)^T] - E[X_i\epsilon_i] E[X_i \epsilon_i]^T \\
    &=E[X_i X_i^T \epsilon_i^2] = E_x [E(X_i X_i^T \epsilon^2|X)] = \sigma^2 E[X_i X_i^T] = \sigma^2 Q
\end{align}
$$

再回到式2.16，有：

$$
\hat \beta - \beta = (\frac{X^TX}{N})^{-1}\frac{X^T\epsilon}{N} \tag{2.26}
$$

如果要利用中心极限定理，那么就要两边同乘$\sqrt{N}$，即有：

$$
\sqrt{N} (\hat \beta - \beta) = (\frac{X^TX}{N})^{-1}\frac{X^T\epsilon}{\sqrt{N}}
$$

第二项的前半部分和以前一样，依概率收敛到$Q^{-1}$，后半部分：

$$
\frac{X^T\epsilon}{\sqrt{N}} = \frac{1}{\sqrt{N}} \sum X_i \epsilon_i
$$

由式2.22，$z_i = X_i \epsilon_i$期望为0。由中心极限定理知$\frac{X^T\epsilon}{\sqrt{N}} = \frac{1}{\sqrt{N}} \sum X_i\epsilon_i \mathop{\to}\limits^d N(0,Var(X_i \epsilon_i)) = N(0,\sigma^2 Q)$。那么，对于式2.26，前半项依概率收敛到$Q^{-1}$，后半项依分布收敛到$N(0,\sigma^2 Q)$，那么：

$$
\sqrt{N} (\hat \beta - \beta) \mathop{\to}\limits^d Q^{-1} N(0,\sigma^2 Q) = N(0,\sigma^2 Q^{-1}) \tag{2.29}
$$

我们可以拿式2.29做假设检验了，但是Q和$\sigma^2$都是不可观测的，我们需要一组一致估计量$\hat \sigma^2$和$\hat Q$，使得二者均依概率收敛于$\sigma^2$和$Q$。参照对2.16前半部分的证明，我们可以拿出一个$\hat Q$：

$$
\hat Q = \frac{1}{N} \sum X_i X_i^T = \frac{1}{N}X^TX
$$

而$\hat \sigma^2 = \frac{1}{N-K} \sum \hat \epsilon_i^2$。证明的思路是，由于$\hat \epsilon$是$\hat \beta$的函数，且我们知道$\hat \beta$具有一致性，问题就是要写出$\hat\epsilon$与$\epsilon$和$\hat \beta$的函数关系。下面给出证明：
