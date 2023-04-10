# 预备内容

如果$X = \ln(Y)$服从正态分布，则$Y$服从对数正态分布。假定$X \in N(\mu, \sigma^2)$，则$Y$服从的对数正态分布参数为：

$$
E(Y) = \exp(\mu + \frac{\sigma^2}{2}), \quad \operatorname{Var}(Y) = [\exp (\sigma^2) - 1]\exp(2\mu + \sigma^2)
$$

反之，如果$Y \sim LN(\mu_y, \sigma_y^2)$，则$X$服从的正态分布为：

$$
E(X) = \ln \left(\frac{\mu_y}{\sqrt{1+\sigma_y^2 / \mu_y^2}}\right), \quad \operatorname{Var}(X) = \ln (1+\frac{\sigma_y^2}{\mu_y^2})
$$
