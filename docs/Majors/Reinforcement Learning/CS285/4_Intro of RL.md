---
comments: true
---
# 强化学习简介

对应CS285 Lecture 4 “Introduction to Reinforcement Learning”。

## 1. 基础定义

在上一讲中，已经给出了RL所需的若干定义，如state, action, reward等，以及policy的表示方法。这里再次给出这些定义：

- $\mathbf{s}_t$: 状态，表示环境在第$t$期的状态;
- $\mathbf{o}_t$: 观察，表示智能体在第$t$期的观察，仅用于POMDP问题，因为此时智能体无法观测环境的状态$\mathbf{s}_t$;
- $\mathbf{a}_t$: 动作，表示智能体在第$t$期的动作;
- $\pi_\theta(\mathbf{a}_t \mid \mathbf{o}_t)$: 策略，表示智能体在观察$\mathbf{o}_t$下选择动作$\mathbf{a}_t$的概率,POMDP中使用;
- $\pi_\theta(\mathbf{a}_t \mid \mathbf{s}_t)$: 策略，表示智能体在状态$\mathbf{s}_t$下选择动作$\mathbf{a}_t$的概率。

RL的状态转移过程$p(\mathbf{s}_{t+1} \mid \mathbf{s}_t, \mathbf{a}_t)$满足马尔可夫性质，即这个过程与历史上的其他状态$\mathbf{s}_{j}, j \leq t-1$毫无关系。

![1681371069937](image/2_ImitationLearning/1681371069937.png)

在这个过程中，收益$r(\mathbf{s}_t,\mathbf{a}_t)$表示了智能体在状态$\mathbf{s}_t$下选择动作$\mathbf{a}_t$后获得的即时收益，它指示了当前状态和状态下的动作是否是合理的，以及有没有更好的选择。

以上的内容（状态、观测、动作、策略、状态转移、即时收益）构成了RL的基础定义。

### 强化学习中的马尔科夫链

考虑到状态转移过程，我们需要聊一下马尔科夫链。马尔科夫链由两部分构成：$\mathcal{M} = \{\mathcal{S},\mathcal{T}\}$，前者是状态空间，后者是状态转移乘子。状态空间$\mathcal{S}$可以是连续的，也可以是离散的，关键在于状态转移概率如何形成：$p(s_{t+1} \mid s_t)$。定义概率$\mu_{t,i} = p(s_t = i)$，那么状态转移乘子$\mathcal{T}_{i,j} = p(s_{t+1} = i \mid s_t = j)$，即在状态$j$下转移到状态$i$的概率。那么就可以简化为$\vec \mu_{t+1} = \mathcal{T} \vec \mu_t$，即状态转移过程。

### 强化学习中的马尔科夫决策过程

向马尔科夫链中加入动作和收益，就变成了马尔科夫决策过程（MDP）$\mathcaL{M} = \{\mathcal{A},\mathcal{S},\mathcal{T},r\}。设动作集为$\mathcal{A}$，使所有的合法动作$a \in \mathcal{A}$（可以是连续的，也可以是分散的）。仍然令第$t$期状态为$j$的概率为$\mu_{t,j} = p(s_t = j)$，令第$t$期选取动作$k$的概率为$\xi_{t,k} = p(a_t = k)$，而状态转移乘子变为$\mathcal{T}_{ijk} = p(s_{t+1}=i \mid s_t = j,a_t = k)，则有：

$$
\mu_{t+1},i = \sum_{j,k} \mathcal{T}_{ijk} \mu_{t,j} \xi_{t,k}
$$

而收益函数则是$\mathcal{S}$和$\mathcal{A}$的映射，即$r:\mathcal{S} \times \mathcal{A} \to \mathbb{R}$。

### 部分可观测的马尔科夫决策过程

更进一步，我们可以写出POMDP（Partially-observable Markovian Decision Process）的形式化表述。在此基础上多了两项：

- 观察空间$\mathcal{O}$，即智能体对状态的观测$o \in \mathcal{O}；
- 泄露概率$\Epsilon = p(o_t \mid s_t)$，即状态$s_t$下观测为$o_t$的概率，emission probability。

但是POMDP下行为人的收益不取决于观测，而是取决于不可观测的实际状态：$r(s,a)$。

### 目标

## 2. 算法思路和价值函数

## 3. 算法类型

## 4. 如何在算法间取舍？
