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

考虑到状态转移过程，我们需要聊一下马尔科夫链。马尔科夫链由两部分构成：$\mathcal{M} = \{\mathcal{S},\mathcal{T}\}$，前者是状态空间，后者是状态转移乘子。状态空间$\mathcal{S}$可以是连续的，也可以是离散的，关键在于状态转移概率如何形成：$p(s_{t+1} \mid s_t)$。首先定义概率$\mu_{t,i} = p(s_t = i)$，那么状态转移乘子$\mathcal{T}_{i,j} = p(s_{t+1} = i \mid s_t = j)$，即在状态$j$下转移到状态$i$的概率。那么就可以简化为$\vec \mu_{t+1} = \mathcal{T} \vec \mu_t$，即状态转移过程。

## 2. 强化学习与监督学习的区别