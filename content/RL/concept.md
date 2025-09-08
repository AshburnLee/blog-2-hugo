+++
date = '2025-08-31T12:52:05+08:00'
draft = false
title = 'Concept'
tags = ["Reinforcement Learning","General"]
categories = ["Reinforcement Learning"]
+++


## 环境包装器 (environment wrappers) 
是一种修改现有环境而不直接更改其底层代码的便捷方法 . 包装器允许您避免大量重复代码，并使您的环境更模块化 . 重要的是，包装器可以链接起来以组合它们的效果，并且大多数通过 gym.make() 【python gymnasium 包】生成的环境默认情况下已经被包装。


### 环境包装器的作用:
- 转换 Actions (动作)：
- 转换 Observations (观测)：
- 转换 Rewards (奖励)：
- 自动重置环境：有些用户可能想要一个包装器，当其包装的环境达到完成状态时，该包装器将自动重置其包装的环境。这种环境的一个优点是，当超出完成状态时，它永远不会像标准 gym 环境那样产生未定义的行为。


### 如何使用包装器
要包装一个环境，您必须首先初始化一个基本环境。 然后，您可以将此环境以及（可能可选的）参数传递给包装器的构造函数

~~~python
import gymnasium as gym
from gymnasium.wrappers import RescaleAction

# 创建一个基本环境
base_env = gym.make("Hopper-v4")

# 使用 RescaleAction 包装器，将动作范围缩放到 [0, 1]
wrapped_env = RescaleAction(base_env, min_action=0, max_action=1)
~~~

### Gymnasium 中的常见包装器
- gymnasium.Wrapper: 所有包装器的基类
- gymnasium.ActionWrapper: 用于转换动作的包装器
- gymnasium.ObservationWrapper: 用于转换观测的包装器
- gymnasium.RewardWrapper: 用于转换奖励的包装器
- gym.wrappers.AutoResetWrapper: 用于自动重置环境的包装器

环境包装器是强化学习中一个强大的工具，可以帮助您修改和定制环境，以满足您的特定需求。它们提供了一种模块化和可重用的方式来转换动作、观测和奖励，并添加其他功能。


## 归一化

为什么RL 需要归一化？

- 提高训练稳定性：归一化可以使神经网络的输入或输出值接近正态分布，这有**助于激活函数正常工作**，并避免随机初始化的参数需要被过度调整. 它可以减少模型对初始化的敏感性，使得训练过程更加稳定。
- 加速收敛：归一化消除了数据特征之间的**量纲影响**，使得梯度下降算法更快地找到全局最优解，从而加速模型的收敛速度。
- 提高泛化能力：归一化可以减少**特征之间的相关性**，从而提高模型的稳定性和精度，增强模型的泛化能力。
- 允许使用更高的学习率：归一化可以使**参数空间更加平滑**，因此可以使用更高的学习率，而不会导致训练过程不稳定。
- 解决数据可比性问题：归一化可以将**有量纲转化为无量纲**，同时将数据归一化至同一量级，解决数据间的可比性问题。


## Bellman 方程

Value-based methods 通过迭代更新价值函数来学习。更新的依据是 贝尔曼方程 (Bellman Equation)，该方程描述了当前状态的价值与未来状态的价值之间的关系

例如，对于 **Q-Learning** 算法，其更新公式为：

`Q(s, a) = Q(s, a) + α [R + γ max_a' Q(s', a') - Q(s, a)]`


## Bias-variance Tradeoff

机器学习中，Bias-Variance Tradeoff 描述的是模型在泛化能力上的一个核心挑战.

- Bias（偏差）：指模型**预测值**与**真实值**之间的系统性差异。高偏差的模型通常过于简化，无法捕捉数据中的复杂关系，导致欠拟合（Underfitting）。
- Variance（方差）：指模型对**训练数据微小变化的敏感程度**。高方差的模型通常过于复杂，过度拟合训练数据中的噪声，导致在未见过的数据上表现不佳（Overfitting）。

在强化学习中，Bias-Variance Tradeoff 不仅体现在模型对数据的拟合程度上，还体现在对价值函数的估计上 。具体来说，当我们使用 Monte Carlo (MC) 方法和 Temporal Difference (TD) 方法来估计价值函数时，会面临不同的 Bias 和 Variance：

- Monte Carlo (MC) 方法：

  - 原理：MC 方法通过完整 episode 的回报来估计价值函数。

  - Bias：MC 方法是无偏的 (unbiased)，因为它是通过实际的回报来计算的。

  - Variance：MC 方法的方差较高，**因为 episode 的回报受到 episode 中所有动作的影响**。如果 episode 中存在随机性，回报的波动会很大。

- Temporal Difference (TD) 方法：

  - 原理：TD 方法通过当前状态的奖励和下一个状态的价值函数来估计价值函数。

  - Bias：TD 方法是有偏的 (biased)，因为它依赖于对下一个状态价值函数的估计。如果价值函数的估计不准确，TD 方法会引入偏差 

  - Variance：TD 方法的方差较低，因为它只依赖于当前状态的奖励和下一个状态的价值函数，对 episode 中后续动作的依赖较小


从拟合角度讲 Underfitting 和 Overfitting 的角度

  - Monte Carlo (MC)：类似于考虑所有导致死亡的因素，包括很久以前的因素。这可能导致过度拟合 (Overfitting)，**因为它考虑了所有可能的因素，包括噪声** 。
  - Temporal Difference (TD)：类似于只考虑最近的因素。这可能导致欠拟合 (Underfitting)，**因为它忽略了长期因素**。

如何权衡 这个Tradeoff？
1. 调整 Discount Factor (γ)：
    较高的 γ 值：考虑更长远的回报，可能增加 Variance。
    较低的 γ 值：更关注即时回报，可能增加 Bias。
2. 使用 使用 Eligibility Traces。
3. 选择合适的算法。

理解不同算法和参数对 Bias 和 Variance 的影响，可以帮助我们选择合适的策略，从而获得更好的学习效果。

Bias-Variance Tradeoff 真正关注的是**模型在未见过的数据上的表现**。如果模型只是简单地记住了训练数据，而不能泛化到新数据，那么它就不是一个好的模型。


## RLHF

RLHF (Reinforcement Learning from Human Feedback) 是一种**训练框架或训练方法** 。它结合了监督学习、强化学习和人类反馈，用于训练 AI 模型，使其行为更符合人类的偏好和价值观。

1. 预训练语言模型 (Pretraining a Language Model)： 首先，使用大量的文本数据预训练一个语言模型。这个模型可以是一个 Transformer 模型。

2. 训练奖励模型 (Training a Reward Model)： 
    
    收集人类对模型输出的偏好数据。例如，对于同一个 prompt，让模型生成多个不同的回复，然后让人类对这些回复进行排序。

    使用这些偏好数据训练一个奖励模型。奖励模型的目标是预测人类对模型输出的偏好程度。

3. 使用强化学习微调语言模型 (Fine-tuning the Language Model with Reinforcement Learning)：

    使用奖励模型作为奖励函数，使用强化学习算法（例如 Proximal Policy Optimization (PPO)）来微调预训练的语言模型。

    通过强化学习，语言模型可以学习生成能够获得更高奖励（即更符合人类偏好）的输出。


## 多模态 RL

Multi-Modal Reinforcement Learning, MMRL) 是一种强化学习方法，它利用来自多种不同类型的数据（即模态）来训练智能体，使其能够在复杂环境中做出更明智的决策。

模态 (Modalities) 指的是**不同类型的数据来源或表示形式**。常见的模态包括：

  - 视觉 (Vision)：图像、视频
  - 语言 (Language)：文本、语音
  - 触觉 (Tactile)：触觉传感器数据
  - 听觉 (Audio)：声音
  - 其他传感器数据：例如，温度、湿度、压力等


## 具身智能 

Embodied Intelligence。是指智能体（如机器人、无人机、智能汽车等）通过自身的物理实体与环境实时交互，实现感知、认知、决策和行动一体化的智能系统。

智能的本质不仅仅源于“头脑”，而是**必须通过身体与环境的动态互动来塑造和体现**。

具身智能被认为是推动人工智能迈向通用智能（AGI）的关键路径之一。

RL 是实现具身智能的重要工具。RL 提供了一种框架，用于训练智能体在与环境的互动中学习最优策略。

{去机器人社区看看，找找基于Jetson的机器人项目}

方言专家{对话}


