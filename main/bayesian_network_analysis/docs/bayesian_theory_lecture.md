# 贝叶斯统计学理论入门

## 1. 什么是贝叶斯统计学？

贝叶斯统计学是一种基于**概率论**的统计推断方法，其核心思想是：

> **一切都可被视为随机变量，并用概率分布来描述我们的不确定性。**

### 频率学派 vs 贝叶斯学派

| 方面 | 频率学派 | 贝叶斯学派 |
|------|----------|------------|
| 概率定义 | 长期频率（客观） | 主观信念（主观或客观） |
| 参数 | 固定常数 | 随机变量 |
| 核心工具 | 采样分布 | 后验分布 |
| 信息来源 | 仅数据 | 数据 + 先验 |

---

## 2. 贝叶斯定理

贝叶斯统计学的核心是**贝叶斯定理**：

$$P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$$

其中：
- $P(\theta | D)$ = **后验概率** (Posterior) - 给定数据后的信念
- $P(D | \theta)$ = **似然函数** (Likelihood) - 数据在给定参数下的概率
- $P(\theta)$ = **先验概率** (Prior) - 在看到数据前的信念
- $P(D)$ = **边缘似然** (Marginal Likelihood) - 数据的总概率

### 直观理解

```
后验 ∝ 似然 × 先验
（新信念） （数据）   （旧信念）
```

---

## 3. 先验分布 (Prior)

先验分布代表**在观察数据之前**我们对参数的信念。

### 先验类型

1. **信息先验 (Informative Prior)**
   - 基于已有知识
   - 例：Previous studies show $\mu \approx 100$

2. **无信息先验 (Non-informative Prior)**
   - 表示"不知道"
   - 例：Uniform prior, Jeffreys prior

3. **共轭先验 (Conjugate Prior)**
   - 与似然函数共轭，简化计算
   - 例：Beta-Binomial, Normal-Normal

---

## 4. 后验分布 (Posterior)

后验分布是**贝叶斯分析的终点**。

### 从后验分布可以提取：
- **后验均值** (Posterior Mean): $E[\theta | D]$
- **后验中位数** (Posterior Median)
- **后验众数** (Posterior Mode)
- **可信区间** (Credible Interval): 如 95% 可信区间

### 可信区间 vs 置信区间

| 可信区间 (Bayesian) | 置信区间 (Frequentist) |
|---------------------|------------------------|
| 参数有 95% 概率落在区间内 | 重复采样 95% 的区间包含真值 |
| 可直接解读 | 不可直接解读 |

---

## 5. 贝叶斯网络 (Bayesian Network)

### 什么是贝叶斯网络？

贝叶斯网络是一种**有向无环图 (DAG)**，用于表示变量之间的**条件依赖关系**。

### 核心概念

1. **条件独立性 (Conditional Independence)**
   - $A \perp B | C$ 表示在给定 C 的情况下，A 和 B 条件独立
   - 这是贝叶斯网络的核心

2. **d-分离 (d-separation)**
   - 用于判断两个节点是否条件独立
   - 路径被阻塞的三种情况：
     - 链式：A → B → C（给定 B 时，A ⊥ C）
     - 叉式：A ← B → C（给定 B 时，A ⊥ C）
     - 对撞：A → B ← C（不给定 B 时，A ⊥ C；给定 B 时，A 和 B 相关）

### 在 McNally et al. (2017) 论文中的应用

论文使用的**贝叶斯网络马尔可夫检查 (BNMC)** 核心逻辑：

```
假设检验：
- H0 (零假设): 症状是条件独立的（存在潜在因素）
- H1 (备择假设): 症状之间有直接因果联系

方法：
1. 构建包含潜在因子的贝叶斯网络
2. 测试症状对的条件独立性
3. 如果条件独立 → 支持潜在因子模型
   如果条件不独立 → 支持网络模型
```

---

## 6. 马尔可夫性质

### 马尔可夫毯 (Markov Blanket)

对于节点 $X_i$，其马尔可夫毯包括：
- 父节点 (Parents)
- 子节点 (Children)
- 其他子节点的父节点 (Spouses)

给定马尔可夫毯，$X_i$ 与其他所有节点条件独立。

### 全局马尔可夫性

如果两组节点被第三组节点 d-分离，则它们条件独立。

---

## 7. 计算方法

### 精确推断
- 变量消除 (Variable Elimination)
- 信念传播 (Belief Propagation)

### 近似推断
- 马尔可夫链蒙特卡洛 (MCMC)
- 变分推断 (Variational Inference)

---

## 8. 实际应用步骤

```python
# 贝叶斯网络分析基本流程

# 1. 定义网络结构（基于领域知识或学习）
model = BayesianNetwork([
    ('Latent', 'Intrusion'),
    ('Latent', 'Avoidance'),
    ('Intrusion', 'Nightmares')
])

# 2. 估计参数（最大似然或贝叶斯）
model.fit(data, estimator=MaximumLikelihoodEstimator)

# 3. 进行推断
from pgmpy.inference import VariableElimination
inference = VariableElimination(model)

# 4. 查询条件概率
result = inference.query(variables=['Avoidance'], evidence={'Intrusion': 1})
```

---

## 9. 本论文的关键洞见

McNally et al. (2017) 的核心论点：

1. **网络方法的问题**：使用偏相关（partial correlation）构建的网络可能产生虚假的因果关系

2. **贝叶斯网络检验**：通过检验条件独立性，可以区分：
   - **潜在因子模型**：所有症状由一个隐藏因子引起
   - **网络模型**：症状之间有直接因果联系

3. **结论**：在他们的数据中，症状在控制潜在因子后变得条件独立，支持了"症状不直接相互因果"的观点

---

## 推荐阅读

1. **入门**:
   - "Bayesian Statistics for Beginners" - a step by step approach

2. **进阶**:
   - "Pattern Recognition and Machine Learning" - Bishop (Chapter 8)
   - "Probabilistic Graphical Models" - Koller & Friedman

3. **贝叶斯网络**:
   - "Bayesian Networks and Decision Graphs" - Jensen

---

*本讲义由你的贝叶斯小讲师编写 🎓*
