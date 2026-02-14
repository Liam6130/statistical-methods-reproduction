# 贝叶斯网络理论完全指南

**——从零开始的系统学习笔记**

---

## 第一章：贝叶斯统计学基础

### 1.1 什么是概率？

在讨论贝叶斯之前，我们先问一个根本问题：**什么是概率？**

#### 频率学派 vs 贝叶斯学派

| | 频率学派 | 贝叶斯学派 |
|--|--|--|
| 概率定义 | 长期频率（抛100次硬币，正面约50次） | 主观信念（我90%确定明天会下雨） |
| 参数观点 | 参数是**固定常数** | 参数是**随机变量** |
| 信息来源 | 只有数据 | 数据 + 先验知识 |
| 核心工具 | 采样分布 | 后验分布 |

**关键区别**：贝叶斯学派认为概率可以代表"不确定性程度"，而不仅仅是"长期频率"。

---

### 1.2 贝叶斯定理

这是整个贝叶斯统计学的核心：

$$P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$$

#### 逐项解释

| 符号 | 名称 | 含义 |
|------|------|------|
| $P(\theta)$ | 先验概率 (Prior) | 在看到数据前，我们对参数的信念 |
| $P(D \| \theta)$ | 似然函数 (Likelihood) | 如果参数是$\theta$，观察到数据D的概率 |
| $P(\theta \| D)$ | 后验概率 (Posterior) | 看到数据后，更新后的信念 |
| $P(D)$ | 边缘似然 (Marginal Likelihood) | 数据的总体概率（归一化常数） |

#### 一个经典例子

**问题**：某种癌症的患病率是1%。有一种检测方法：
- 如果真的有癌症，检测阳性的概率是99%（灵敏度）
- 如果没有癌症，检测阳性的概率是5%（假阳性率）

现在你检测结果是阳性，你真的有癌症的概率是多少？

**频率学派解法**：
- 10000人中，100人患癌
- 其中99人检测阳性
- 9900人不患癌，其中5%假阳性 = 495人
- 所以阳性的人中，真正患癌的 = 99 / (99 + 495) = **16.7%**

**贝叶斯解法**：
$$P(癌|阳性) = \frac{P(阳性|癌) \cdot P(癌)}{P(阳性)}$$
$$= \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} = 16.7\%$$

**结论**：即使检测阳性，你真正患癌的概率也只有16.7%！

---

### 1.3 先验分布 (Prior Distribution)

先验分布 = 在看到数据之前，我们对参数的信念。

#### 先验的类型

**1. 无信息先验 (Non-informative Prior)**
- 表示"我不知道"
- 例如：均匀分布 U(0, 1)

**2. 信息先验 (Informative Prior)**
- 基于已有知识
- 例如：根据以往研究，参数$\theta$大约是5

**3. 共轭先验 (Conjugate Prior)**
- 与似然函数共轭，数学上更方便
- 例如：Beta分布 + 二项分布 → Beta分布

#### 先验的影响

- **强先验**：数据少时影响大
- **弱先验**：数据多时影响小
- 当数据足够多时，**后验几乎与先验无关**

---

### 1.4 后验分布 (Posterior Distribution)

后验分布 = 贝叶斯分析的**终点**。

从后验分布我们可以提取：

1. **后验均值**：$E[\theta | D]$
2. **后验中位数**：$\tilde{\theta}$
3. **后验众数**：$\arg\max P(\theta | D)$
4. **可信区间 (Credible Interval)**：如95%可信区间

#### 可信区间 vs 置信区间

| | 可信区间 (Bayesian) | 置信区间 (Frequentist) |
|--|--|--|
| 含义 | 参数有95%概率落在区间内 | 重复实验100次，95次的区间包含真值 |
| 解读 | ✓ 可以直接解读 | ✗ 不能直接解读 |
| 本质 | 概率分布 | 频率覆盖 |

---

## 第二章：贝叶斯网络基础

### 2.1 什么是贝叶斯网络？

**贝叶斯网络 = 有向无环图(DAG) + 条件概率表**

它用图形化方式表示变量之间的**条件依赖关系**。

#### 一个简单例子

```
[感冒] → [发烧]
[感冒] → [咳嗽]
```

意思是：
- 感冒直接导致发烧
- 感冒直接导致咳嗽
- 给定"感冒"后，"发烧"和"咳嗽"条件独立

---

### 2.2 条件独立性 (Conditional Independence)

这是贝叶斯网络最核心的概念！

#### 定义

$A \perp B | C$ 表示：**在给定C的条件下，A和B条件独立**

即：$P(A | B, C) = P(A | C)$

#### 直观理解

**例子**：收入、学历、工作经验

- 收入 ⊥ 学历 ✗ (不独立，收入和学历相关)
- 收入 ⊥ 学历 | 工作经验 ✓ (给定工作经验后，学历和收入独立)

因为：工作经验 → 学历，收入

---

### 2.3 d-分离 (d-separation)

d-分离 = 判断两个节点是否条件独立的图形化方法

#### 三种基本结构

**1. 链式 (Chain)**
```
A → B → C
```
- A ⊥ C | B ✓ (给定B后，A和C独立)
- 原因：B"阻断"了A到C的信息

**2. 叉式 (Fork)**
```
A ← B → C
```
- A ⊥ C | B ✓ (给定B后，A和C独立)
- 原因：B是"共同原因"

**3. 对撞 (Collider)**
```
A → B ← C
```
- A ⊥ C | B ✗ (给定B后，A和C相关！)
- A ⊥ C ✓ (不给定B时，A和C独立)
- 原因：B是"共同结果"

---

### 2.4 贝叶斯网络的数学表示

对于节点$X_1, ..., X_n$，联合概率分布：

$$P(X_1, ..., X_n) = \prod_{i=1}^{n} P(X_i | Pa(X_i))$$

其中$Pa(X_i)$是$X_i$的父节点。

**例子**：
```
[A] → [B] → [C]
```

$$P(A, B, C) = P(A) \cdot P(B|A) \cdot P(C|B)$$

---

## 第三章：BNMC方法详解

### 3.1 论文在做什么？

McNally等人(2017)想检验两种模型（针对**OCD强迫症**）：

**模型A：潜在因子模型**
```
[潜在因子] → [Wash]
[潜在因子] → [Check]
[潜在因子] → [Hoarding]
...
```

**模型B：网络模型**
```
[Wash] ↔ [Check] ↔ [Obsessions] ↔ ...
```

---

### 3.2 如何区分两种模型？

关键思想：**条件独立性检验**

如果模型A（潜在因子模型）为真：
- 给定潜在因子后，所有症状应该**条件独立**

如果模型B（网络模型）为真：
- 症状之间有直接因果联系，**不会条件独立**

---

### 3.3 偏相关 (Partial Correlation)

偏相关 = 排除其他变量影响后，两个变量之间的相关。

#### 计算公式

$$r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ} \cdot r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}$$

#### 直观理解

- **边缘相关**$r_{XY}$：X和Y的直接相关
- **偏相关**$r_{XY \cdot Z}$：排除Z的影响后，X和Y的相关

---

### 3.4 BNMC的检验流程

```
1. 使用因子分析估计潜在因子
   FactorAnalysis(n_components=1)

2. 对每对症状(i, j)：
   - 计算边缘相关 r_ij
   - 计算偏相关 r_ij|latent
   - 置换检验判断显著性

3. 统计结果：
   - 如果偏相关不显著 → 支持潜在因子模型
   - 如果偏相关显著 → 支持网络模型
```

---

### 3.5 我们的复现结果

| 指标 | 结果 |
|------|------|
| 测试的症状对数 | 120对 |
| 边缘相关显著 | 120对 (100%) |
| 偏相关显著 | 20对 (17%) |
| **条件独立** | **99对 (83%)** |

**结论**：83%的症状对在控制潜在因子后变得条件独立，支持**潜在因子模型**。

---

## 第四章：代码实现解读

### 4.1 数据模拟

```python
# 使用Beta分布生成0-3评分的症状数据
def generate_symptom_with_target_mean(n_samples, target_mean, target_std):
    # 将0-3评分转换为0-1的潜在严重程度
    target_mean_scaled = target_mean / 3.0

    # 使用Beta分布模拟潜在因子
    alpha = ...
    beta = ...
    latent = np.random.beta(alpha, beta, n_samples)

    # 转换为有序分类（0, 1, 2, 3）
    ordinal = np.zeros(n_samples, dtype=int)
    ordinal[latent >= thresholds[0]] = 1
    ordinal[latent >= thresholds[1]] = 2
    ordinal[latent >= thresholds[2]] = 3

    return ordinal
```

---

### 4.2 条件独立性检验

```python
def partial_correlation(data, var1, var2, controls):
    """计算偏相关"""
    if not controls:
        return data[var1].corr(data[var2])

    # 回归法计算偏相关
    X = data[controls].values
    y1 = data[var1].values
    y2 = data[var2].values

    # 残差化
    from numpy.linalg import lstsq
    X_design = np.hstack([np.ones((len(X), 1)), X])

    beta1 = lstsq(X_design, y1, rcond=None)[0]
    beta2 = lstsq(X_design, y2, rcond=None)[0]

    resid1 = y1 - X_design @ beta1
    resid2 = y2 - X_design @ beta2

    return np.corrcoef(resid1, resid2)[0, 1]
```

---

### 4.3 置换检验

```python
def conditional_independence_test(data, var1, var2, controls, n_permutations=1000):
    """置换检验判断条件独立"""

    # 观察到的偏相关
    obs_pc = partial_correlation(data, var1, var2, controls)

    # 置换检验
    perm_pcs = []
    for _ in range(n_permutations):
        perm_data = data.copy()
        perm_data[var1] = np.random.permutation(data[var1])  # 打乱顺序
        perm_pc = partial_correlation(perm_data, var1, var2, controls)
        perm_pcs.append(perm_pc)

    # p值：观察到的偏相关比置换后的更极端的比例
    p_value = np.mean(np.abs(perm_pcs) >= np.abs(obs_pc))

    return {
        'partial_correlation': obs_pc,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
```

---

## 第五章：常见问题FAQ

### Q1: 贝叶斯分析和普通t检验有什么区别？

- **t检验**：比较两组均值差异
- **贝叶斯分析**：估计参数的完整后验分布，可以计算"效应为正的概率"

### Q2: 先验怎么选？

- 数据少时：选择信息先验（基于文献或专家意见）
- 数据多时：弱先验即可，后验主要由数据决定
- 无信息先验：$\theta \sim U(-\infty, +\infty)$

### Q3: 贝叶斯网络和结构方程模型(SEM)有什么区别？

- **SEM**：通常是频率学派方法
- **贝叶斯网络**：可以同时估计结构和参数的后验分布

### Q4: 什么时候用贝叶斯方法？

1. 数据稀少时
2. 需要结合先验知识时
3. 需要概率性推断时
4. 模型复杂时（MCMC可以处理任意模型）

---

## 第六章：推荐学习资源

### 入门书籍
- "Bayesian Statistics for Beginners" - a step by step approach
- 《贝叶斯统计》 - 茆诗松

### 进阶书籍
- "Pattern Recognition and Machine Learning" - Bishop (Chapter 8)
- "Probabilistic Graphical Models" - Koller & Friedman

### 在线课程
- Coursera: "Bayesian Statistics"
- Statistical Rethinking (McElreath)

### Python库
- `pymc` / `pymc3` - 贝叶斯建模
- `arviz` - 后验分析
- `pgmpy` - 概率图模型
- `networkx` - 网络分析

---

## 总结

1. **贝叶斯定理**：后验 ∝ 似然 × 先验
2. **贝叶斯网络**：用DAG表示条件依赖
3. **条件独立**：给定C后，A和B独立
4. **BNMC方法**：通过检验条件独立性区分潜在因子模型和网络模型
5. **核心洞见**：症状之间的相关可能只是共同原因的表现

---

*本学习笔记由Claude Code AI辅助编写*
*旨在帮助理解贝叶斯网络理论*
