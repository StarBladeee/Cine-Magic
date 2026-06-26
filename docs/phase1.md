# CineMagic Phase 1 — 基线模型说明文档

## 概述

Phase 1 的目标是在豆瓣 Top200 电影数据集上搭建推荐系统的基础框架，实现四个基线模型，建立评估体系，打通从数据加载到命令行推荐的全流程。所有功能均通过命令行调用，无需 Web UI。

---

## 1. 数据集

| 指标 | 数值 |
|------|------|
| 数据来源 | data/douban/top200.csv + top200_ratings.csv |
| 电影数 | 200 → 过滤后 192 |
| 评分记录 | 38,608 → 过滤后 13,227 |
| 用户数 | 20,652 → 过滤后 1,280 |
| 评分范围 | 1.0 — 5.0 |
| 平均评分 | 3.86 |
| 矩阵稀疏度（过滤前） | 0.944% |
| 矩阵稀疏度（过滤后） | 5.385% |

过滤条件：最少 5 次评分的用户 + 最少被评 5 次的电影。训练/测试按 80/20 随机划分。

---

## 2. 模型原理与实现

### 2.1 Popularity（热门基线）

**原理**

最简单的非个性化推荐策略，核心思想是"谁火推谁"——对所有用户推荐相同的热门电影，完全不考虑用户差异。流行度得分综合考虑四个维度：

```
pop_score = 0.6 × (豆瓣评分 / 10)              ← 豆瓣社区评价，归一化到 [0,1]
          + 0.4 × min(log(豆瓣评分数+1) / 15, 1) ← 评分数做 log 压缩，防头部支配
          + 0.3 × (用户评分均值 / 5)             ← 本站用户对这部电影的打分
          + 0.1 × min(log(评分人数+1) / 5, 1)    ← 本站评分人数
```

最终对权重和做归一化。推荐时按 pop_score 降序取 Top-K。此外支持按类型筛选的热门推荐（`get_popular_by_genre`）。

**实现**：`src/models/popularity.py` — `PopularityRecommender`

**适用场景**：新用户冷启动、全局热门榜单、按类型热门排行。不依赖任何用户行为，性能最优。

---

### 2.2 SVD（矩阵分解 / 隐语义模型）

**原理**

矩阵分解是协同过滤的经典方法，核心是将稀疏的 User-Item 评分矩阵 **R** (m×n) 分解为两个低秩矩阵的乘积：

```
R ≈ P × Qᵀ

P ∈ R^(m×k) : 用户的 k 维隐向量（User Embedding），表示用户偏好的 k 个潜在因子
Q ∈ R^(n×k) : 电影的 k 维隐向量（Item Embedding），表示电影在这 k 个因子上的属性
```

k 是一个远小于 m 和 n 的数值（我们在配置中设为 50），称为"隐因子维度"或"潜因子数"。这些维度不是人工定义的（如"动作片偏好"），而是算法从评分数据中自动习得的。

带偏置项的 SVD 预测公式（Funk SVD）：

```
r̂(u,i) = μ + bᵤ + bᵢ + pᵤ · qᵢ
```

| 符号 | 含义 | 示例 |
|------|------|------|
| μ | 全局平均评分 | 3.33 |
| bᵤ | 用户偏置（这人的打分习惯偏高还是偏低） | +0.5（这个用户普遍给分高）|
| bᵢ | 物品偏置（这部电影本身质量偏高还是偏低） | +1.2（公认好片） |
| pᵤ | 用户的 k 维隐向量 | [0.2, -0.5, 0.8, ...] |
| qᵢ | 电影的 k 维隐向量 | [0.3, -0.1, 0.6, ...] |
| pᵤ · qᵢ | 隐向量内积，表示用户与电影在多维空间中的匹配度 | |

**训练目标**（带正则化的 SGD）：

```
min Σ (r(u,i) - r̂(u,i))² + λ(||pᵤ||² + ||qᵢ||² + bᵤ² + bᵢ²)
                          ↑ L2 正则化，防止过拟合
```

训练过程中，每次取一个 (u, i, r) 三元组，计算误差，按梯度方向更新参数：

```
eᵤᵢ = r(u,i) - r̂(u,i)
bᵤ ← bᵤ + η(eᵤᵢ - λ·bᵤ)
bᵢ ← bᵢ + η(eᵤᵢ - λ·bᵢ)
pᵤ ← pᵤ + η(eᵤᵢ·qᵢ - λ·pᵤ)
qᵢ ← qᵢ + η(eᵤᵢ·pᵤ - λ·qᵢ)
```

**相似电影**

SVD 不需要显式计算物品相似度。通过隐向量的余弦相似度即可找到相似的电影：

```
sim(i, j) = (qᵢ · qⱼ) / (||qᵢ|| × ||qⱼ||)
```

**实现**：`src/models/collaborative.py` — `SVDRecommender`

**超参数**（config.yaml）：

| 参数 | 值 | 说明 |
|------|----|------|
| n_factors | 50 | 隐因子维度，越大模型容量越大但训练越慢 |
| n_epochs | 20 | SGD 在整个训练集上的迭代轮数 |
| lr_all | 0.005 | 学习率，控制参数更新步长 |
| reg_all | 0.02 | L2 正则化系数，越大越防过拟合 |
| biased | true | 是否使用用户/物品偏置项 |

---

### 2.3 UserKNN（基于用户的协同过滤）

**原理**

核心思想：**"跟你品味相似的人喜欢的东西，你大概率也会喜欢。"**

```
步骤 1 — 计算用户相似度
    从 user-item 评分矩阵中取两个用户 u 和 v 共同评过分的电影集合 I_uv
    用 Pearson 相关系数（减均值去偏置后的余弦相似度）衡量相似性：
    
    sim(u,v) = Σ(rᵤᵢ - r̄ᵤ)(rᵥᵢ - r̄ᵥ) / √[Σ(rᵤᵢ - r̄ᵤ)² × Σ(rᵥᵢ - r̄ᵥ)²]
    
    使用 Pearson 而非普通余弦相似度的原因：不同用户的评分尺度差异很大
    — 有人出手就是 5 星，有人给 3 星已经算很高。减去各自由均值消除了这种偏差。

步骤 2 — 选择邻居
    取与目标用户 u 相似度最高的 K 个用户作为邻居（K=40）

步骤 3 — 预测评分
    对目标电影 i，加权平均邻居们对 i 的评分偏差：
    
    r̂(u,i) = r̄ᵤ + Σ[sim(u,v) × (rᵥᵢ - r̄ᵥ)] / Σ|sim(u,v)|

    即"邻居们觉得这部电影（比其他电影）好多少"的加权平均，
    再加回目标用户自己的平均分。
```

**实现**：`src/models/collaborative.py` — `KNNRecommender(user_based=True)`

---

### 2.4 ItemKNN（基于物品的协同过滤）

**原理**

核心思想：**"你喜欢 A，B 和 A 很像，那你大概率也喜欢 B"**——这是 Amazon 在 2003 年提出的经典算法，至今仍广泛使用。

```
步骤 1 — 计算物品相似度
    两部电影 i 和 j 的相似度 = 看过它们的用户评分模式的 Pearson 相关系数
    "如果所有给《肖申克的救赎》打高分的人也倾向于给《绿里奇迹》打高分，
     那这两部电影就是相似的"

步骤 2 — 预测评分
    对目标电影 i，找到用户 u 评过分的电影中与 i 最相似的 K 部，
    加权平均：
    
    r̂(u,i) = r̄ᵢ + Σ[sim(i,j) × (rᵤⱼ - r̄ⱼ)] / Σ|sim(i,j)|

步骤 3 — 相似电影检索
    直接通过已储存好的相似度矩阵做 KNN 查询
```

**UserKNN vs ItemKNN 对比**

| | UserKNN | ItemKNN |
|--|---------|---------|
| 计算对象 | 用户-用户相似度 | 物品-物品相似度 |
| 可解释性 | "和你口味相似的也喜欢" | "因为你看过 X，推荐 Y" |
| 稳定性 | 新用户加入需重算全部相似度 | 物品相似度相对稳定，可预计算 |
| 适用场景 | 用户远少于物品时 | 物品远少于用户时（如电商） |
| 新用户 | 冷启动严重（没有评分无法找邻居） | 可以用热门物品兜底 |
| 新物品 | 可以被现有用户推荐出去 | 冷启动严重（没人评过无法算相似度） |

**实现**：`src/models/collaborative.py` — `KNNRecommender(user_based=False)`

---

### 2.5 模型基类设计

四个模型都继承自 `BaseRecommender`（`src/models/base.py`），统一了接口：

```python
class BaseRecommender(ABC):
    def fit(ratings_df, movies_df=None)     # 训练
    def predict(user_id, movie_id) -> float  # 预测单个评分
    def recommend(user_id, n=10) -> list     # 生成 Top-N 推荐
    def get_similar_movies(movie_id, n=10)   # 相似电影（KNN/SVD 实现）
```

所有模型内部维护 `user_id_to_idx` / `movie_id_to_idx` 索引映射，以及 `all_movie_ids` 候选集。基类还提供了 `recommend_with_details`（附带电影名称/类型等详情）和 `batch_recommend`（批量推荐）等便捷方法。

---

## 3. 特征工程

`src/features/build_features.py` 提供两个维度的特征提取：

**电影特征**
- `genre_*`：类型 Multi-Hot（剧情、喜剧、动作...共 21 类）
- `year` / `decade`：发行年份 / 年代区间（1990s, 2000s...）
- `douban_score`：豆瓣评分（缺失值填均值）
- `douban_votes_log`：豆瓣评分数 log 归一化
- `mins`：电影时长
- `region_*`：地区 Multi-Hot（中国大陆、美国、日本...）

**用户特征**
- `rating_count`：评分次数
- `rating_mean` / `rating_std`：评分均值与标准差
- `active_days`：评分时间跨度（天）
- `high_ratio` / `low_ratio`：高分(≥4)和低分(≤2)的比例
- `pref_*`：加权类型偏好分布（以用户评分为权重对各电影类型做加权平均）

---

## 4. 评估指标

`src/evaluation/metrics.py` 提供完整的评估体系：

### 评分预测指标（衡量预测精度）

| 指标 | 公式 | 含义 |
|------|------|------|
| RMSE | √(Σ(ŷ - y)² / N) | 均方根误差，对大误差惩罚更重 |
| MAE | Σ\|ŷ - y\| / N | 平均绝对误差，更直观 |

### 排序质量指标（衡量 Top-K 推荐质量）

以 `rating ≥ 3.5` 作为"喜欢"的判定阈值：

| 指标 | 含义 |
|------|------|
| Precision@K | 推荐列表中"用户真正喜欢的"占推荐数的比例 |
| Recall@K | "用户真正喜欢的"有多少被推荐出来（覆盖率） |
| NDCG@K | 考虑位置权重的排序质量（排前面的推荐对了贡献更大） |

### 覆盖与多样性

| 指标 | 含义 |
|------|------|
| Catalog Coverage | 推荐覆盖了多少部不同的电影 / 总电影数 |
| Intra-list Diversity | 推荐列表内电影的类型多样性（基于 Jaccard 距离） |

---

## 5. 模型效果对比

在 Top200 数据集上，过滤后（1,280 用户 × 192 电影，13,227 条训练评分 × 2,646 条测试评分），四个模型的对比：

| 模型 | RMSE ↓ | MAE ↓ | P@10 | R@10 | NDCG@10 | Coverage | Diversity |
|------|--------|-------|------|------|---------|----------|-----------|
| **SVD** | **0.8025** | **0.6370** | **0.0227** | 0.0895 | **0.0579** | 0.3177 | 0.7583 |
| Popularity | 0.9556 | 0.7441 | 0.0204 | 0.0895 | 0.0481 | 0.1042 | 0.7863 |
| UserKNN | 0.9136 | 0.7083 | 0.0143 | 0.0581 | 0.0336 | 0.8802 | 0.7822 |
| ItemKNN | 1.0286 | 0.7972 | 0.0123 | 0.0598 | 0.0329 | 0.9896 | 0.8010 |

### 关键发现

1. **SVD 全面最优**：在 RMSE 和所有 Top-K 指标上领先，仅 50 维隐向量就抓住了足够多的评分模式。且提供了可用的相似电影检索。

2. **Popularity 作为基线并不差**：RMSE 比 SVD 差 19%，但在 Recall@10 上与 SVD 持平。这说明 Top200 榜单上的电影本身都是高分热门，popularity bias 在这个数据集上已经很强。但其 Coverage 仅 10%，多样性差。

3. **UserKNN > ItemKNN**：UserKNN 的 RMSE 比 ItemKNN 低 11% 以上。这是因为在 192 部电影 × 1280 用户的数据上，用户维度的信号比物品维度更可靠——每个用户的评分行为相对稳定，而 192 部电影之间可能过于相似（都是高分片）。

4. **KNN 覆盖率虚高**：UserKNN 和 ItemKNN 覆盖率接近 100%，但推荐质量并不好。因为它们倾向于把"冷门但碰巧相似"的电影推荐出来，而不是真正的好电影。高覆盖率 ≠ 好推荐。

5. **模型选择建议**：当前阶段 SVD 是主推荐引擎的首选；Popularity 作为冷启动用户的兜底策略；ItemKNN 的 `get_similar_movies` 在"相似电影"场景下有用。

---

## 6. 项目结构

```
cinemagic/
├── requirements.txt                # Python 依赖
├── config/
│   └── config.yaml                 # 全局配置（路径、模型超参、评估设置）
├── src/
│   ├── config.py                   # 配置加载器（YAML → dict，单例模式）
│   ├── data/
│   │   ├── loader.py               # 数据加载 + 查询接口
│   │   └── preprocess.py           # 过滤、划分、构建矩阵
│   ├── features/
│   │   └── build_features.py       # 电影/用户特征工程
│   ├── models/
│   │   ├── base.py                 # BaseRecommender 抽象基类
│   │   ├── popularity.py           # PopularityRecommender
│   │   └── collaborative.py        # SVD / UserKNN / ItemKNN
│   └── evaluation/
│       └── metrics.py              # 评估指标 + 综合评估 + 模型对比
├── scripts/
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   └── recommend.py                # 命令行推荐入口
├── notebooks/
│   └── 01_eda.ipynb                # 数据探索 Notebook
├── models/                         # 训练好的 .pkl 模型文件
└── output/                         # 评估结果 .json / .csv
```

---

## 7. 使用方式

```bash
# 训练全部模型
python scripts/train.py --model all

# 评估对比
python scripts/evaluate.py --model all --topk 5,10,20

# 个性化推荐
python scripts/recommend.py --user <user_id> --model svd --topk 10

# 相似电影
python scripts/recommend.py --similar-to 1292052 --model item_knn

# 热门推荐（按类型筛选）
python scripts/recommend.py --hot --genre 动画 --topk 20

# 查看用户历史
python scripts/recommend.py --user <user_id> --history

# 电影详情
python scripts/recommend.py --movie 1292052

# 数据探索
jupyter notebook notebooks/01_eda.ipynb
```

---

## 8. 后续计划（Phase 2+）

1. **LightGCN** — 图神经网络协同过滤，当前稀疏场景 SOTA，替代 SVD 成为主召回模型
2. **内容召回** — 用中文 BERT（如 BGE）对 STORYLINE/TAGS 编码，通过 FAISS 做语义相似检索
3. **DeepFM 精排** — 在召回候选集基础上用 FM+DNN 联合排序
4. **全量数据扩展** — 从 Top200 扩展到 14 万电影 × 60 万用户
5. **在线服务** — FastAPI 后端 + Streamlit 前端
