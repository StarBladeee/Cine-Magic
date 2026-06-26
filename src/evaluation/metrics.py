"""
CineMagic 模型评估模块

提供评分预测指标（RMSE/MAE）和排序质量指标（Precision@K/Recall@K/NDCG@K）
以及覆盖率、多样性等推荐系统专用指标。
"""

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import List, Dict


# ──────────────────────────────────────────────────────────────
# 评分预测指标
# ──────────────────────────────────────────────────────────────

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """RMSE (均方根误差)"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """MAE (平均绝对误差)"""
    return np.mean(np.abs(y_true - y_pred))


# ──────────────────────────────────────────────────────────────
# 排序质量指标
# ──────────────────────────────────────────────────────────────

def precision_at_k(actual: list[str], predicted: list[str], k: int) -> float:
    """
    Precision@K: 推荐列表中用户真正喜欢的比例。

    Args:
        actual:    用户实际喜欢的电影 ID 列表（测试集中评分 >= threshold 的）
        predicted: 推荐列表（按得分降序，取前 k 个）
        k:         截断值
    """
    actual_set = set(actual)
    if len(predicted) == 0:
        return 0.0
    top_k = predicted[:k]
    return len([item for item in top_k if item in actual_set]) / min(k, len(top_k))


def recall_at_k(actual: list[str], predicted: list[str], k: int) -> float:
    """
    Recall@K: 用户真正喜欢的电影中有多少被推荐出来。

    Args:
        actual:    用户实际喜欢的电影 ID 列表
        predicted: 推荐列表
        k:         截断值
    """
    actual_set = set(actual)
    if len(actual_set) == 0:
        return 0.0
    top_k = predicted[:k]
    return len([item for item in top_k if item in actual_set]) / len(actual_set)


def ndcg_at_k(actual: list[str], predicted: list[str], k: int) -> float:
    """
    NDCG@K (Normalized Discounted Cumulative Gain):
    考虑位置权重的排序质量指标。

    Args:
        actual:    用户实际喜欢并按相关性排好序的电影 ID 列表
        predicted: 推荐列表
        k:         截断值
    """
    actual_set = set(actual)
    top_k = predicted[:k]

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        if item in actual_set:
            dcg += 1.0 / np.log2(i + 2)  # i+2 因为 i 从 0 开始, log(rank+1)=log(i+2)

    # IDCG (理想情况：所有相关物品排在前面)
    ideal_hits = min(len(actual_set), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0


# ──────────────────────────────────────────────────────────────
# 覆盖率与多样性
# ──────────────────────────────────────────────────────────────

def catalog_coverage(recommended_items: list[str], all_items: list[str]) -> float:
    """推荐覆盖的电影占全部电影的比例"""
    if len(all_items) == 0:
        return 0.0
    return len(set(recommended_items)) / len(set(all_items))


def intra_list_diversity(
    recommended_lists: list[list[str]],
    item_features: pd.DataFrame,
) -> float:
    """
    推荐列表内的平均多样性（基于类型 Jaccard 距离）。

    Args:
        recommended_lists: 每个用户的推荐列表 [[movie_id, ...], ...]
        item_features:     电影特征 DataFrame (index=movie_id, 至少含类型列)
    """
    # 取类型列
    genre_cols = [c for c in item_features.columns if c.startswith("genre_")]
    if not genre_cols:
        return 0.0

    diversities = []
    for rec_list in recommended_lists:
        if len(rec_list) < 2:
            diversities.append(0.0)
            continue
        valid = [m for m in rec_list if m in item_features.index]
        if len(valid) < 2:
            diversities.append(0.0)
            continue

        genre_vecs = item_features.loc[valid, genre_cols].values
        total = 0.0
        count = 0
        for i in range(len(valid)):
            for j in range(i + 1, len(valid)):
                intersection = np.sum(genre_vecs[i] * genre_vecs[j])
                union = np.sum(np.clip(genre_vecs[i] + genre_vecs[j], 0, 1))
                if union > 0:
                    total += 1.0 - intersection / union
                count += 1
        diversities.append(total / count if count > 0 else 0.0)

    return np.mean(diversities) if diversities else 0.0


# ──────────────────────────────────────────────────────────────
# 综合评估
# ──────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame | None = None,
    k_values: list[int] | None = None,
    rating_threshold: float = 3.5,
    verbose: bool = True,
) -> dict:
    """
    对推荐模型进行全面评估。

    Args:
        model:           推荐模型实例（必须实现了 fit/predict/recommend）
        train_df:        训练集评分 DataFrame
        test_df:         测试集评分 DataFrame
        movies_df:       电影元数据（用于多样性计算）
        k_values:        Top-K 指标列表
        rating_threshold: >= 此评分视为"喜欢"

    Returns:
        评估结果字典
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # 1. 评分预测指标（RMSE/MAE）
    y_true = []
    y_pred = []
    for _, row in test_df.iterrows():
        try:
            pred = model.predict(row["userId"], row["movieId"])
            y_true.append(row["rating"])
            y_pred.append(pred)
        except Exception:
            continue

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    rmse_val = rmse(y_true, y_pred)
    mae_val = mae(y_true, y_pred)

    # 2. 排序质量指标（Precision/Recall/NDCG @K）
    # 为每个用户构建 actual liked set
    user_actual = defaultdict(list)
    for _, row in test_df.iterrows():
        if row["rating"] >= rating_threshold:
            user_actual[row["userId"]].append(row["movieId"])

    # 为每个用户生成推荐
    topk_metrics = {k: {"precision": [], "recall": [], "ndcg": []} for k in k_values}
    user_recs = {}  # 记录每个用户的推荐用于后续 diversity 计算
    all_recommended = set()

    test_users = list(user_actual.keys())
    for user_id in test_users:
        try:
            recs = model.recommend(user_id, n=max(k_values), exclude_seen=True)
            rec_movie_ids = [m for m, _ in recs]
        except Exception:
            rec_movie_ids = model.all_movie_ids[:max(k_values)]

        user_recs[user_id] = rec_movie_ids
        all_recommended.update(rec_movie_ids)
        actual = user_actual.get(user_id, [])

        for k in k_values:
            topk_metrics[k]["precision"].append(precision_at_k(actual, rec_movie_ids, k))
            topk_metrics[k]["recall"].append(recall_at_k(actual, rec_movie_ids, k))
            topk_metrics[k]["ndcg"].append(ndcg_at_k(actual, rec_movie_ids, k))

    for k in k_values:
        for metric in ["precision", "recall", "ndcg"]:
            vals = topk_metrics[k][metric]
            topk_metrics[k][metric] = np.mean(vals) if vals else 0.0

    # 3. 覆盖率
    coverage = catalog_coverage(list(all_recommended), model.all_movie_ids)

    # 4. 多样性
    diversity_val = 0.0
    if movies_df is not None:
        from src.features.build_features import extract_movie_features
        try:
            movie_features = extract_movie_features(movies_df)
            diversity_val = intra_list_diversity(list(user_recs.values()), movie_features)
        except Exception:
            diversity_val = 0.0

    results = {
        "rmse": float(rmse_val),
        "mae": float(mae_val),
        "coverage": float(coverage),
        "diversity": float(diversity_val),
        "topk": {k: {m: float(vals[m]) for m in vals} for k, vals in topk_metrics.items()},
        "num_test_users": len(test_users),
    }

    if verbose:
        _print_results(results)

    return results


def compare_models(
    models: dict,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    movies_df: pd.DataFrame | None = None,
    k_values: list[int] | None = None,
    rating_threshold: float = 3.5,
) -> pd.DataFrame:
    """
    比较多个模型的表现。

    Args:
        models: {模型名称: 模型实例}

    Returns:
        比较结果 DataFrame
    """
    if k_values is None:
        k_values = [5, 10, 20]

    rows = []
    for name, model in models.items():
        result = evaluate_model(
            model, train_df, test_df, movies_df,
            k_values=k_values, rating_threshold=rating_threshold,
            verbose=False,
        )
        row = {
            "model": name,
            "rmse": result["rmse"],
            "mae": result["mae"],
            "coverage": result["coverage"],
            "diversity": result["diversity"],
        }
        for k in k_values:
            row[f"precision@{k}"] = result["topk"][k]["precision"]
            row[f"recall@{k}"] = result["topk"][k]["recall"]
            row[f"ndcg@{k}"] = result["topk"][k]["ndcg"]
        rows.append(row)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────
# 辅助
# ──────────────────────────────────────────────────────────────

def _print_results(results: dict) -> None:
    """打印评估结果"""
    k_values = list(results["topk"].keys())
    print("=" * 60)
    print("Model Evaluation Results")
    print("=" * 60)
    print(f"  Test users: {results['num_test_users']}")
    print(f"  RMSE:       {results['rmse']:.4f}")
    print(f"  MAE:        {results['mae']:.4f}")
    print(f"  Coverage:   {results['coverage']:.4f}")
    print(f"  Diversity:  {results['diversity']:.4f}")
    print(f"  --- Top-K Metrics ---")
    for k in sorted(k_values):
        m = results["topk"][k]
        print(f"  @{k:>2}:  Precision={m['precision']:.4f}  "
              f"Recall={m['recall']:.4f}  NDCG={m['ndcg']:.4f}")
    print("=" * 60)
