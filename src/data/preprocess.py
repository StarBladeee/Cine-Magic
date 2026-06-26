"""
CineMagic 数据预处理模块

提供数据清洗、过滤稀疏用户/物品、训练/测试集划分、用户-物品矩阵构建等功能。
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split as skl_train_test_split


def filter_sparse(
    ratings_df: pd.DataFrame,
    min_user_ratings: int = 5,
    min_item_ratings: int = 5,
) -> pd.DataFrame:
    """
    过滤评分过少的用户和电影。

    Args:
        ratings_df:        评分 DataFrame (userId, movieId, rating)
        min_user_ratings:  用户最少评分次数
        min_item_ratings:  电影最少被评次数

    Returns:
        过滤后的评分 DataFrame
    """
    df = ratings_df.copy()

    while True:
        n_start = len(df)

        # 过滤电影
        item_counts = df.groupby("movieId").size()
        valid_items = item_counts[item_counts >= min_item_ratings].index
        df = df[df["movieId"].isin(valid_items)]

        # 过滤用户
        user_counts = df.groupby("userId").size()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df["userId"].isin(valid_users)]

        if len(df) == n_start:
            break

    return df


def train_test_split(
    ratings_df: pd.DataFrame,
    test_size: float = 0.2,
    method: str = "random",
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    划分训练集和测试集。

    Args:
        ratings_df: 评分 DataFrame
        test_size:  测试集比例
        method:     "random" — 随机划分; "time" — 按时间划分（每个用户最近的评分归测试集）
        random_seed: 随机种子

    Returns:
        (train_df, test_df)
    """
    if method == "time":
        df = ratings_df.sort_values("timestamp")
        test_rows = []
        for user_id, group in df.groupby("userId"):
            n_test = max(1, int(len(group) * test_size))
            test_rows.append(group.iloc[-n_test:])
        test_df = pd.concat(test_rows, ignore_index=True)
        test_idx = set(test_df.index)
        train_df = ratings_df[~ratings_df.index.isin(test_idx)].copy()
        return train_df, test_df

    elif method == "random":
        return skl_train_test_split(
            ratings_df,
            test_size=test_size,
            random_state=random_seed,
            stratify=None,
        )

    else:
        raise ValueError(f"Unknown split method: {method}. Use 'random' or 'time'.")


def build_user_item_matrix(
    ratings_df: pd.DataFrame,
    normalize: bool = False,
) -> tuple[csr_matrix, dict, dict]:
    """
    构建用户-物品稀疏评分矩阵。

    Args:
        ratings_df: 评分 DataFrame (userId, movieId, rating)
        normalize:  是否按用户去均值（每个评分减去该用户平均评分）

    Returns:
        (matrix, user_id_to_idx, movie_id_to_idx)
        matrix:         CSR 稀疏矩阵，shape=(n_users, n_items)
        user_id_to_idx: {user_id: row_index}
        movie_id_to_idx:{movie_id: col_index}
    """
    users = ratings_df["userId"].unique()
    items = ratings_df["movieId"].unique()

    user_id_to_idx = {uid: i for i, uid in enumerate(users)}
    movie_id_to_idx = {mid: i for i, mid in enumerate(items)}

    rows = [user_id_to_idx[uid] for uid in ratings_df["userId"]]
    cols = [movie_id_to_idx[mid] for mid in ratings_df["movieId"]]
    data = ratings_df["rating"].values.astype(np.float64)

    matrix = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    if normalize:
        # 按用户去均值
        row_means = matrix.mean(axis=1).A1
        row_means = np.where(np.isnan(row_means), 0.0, row_means)
        # 对非零元素去均值
        matrix = matrix.tocoo()
        norm_data = data - row_means[rows]
        matrix = csr_matrix((norm_data, (rows, cols)), shape=(len(users), len(items)))

    return matrix, user_id_to_idx, movie_id_to_idx


def get_popularity_baselines(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算电影的流行度基线得分。

    结合用户评分均值、豆瓣评分、豆瓣评分人数。

    Args:
        ratings_df: 评分 DataFrame
        movies_df:  电影元数据 DataFrame

    Returns:
        DataFrame 以 movie_id 为索引，列: avg_rating, user_count, douban_score, douban_votes, pop_score
    """
    # 用户评分统计
    rating_stats = ratings_df.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        user_count=("rating", "count"),
    )

    # 合并豆瓣数据
    result = movies_df[["DOUBAN_SCORE", "DOUBAN_VOTES"]].copy()
    result.columns = ["douban_score", "douban_votes"]
    result = rating_stats.join(result, how="left")

    # 缺失值处理
    result["douban_score"] = result["douban_score"].fillna(0)
    result["douban_votes"] = result["douban_votes"].fillna(0)

    return result
