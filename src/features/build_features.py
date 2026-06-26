"""
CineMagic 特征工程模块

从电影元数据和用户评分历史中提取可用于推荐模型的特征。
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# ──────────────────────────────────────────────────────────────
# 电影特征
# ──────────────────────────────────────────────────────────────

def extract_movie_features(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    从电影元数据中提取结构化特征。

    包含:
    - 类型 Multi-Hot (所有出现的类型)
    - 年份 / 年代区间
    - 豆瓣评分
    - 豆瓣评分数（log 归一化）
    - 电影时长
    - 地区

    Returns:
        DataFrame，以 movie_id 为索引，列为各类特征
    """
    features = pd.DataFrame(index=movies_df.index)

    # 1. 类型 Multi-Hot
    genre_mlb = MultiLabelBinarizer()
    genres_list = movies_df["GENRES"].fillna("").apply(lambda s: [g.strip() for g in s.split("/") if g.strip()])
    genre_encoded = genre_mlb.fit_transform(genres_list)
    genre_cols = [f"genre_{g}" for g in genre_mlb.classes_]
    genre_df = pd.DataFrame(genre_encoded, index=movies_df.index, columns=genre_cols)
    features = pd.concat([features, genre_df], axis=1)

    # 2. 年份
    features["year"] = movies_df["YEAR"].apply(lambda y: y if pd.notna(y) and 1900 < y < 2030 else None)
    features["decade"] = features["year"].apply(_year_to_decade)

    # 3. 豆瓣评分（缺失填均值）
    scores = movies_df["DOUBAN_SCORE"].replace(0.0, np.nan)
    features["douban_score"] = scores.fillna(scores.mean() if scores.notna().any() else 0.0)

    # 4. 豆瓣评分数（log归一化）
    votes = movies_df["DOUBAN_VOTES"].replace(0.0, np.nan)
    if votes.notna().any():
        vote_median = votes.median()
        votes = votes.fillna(vote_median)
    else:
        votes = votes.fillna(1)
    features["douban_votes_log"] = np.log1p(votes)

    # 5. 时长
    features["mins"] = movies_df["MINS"].replace(0.0, np.nan).fillna(movies_df["MINS"].median() if not movies_df["MINS"].empty else 90.0)

    # 6. 地区 One-Hot
    region_mlb = MultiLabelBinarizer()
    regions_list = movies_df["REGIONS"].fillna("").apply(lambda s: [r.strip() for r in s.split("/") if r.strip()])
    region_encoded = region_mlb.fit_transform(regions_list)
    region_cols = [f"region_{r}" for r in region_mlb.classes_]
    region_df = pd.DataFrame(region_encoded, index=movies_df.index, columns=region_cols)
    features = pd.concat([features, region_df], axis=1)

    return features


# ──────────────────────────────────────────────────────────────
# 用户特征
# ──────────────────────────────────────────────────────────────

def extract_user_features(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    从用户评分历史中提取用户画像特征。

    包含:
    - 评分次数
    - 评分均值 / 标准差
    - 评分时间跨度（天）
    - 偏好类型分布（加权平均）
    - 高评分比例（rating >= 4）
    - 低评分比例（rating <= 2）

    Args:
        ratings_df: 评分 DataFrame
        movies_df:  电影元数据 DataFrame（用于获取电影类型）

    Returns:
        DataFrame，以 user_id 为索引
    """
    users = ratings_df["userId"].unique()
    features = pd.DataFrame(index=users)

    group = ratings_df.groupby("userId")

    # 基本统计
    features["rating_count"] = group["rating"].count()
    features["rating_mean"] = group["rating"].mean()
    features["rating_std"] = group["rating"].std().fillna(0.0)

    # 时间跨度
    if ratings_df["timestamp"].notna().any():
        time_span = group["timestamp"].apply(lambda x: (x.max() - x.min()).days if len(x) > 1 else 0)
        features["active_days"] = time_span
    else:
        features["active_days"] = 0

    # 高/低评分比例
    features["high_ratio"] = group["rating"].apply(lambda x: (x >= 4).sum() / len(x))
    features["low_ratio"] = group["rating"].apply(lambda x: (x <= 2).sum() / len(x))

    # 偏好类型分布
    # 构建电影类型 Multi-Hot
    genre_mlb = MultiLabelBinarizer()
    genres_list = movies_df["GENRES"].fillna("").apply(lambda s: [g.strip() for g in s.split("/") if g.strip()])
    genre_encoded = genre_mlb.fit_transform(genres_list)
    genre_df = pd.DataFrame(genre_encoded, index=movies_df.index, columns=genre_mlb.classes_)

    # 对每个用户，计算其看过电影的加权平均类型分布（以评分为权重）
    for genre in genre_mlb.classes_:
        features[f"pref_{genre}"] = 0.0

    for user_id in users:
        user_ratings = ratings_df[ratings_df["userId"] == user_id]
        user_movies = user_ratings["movieId"].tolist()
        valid_movies = [m for m in user_movies if m in genre_df.index]

        if not valid_movies:
            continue

        weights = user_ratings[user_ratings["movieId"].isin(valid_movies)].set_index("movieId")["rating"]
        movie_genres = genre_df.loc[valid_movies]

        # 加权平均
        weighted = movie_genres.multiply(weights, axis=0).sum()
        total_weight = weights.sum()
        if total_weight > 0:
            user_prefs = weighted / total_weight
            for genre in genre_mlb.classes_:
                if genre in user_prefs:
                    features.at[user_id, f"pref_{genre}"] = user_prefs[genre]

    return features


# ──────────────────────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────────────────────

def _year_to_decade(year: float | None) -> str | None:
    """将年份转为年代区间，如 1994 -> '1990s'"""
    if pd.isna(year) or year is None:
        return None
    year = int(year)
    decade = (year // 10) * 10
    return f"{decade}s"


def get_genre_list(movies_df: pd.DataFrame) -> list[str]:
    """获取所有出现的电影类型列表"""
    all_genres = set()
    for genres_str in movies_df["GENRES"].dropna():
        for g in genres_str.split("/"):
            g = g.strip()
            if g:
                all_genres.add(g)
    return sorted(all_genres)
