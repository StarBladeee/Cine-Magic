"""
CineMagic 数据加载器

负责从CSV文件加载电影元数据和评分交互数据，并提供便捷查询接口。
"""

import pandas as pd
from typing import List


def load_movies(path: str) -> pd.DataFrame:
    """
    加载电影元数据。

    列包括: MOVIE_ID, NAME, ALIAS, ACTORS, DIRECTORS, DOUBAN_SCORE, DOUBAN_VOTES,
            GENRES, LANGUAGES, MINS, REGIONS, RELEASE_DATE, STORYLINE, TAGS, YEAR,
            ACTOR_IDS, DIRECTOR_IDS 等

    Args:
        path: movies CSV 文件路径

    Returns:
        DataFrame，以 MOVIE_ID 为索引
    """
    dtype = {
        "MOVIE_ID": str,
        "NAME": str,
        "ALIAS": str,
        "COVER": str,
        "DOUBAN_SCORE": float,
        "DOUBAN_VOTES": float,
        "GENRES": str,
        "LANGUAGES": str,
        "MINS": float,
        "REGIONS": str,
        "RELEASE_DATE": str,
        "STORYLINE": str,
        "TAGS": str,
        "YEAR": float,
        "ACTORS": str,
        "DIRECTORS": str,
        "ACTOR_IDS": str,
        "DIRECTOR_IDS": str,
        "IMDB_ID": str,
        "OFFICIAL_SITE": str,
        "SLUG": str,
    }

    df = pd.read_csv(
        path,
        dtype=dtype,
        encoding="utf-8-sig",
    )
    df.set_index("MOVIE_ID", inplace=True)
    # 处理年份异常值
    df["YEAR"] = df["YEAR"].apply(lambda y: int(y) if pd.notna(y) and 1900 < y < 2030 else None)
    return df


def load_ratings(path: str) -> pd.DataFrame:
    """
    加载用户评分数据。

    列: ratingId, userId, movieId, rating, timestamp

    Args:
        path: ratings CSV 文件路径

    Returns:
        DataFrame
    """
    dtype = {
        "ratingId": str,
        "userId": str,
        "movieId": str,
        "rating": float,
        "timestamp": str,
    }
    df = pd.read_csv(
        path,
        dtype=dtype,
        encoding="utf-8-sig",
    )
    # 解析时间戳（格式: "2018-09-05 19:42:07"）
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    return df


def load_top200_data(movies_path: str, ratings_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    一次性加载 top200 数据集。

    Args:
        movies_path:  top200.csv 路径
        ratings_path: top200_ratings.csv 路径

    Returns:
        (movies_df, ratings_df)
    """
    movies = load_movies(movies_path)
    ratings = load_ratings(ratings_path)
    return movies, ratings


def get_movie_by_id(movies_df: pd.DataFrame, movie_id: str) -> dict | None:
    """根据 movie_id 返回电影信息字典，不存在则返回 None"""
    if movie_id not in movies_df.index:
        return None
    row = movies_df.loc[movie_id]
    return row.to_dict()


def get_movies_by_ids(movies_df: pd.DataFrame, movie_ids: List[str]) -> pd.DataFrame:
    """根据 movie_id 列表返回对应的电影子集 DataFrame"""
    valid_ids = [mid for mid in movie_ids if mid in movies_df.index]
    return movies_df.loc[valid_ids]


def get_user_history(ratings_df: pd.DataFrame, user_id: str) -> pd.DataFrame:
    """
    获取指定用户的观影历史（按时间降序）。

    Args:
        ratings_df: 评分 DataFrame
        user_id:    用户 ID

    Returns:
        该用户评过分的记录 DataFrame（按时间降序）
    """
    history = ratings_df[ratings_df["userId"] == user_id].copy()
    return history.sort_values("timestamp", ascending=False, na_position="last")


def get_rating_stats(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    获取电影的评分统计信息。

    Returns:
        DataFrame，列: movie_id, douban_score, douban_votes, avg_user_rating, user_rating_count
    """
    stats = movies_df[["DOUBAN_SCORE", "DOUBAN_VOTES"]].copy()
    stats.rename(columns={"DOUBAN_SCORE": "douban_score", "DOUBAN_VOTES": "douban_votes"}, inplace=True)
    return stats
