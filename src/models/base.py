"""
CineMagic 推荐模型抽象基类

所有推荐模型必须实现 fit / predict / recommend 三个方法。
"""

from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import numpy as np


class BaseRecommender(ABC):
    """推荐模型抽象基类"""

    def __init__(self, name: str = "base"):
        self.name = name
        self.is_fitted = False
        # 子类训练后应填充这些索引映射
        self.user_id_to_idx: dict = {}
        self.movie_id_to_idx: dict = {}
        self.idx_to_user_id: dict = {}
        self.idx_to_movie_id: dict = {}
        self.all_movie_ids: List[str] = []

    @abstractmethod
    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """
        训练模型。

        Args:
            ratings_df: 评分 DataFrame (userId, movieId, rating)
            movies_df:  电影元数据 DataFrame（可选，部分模型需要）
        """
        ...

    @abstractmethod
    def predict(self, user_id: str, movie_id: str) -> float:
        """
        预测单个用户对单个电影的评分。

        Args:
            user_id:  用户 ID
            movie_id: 电影 ID

        Returns:
            预测评分 (1-5)
        """
        ...

    @abstractmethod
    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        为指定用户生成推荐列表。

        Args:
            user_id:          用户 ID
            n:                推荐数量
            exclude_seen:     是否排除已看过的电影
            candidate_movies: 候选电影列表（默认全部）

        Returns:
            [(movie_id, predicted_score), ...] 列表，按预测分数降序
        """
        ...

    def recommend_with_details(
        self,
        user_id: str,
        movies_df: pd.DataFrame,
        n: int = 10,
        exclude_seen: bool = True,
    ) -> list[dict]:
        """
        推荐并附带电影详情（用于展示）。

        Args:
            user_id:   用户 ID
            movies_df: 电影元数据 DataFrame
            n:         推荐数量

        Returns:
            [{movie_id, name, genres, douban_score, predicted_score, ...}, ...]
        """
        recs = self.recommend(user_id, n=n, exclude_seen=exclude_seen)
        results = []
        for movie_id, pred_score in recs:
            info = self._get_movie_info(movies_df, movie_id)
            info["predicted_score"] = round(pred_score, 3)
            results.append(info)
        return results

    def batch_recommend(
        self,
        user_ids: List[str],
        n: int = 10,
        exclude_seen: bool = True,
    ) -> dict[str, list[tuple[str, float]]]:
        """
        批量为多个用户生成推荐。

        Returns:
            {user_id: [(movie_id, score), ...], ...}
        """
        return{
            uid: self.recommend(uid, n=n, exclude_seen=exclude_seen)
            for uid in user_ids
        }

    def get_similar_movies(
        self,
        movie_id: str,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        获取与指定电影相似的其他电影（基于模型内部相似度）。

        默认实现：抛出 NotImplementedError，子类可覆盖。
        """
        raise NotImplementedError(
            f"{self.name} does not support get_similar_movies"
        )

    # ── 内部辅助 ──

    def _get_movie_info(self, movies_df: pd.DataFrame, movie_id: str) -> dict:
        """从 movies_df 中提取单部电影的基础信息"""
        if movie_id in movies_df.index:
            row = movies_df.loc[movie_id]
            return {
                "movie_id": movie_id,
                "name": row.get("NAME", ""),
                "genres": row.get("GENRES", ""),
                "douban_score": row.get("DOUBAN_SCORE", 0.0),
                "douban_votes": row.get("DOUBAN_VOTES", 0.0),
                "year": row.get("YEAR", None),
                "directors": row.get("DIRECTORS", ""),
                "actors": row.get("ACTORS", ""),
            }
        return {"movie_id": movie_id, "name": "Unknown"}

    def _check_fitted(self) -> None:
        """确保模型已训练，否则抛出异常"""
        if not self.is_fitted:
            raise RuntimeError(f"Model '{self.name}' is not fitted yet. Call fit() first.")

    def _build_index_maps(
        self,
        ratings_df: pd.DataFrame,
    ) -> None:
        """构建 user/movie ID 与内部索引的映射"""
        users = ratings_df["userId"].unique()
        movies = ratings_df["movieId"].unique()
        self.user_id_to_idx = {uid: i for i, uid in enumerate(users)}
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(movies)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}
        self.all_movie_ids = list(movies)
