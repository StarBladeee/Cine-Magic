"""
Popularity 基线推荐模型

基于豆瓣评分和评分人数的加权得分推荐热门电影。
支持全局热门和按类型热门两种策略。
"""

import pandas as pd
import numpy as np
from typing import List

from src.models.base import BaseRecommender


class PopularityRecommender(BaseRecommender):
    """
    热门推荐模型。

    不依赖用户个性化信息，对冷启动用户友好，通常作为召回兜底策略。
    """

    def __init__(
        self,
        name: str = "popularity",
        use_douban_score: bool = True,
        use_douban_votes: bool = True,
        score_weight: float = 0.6,
        votes_weight: float = 0.4,
        min_votes: int = 100,
    ):
        super().__init__(name=name)
        self.use_douban_score = use_douban_score
        self.use_douban_votes = use_douban_votes
        self.score_weight = score_weight
        self.votes_weight = votes_weight
        self.min_votes = min_votes
        # 存储计算后的热门得分
        self.pop_scores: pd.Series | None = None

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """
        根据豆瓣评分和评分数计算每部电影的流行度得分。

        得分公式:
            score = score_weight * douban_score_normalized
                  + votes_weight * log_votes_normalized

        同时考虑用户评分均值和评分次数作为修正。
        """
        self._build_index_maps(ratings_df)

        if movies_df is None:
            raise ValueError("PopularityRecommender requires movies_df")

        scores = pd.Series(0.0, index=self.all_movie_ids)

        # 计算每位用户的评分统计
        user_rating_stats = ratings_df.groupby("movieId")["rating"].agg(["mean", "count"])

        for movie_id in self.all_movie_ids:
            s = 0.0
            n_components = 0
            weight_sum = 0.0

            # 豆瓣评分
            if self.use_douban_score and movie_id in movies_df.index:
                d_score = movies_df.at[movie_id, "DOUBAN_SCORE"]
                if d_score and d_score > 0:
                    s += self.score_weight * (d_score / 10.0)  # 归一化到 [0, 1]
                    weight_sum += self.score_weight
                    n_components += 1

            # 豆瓣评分数
            if self.use_douban_votes and movie_id in movies_df.index:
                d_votes = movies_df.at[movie_id, "DOUBAN_VOTES"]
                if d_votes and d_votes > 0:
                    log_votes = np.log1p(d_votes)
                    s += self.votes_weight * min(log_votes / 15.0, 1.0)  # 归一化
                    weight_sum += self.votes_weight
                    n_components += 1

            # 用户评分均值
            if movie_id in user_rating_stats.index:
                u_mean = user_rating_stats.at[movie_id, "mean"]
                u_count = user_rating_stats.at[movie_id, "count"]
                s += 0.3 * (u_mean / 5.0)
                s += 0.1 * min(np.log1p(u_count) / 5.0, 1.0)
                weight_sum += 0.4

            scores[movie_id] = s / (weight_sum if weight_sum > 0 else 1.0)

        self.pop_scores = scores.sort_values(ascending=False)
        self.is_fitted = True

    def predict(self, user_id: str, movie_id: str) -> float:
        """返回标准化的流行度得分（映射到 1-5 分区间）"""
        self._check_fitted()
        if movie_id not in self.pop_scores.index:
            return 2.5  # 未知电影返回中间分
        # 将 [0,1] 得分线性映射到 [1, 5]
        score = self.pop_scores[movie_id]
        return 1.0 + score * 4.0

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[str] | None = None,
    ) -> list[tuple[str, float]]:
        """返回流行度最高的 N 部电影"""
        self._check_fitted()

        # 确定候选集
        if candidate_movies is not None:
            candidates = [m for m in candidate_movies if m in self.pop_scores.index]
        else:
            candidates = self.pop_scores.index.tolist()

        # 排除已看
        if exclude_seen and user_id in self.idx_to_user_id.values():
            pass  # 本模型不维护用户历史，默认不排除

        # 按得分排序
        top = []
        for movie_id in candidates:
            score = self.pop_scores.get(movie_id, 0.0)
            top.append((movie_id, score))
            if len(top) >= n * 2:  # oversample 以便过滤
                break

        top.sort(key=lambda x: x[1], reverse=True)
        return top[:n]

    def get_popular_by_genre(
        self,
        genre: str,
        movies_df: pd.DataFrame,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        获取指定类型的热门电影。

        Args:
            genre:     电影类型，如 "剧情"、"喜剧"
            movies_df: 电影元数据
            n:         返回数量

        Returns:
            [(movie_id, score), ...]
        """
        self._check_fitted()
        genre_movies = movies_df[movies_df["GENRES"].str.contains(genre, na=False)]
        candidates = [mid for mid in genre_movies.index if mid in self.pop_scores.index]
        recs = [(mid, self.pop_scores[mid]) for mid in candidates]
        recs.sort(key=lambda x: x[1], reverse=True)
        return recs[:n]
