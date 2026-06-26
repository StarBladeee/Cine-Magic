"""
协同过滤推荐模型

基于 scikit-surprise 库实现:
- SVD (矩阵分解)
- UserKNN (基于用户的协同过滤)
- ItemKNN (基于物品的协同过滤)

每个模型封装 surprise 算法，统一 fit/predict/recommend 接口。
"""

import pandas as pd
import numpy as np
from typing import List

from surprise import (
    SVD as SurpriseSVD,
    KNNBasic,
    Reader,
    Dataset,
    accuracy,
)
from surprise.model_selection import PredefinedKFold

from src.models.base import BaseRecommender

# ──────────────────────────────────────────────────────────────
# 工具函数：DataFrame <-> surprise Dataset
# ──────────────────────────────────────────────────────────────

_READER = Reader(rating_scale=(1, 5))


def _df_to_surprise(ratings_df: pd.DataFrame) -> Dataset:
    """将评分 DataFrame 转为 surprise Dataset"""
    data = ratings_df[["userId", "movieId", "rating"]].copy()
    return Dataset.load_from_df(data, _READER)


def _build_trainset(ratings_df: pd.DataFrame):
    """构建 surprise Trainset"""
    data = _df_to_surprise(ratings_df)
    return data.build_full_trainset()


# ──────────────────────────────────────────────────────────────
# SVD 矩阵分解
# ──────────────────────────────────────────────────────────────

class SVDRecommender(BaseRecommender):
    """基于 SVD 矩阵分解的推荐模型"""

    def __init__(
        self,
        name: str = "svd",
        n_factors: int = 50,
        n_epochs: int = 20,
        lr_all: float = 0.005,
        reg_all: float = 0.02,
        biased: bool = True,
        random_state: int = 42,
    ):
        super().__init__(name=name)
        self.n_factors = n_factors
        self.n_epochs = n_epochs
        self.lr_all = lr_all
        self.reg_all = reg_all
        self.biased = biased
        self.random_state = random_state
        self.model: SurpriseSVD | None = None
        # 用户已评电影集合，用于推荐时排除
        self._user_seen: dict[str, set[str]] = {}

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """训练 SVD 模型"""
        self._build_index_maps(ratings_df)

        # 记录用户已看
        self._user_seen = {}
        for _, row in ratings_df.iterrows():
            uid, mid = row["userId"], row["movieId"]
            self._user_seen.setdefault(uid, set()).add(mid)

        trainset = _build_trainset(ratings_df)

        self.model = SurpriseSVD(
            n_factors=self.n_factors,
            n_epochs=self.n_epochs,
            lr_all=self.lr_all,
            reg_all=self.reg_all,
            biased=self.biased,
            random_state=self.random_state,
        )
        self.model.fit(trainset)
        self.is_fitted = True

    def predict(self, user_id: str, movie_id: str) -> float:
        """预测用户对电影的评分"""
        self._check_fitted()
        try:
            inner_uid = self.model.trainset.to_inner_uid(user_id)
        except ValueError:
            return self.model.default_prediction().est  # 2.5 (全局均值)
        try:
            inner_iid = self.model.trainset.to_inner_iid(movie_id)
        except ValueError:
            return self.model.default_prediction().est
        return self.model.estimate(inner_uid, inner_iid)

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[str] | None = None,
    ) -> list[tuple[str, float]]:
        """为指定用户推荐电影"""
        self._check_fitted()

        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()

        if candidate_movies is not None:
            candidates = [m for m in candidate_movies if m not in seen]
        else:
            candidates = [m for m in self.all_movie_ids if m not in seen]

        scores = []
        for movie_id in candidates:
            try:
                inner_uid = self.model.trainset.to_inner_uid(user_id)
                inner_iid = self.model.trainset.to_inner_iid(movie_id)
                est = self.model.estimate(inner_uid, inner_iid)
                if isinstance(est, tuple):
                    est = float(est[0])
                scores.append((movie_id, est))
            except Exception:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def get_similar_movies(
        self,
        movie_id: str,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        通过 SVD 隐向量余弦相似度找相似电影。
        SVD 学到每部电影的隐向量 p_i，用 p_i 之间的余弦相似度衡量。
        """
        self._check_fitted()
        try:
            inner_iid = self.model.trainset.to_inner_iid(movie_id)
        except ValueError:
            return []

        target_vec = self.model.qi[inner_iid]

        # 计算与所有电影的余弦相似度
        sims = []
        for other_id in self.all_movie_ids:
            if other_id == movie_id:
                continue
            try:
                other_inner = self.model.trainset.to_inner_iid(other_id)
                other_vec = self.model.qi[other_inner]
                # 余弦相似度
                dot = np.dot(target_vec, other_vec)
                norm = np.linalg.norm(target_vec) * np.linalg.norm(other_vec)
                sim = dot / (norm + 1e-8)
                sims.append((other_id, sim))
            except ValueError:
                continue

        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:n]


# ──────────────────────────────────────────────────────────────
# UserKNN / ItemKNN
# ──────────────────────────────────────────────────────────────

class KNNRecommender(BaseRecommender):
    """
    基于 KNN 的协同过滤推荐模型。

    可通过 user_based 参数切换:
    - user_based=True  → UserKNN（基于用户的协同过滤）
    - user_based=False → ItemKNN（基于物品的协同过滤）
    """

    def __init__(
        self,
        name: str = "knn",
        user_based: bool = True,
        k: int = 40,
        min_k: int = 1,
        sim_name: str = "pearson_baseline",
        shrinkage: int = 100,
        random_state: int = 42,
    ):
        super().__init__(name=name)
        self.user_based = user_based
        self.k = k
        self.min_k = min_k
        self.sim_name = sim_name
        self.shrinkage = shrinkage
        self.random_state = random_state
        self.model: KNNBasic | None = None
        self._user_seen: dict[str, set[str]] = {}

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """训练 KNN 模型"""
        self._build_index_maps(ratings_df)

        self._user_seen = {}
        for _, row in ratings_df.iterrows():
            uid, mid = row["userId"], row["movieId"]
            self._user_seen.setdefault(uid, set()).add(mid)

        trainset = _build_trainset(ratings_df)

        sim_options = {
            "name": self.sim_name,
            "user_based": self.user_based,
            "shrinkage": self.shrinkage,
        }

        self.model = KNNBasic(
            k=self.k,
            min_k=self.min_k,
            sim_options=sim_options,
            random_state=self.random_state,
        )
        self.model.fit(trainset)
        self.is_fitted = True

    def predict(self, user_id: str, movie_id: str) -> float:
        """预测用户对电影的评分"""
        self._check_fitted()
        try:
            inner_uid = self.model.trainset.to_inner_uid(user_id)
        except ValueError:
            return self.model.default_prediction().est
        try:
            inner_iid = self.model.trainset.to_inner_iid(movie_id)
        except ValueError:
            return self.model.default_prediction().est
        result = self.model.estimate(inner_uid, inner_iid)
        # estimate 返回 (est, details) 元组或 float
        if isinstance(result, tuple):
            return float(result[0])
        return float(result)

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[str] | None = None,
    ) -> list[tuple[str, float]]:
        """为指定用户推荐电影"""
        self._check_fitted()

        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()

        if candidate_movies is not None:
            candidates = [m for m in candidate_movies if m not in seen]
        else:
            candidates = [m for m in self.all_movie_ids if m not in seen]

        scores = []
        for movie_id in candidates:
            try:
                inner_uid = self.model.trainset.to_inner_uid(user_id)
                inner_iid = self.model.trainset.to_inner_iid(movie_id)
                est = self.model.estimate(inner_uid, inner_iid)
                if isinstance(est, tuple):
                    est = float(est[0])
                scores.append((movie_id, est))
            except Exception:
                continue

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    def get_similar_movies(
        self,
        movie_id: str,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """
        获取与指定电影相似的其他电影。

        仅 ItemKNN 模式下有效。
        """
        self._check_fitted()

        if self.user_based:
            raise NotImplementedError(
                "UserKNN does not support get_similar_movies. Use ItemKNN instead."
            )

        try:
            inner_iid = self.model.trainset.to_inner_iid(movie_id)
        except ValueError:
            return []

        # surprise 的 KNNBasic 存储了相似度矩阵在 self.model.sim
        neighbors = self.model.get_neighbors(inner_iid, k=n)
        sims = []
        for inner_nbr in neighbors:
            raw_iid = self.model.trainset.to_raw_iid(inner_nbr)
            sim = self.model.sim[inner_iid, inner_nbr]
            sims.append((raw_iid, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:n]


# ──────────────────────────────────────────────────────────────
# 便捷工厂函数
# ──────────────────────────────────────────────────────────────

def create_model(model_type: str, **kwargs) -> BaseRecommender:
    """
    工厂函数：根据名称创建推荐模型。

    Args:
        model_type: "popularity" | "svd" | "user_knn" | "item_knn"
        **kwargs:   传递给模型构造函数的参数

    Returns:
        BaseRecommender 实例
    """
    model_type = model_type.lower()
    if model_type == "popularity":
        return PopularityRecommender(**kwargs)
    elif model_type == "svd":
        return SVDRecommender(**kwargs)
    elif model_type == "user_knn":
        return KNNRecommender(user_based=True, name="user_knn", **kwargs)
    elif model_type == "item_knn":
        return KNNRecommender(user_based=False, name="item_knn", **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                         f"Choose from: popularity, svd, user_knn, item_knn")
