"""
DeepFM 精排模型

基于 Guo et al. (IJCAI 2017) "DeepFM: A Factorization-Machine based Neural Network
for CTR Prediction" 的 PyTorch 实现。

架构:
    ŷ = sigmoid(y_FM + y_DNN)

    FM 部分:
        - 一阶: 稀疏特征 Embedding 的线性加权和
        - 二阶: 所有特征 Embedding 两两内积求和（FM 交叉）
        - 稠密特征: 直接过 Linear 层

    DNN 部分:
        拼接所有 Embedding → MLP → 标量输出

输入特征:
    - Sparse IDs: user_id, movie_id（通过 Embedding 学习）
    - Multi-Hot: movie_genres（多个类型 Embedding 求和池化）
    - Continuous: douban_score, year, user_rating_mean, user_high_ratio 等（通过 FC 映射后拼接）

训练目标: Binary Cross-Entropy
    正样本 = rating >= threshold_high (默认 4.0)
    负样本 = rating <= threshold_low (默认 2.0)

References:
    https://arxiv.org/abs/1703.04247
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import List, Dict, Optional
from collections import defaultdict

from src.models.base import BaseRecommender


# ─────────────────────────────────────────────────────────
# 特征工程（DeepFM 专用）
# ─────────────────────────────────────────────────────────

class FeatureProcessor:
    """
    DeepFM 特征处理器。

    负责将原始 DataFrame 转为模型可用的特征张量。
    区分三种特征:
    - sparse: 单值离散特征（user_id, movie_id）
    - multi_hot: 多值离散特征（genres）
    - continuous: 连续数值特征
    """

    def __init__(self):
        # 特征维度和词表
        self.sparse_feats: List[str] = []
        self.sparse_vocab_sizes: Dict[str, int] = {}
        self.sparse_maps: Dict[str, dict] = {}  # {feat_name: {value: idx}}

        self.multi_hot_feats: List[str] = []
        self.multi_hot_vocab_sizes: Dict[str, int] = {}
        self.multi_hot_maps: Dict[str, dict] = {}

        self.cont_feats: List[str] = []

        self.feat_dim = 0  # 特征总数（用于 FM）

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
    ) -> None:
        """
        根据数据统计确定所有特征的词表大小。
        """
        # ── Sparse 特征 ──
        self.sparse_feats = ["user_id", "movie_id"]
        self.sparse_maps["user_id"] = {
            uid: i for i, uid in enumerate(ratings_df["userId"].unique())
        }
        self.sparse_maps["movie_id"] = {
            mid: i for i, mid in enumerate(movies_df.index)
        }
        self.sparse_vocab_sizes = {
            "user_id": len(self.sparse_maps["user_id"]),
            "movie_id": len(self.sparse_maps["movie_id"]),
        }

        # ── Multi-Hot 特征（电影类型）──
        all_genres = set()
        for g_str in movies_df["GENRES"].fillna(""):
            for g in g_str.split("/"):
                g = g.strip()
                if g:
                    all_genres.add(g)
        genre_list = sorted(all_genres)

        self.multi_hot_feats = ["movie_genres"]
        self.multi_hot_maps["movie_genres"] = {
            g: i for i, g in enumerate(genre_list)
        }
        self.multi_hot_vocab_sizes["movie_genres"] = len(genre_list)

        # ── Continuous 特征 ──
        self.cont_feats = [
            "douban_score",
            "douban_votes_log",
            "year",
            "user_rating_mean",
            "user_rating_std",
            "user_high_ratio",
            "user_low_ratio",
        ]

        # 计算总特征数（FM 视角）
        self.n_sparse = len(self.sparse_feats)
        self.n_multi_hot = len(self.multi_hot_feats)
        self.n_cont = len(self.cont_feats)
        self.n_fields = self.n_sparse + self.n_multi_hot + self.n_cont
        self.feat_dim = (
            sum(self.sparse_vocab_sizes.values()) +
            sum(self.multi_hot_vocab_sizes.values()) +
            len(self.cont_feats)
        )

    def transform(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
    ) -> Dict[str, torch.Tensor]:
        """
        将评分数据转为特征张量。

        Returns:
            dict with keys:
                sparse_indices: (n_samples, n_sparse_feats)
                multi_hot_indices: (n_samples, n_multi_hot_feats, max_len)
                multi_hot_mask: (n_samples, n_multi_hot_feats, max_len)
                continuous: (n_samples, n_cont_feats)
        """
        n = len(ratings_df)
        n_sparse = len(self.sparse_feats)
        n_mh = len(self.multi_hot_feats)
        n_cont = len(self.cont_feats)

        sparse_indices = np.zeros((n, n_sparse), dtype=np.int64)
        multi_hot_indices_list = []  # list of arrays, each (n, max_len)
        multi_hot_mask_list = []
        continuous = np.zeros((n, n_cont), dtype=np.float32)

        # 预计算电影类型
        movie_genre_idx: Dict[str, List[int]] = {}
        for mid, row in movies_df.iterrows():
            genres = []
            g_str = row.get("GENRES", "")
            if pd.notna(g_str) and g_str:
                for g in g_str.split("/"):
                    g = g.strip()
                    if g in self.multi_hot_maps.get("movie_genres", {}):
                        genres.append(self.multi_hot_maps["movie_genres"][g])
            movie_genre_idx[mid] = genres or [0]  # fallback

        # 用户评分统计
        user_stats = self._compute_user_stats(ratings_df, movies_df)

        for idx, (_, row) in enumerate(ratings_df.iterrows()):
            uid = row["userId"]
            mid = row["movieId"]

            # Sparse
            sparse_indices[idx, 0] = self.sparse_maps["user_id"].get(uid, 0)
            sparse_indices[idx, 1] = self.sparse_maps["movie_id"].get(mid, 0)

            # Multi-Hot (genres)
            genres = movie_genre_idx.get(mid, [0])
            multi_hot_indices_list.append(genres)

            # Continuous
            movie_info = movies_df.loc[mid] if mid in movies_df.index else None
            ustats = user_stats.get(uid, {})

            douban_score = float(movie_info.get("DOUBAN_SCORE", 0) or 0) if movie_info is not None else 0.0
            if not (douban_score > 0):  # handles 0, NaN, None
                douban_score = 5.0
            douban_votes = float(movie_info.get("DOUBAN_VOTES", 0) or 0) if movie_info is not None else 0.0
            year = float(movie_info.get("YEAR", 0) or 0) if movie_info is not None else 0.0
            if not (year > 1900):
                year = 2010.0

            continuous[idx, 0] = douban_score / 10.0  # 归一化
            continuous[idx, 1] = np.log1p(douban_votes) / 15.0
            continuous[idx, 2] = min(year / 2025.0, 1.5)
            continuous[idx, 3] = ustats.get("mean", 3.0) / 5.0
            continuous[idx, 4] = min(ustats.get("std", 1.0) / 3.0, 1.0)
            continuous[idx, 5] = ustats.get("high_ratio", 0.3)
            continuous[idx, 6] = ustats.get("low_ratio", 0.2)

        # Pad multi-hot 到相同长度
        max_len = max(len(x) for x in multi_hot_indices_list)
        mh_padded = np.zeros((n, max_len), dtype=np.int64)
        mh_mask = np.zeros((n, max_len), dtype=np.float32)
        for i, x in enumerate(multi_hot_indices_list):
            mh_padded[i, :len(x)] = x
            mh_mask[i, :len(x)] = 1.0

        return {
            "sparse_indices": torch.tensor(sparse_indices, dtype=torch.long),
            "multi_hot_indices": torch.tensor(mh_padded, dtype=torch.long),
            "multi_hot_mask": torch.tensor(mh_mask, dtype=torch.float32),
            "continuous": torch.tensor(continuous, dtype=torch.float32),
        }

    def _compute_user_stats(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
    ) -> Dict[str, dict]:
        """计算每个用户的评分统计"""
        stats = {}
        for uid, grp in ratings_df.groupby("userId"):
            ratings = grp["rating"].values
            stats[uid] = {
                "mean": float(np.mean(ratings)),
                "std": float(np.std(ratings)) if len(ratings) > 1 else 0.0,
                "high_ratio": float((ratings >= 4.0).mean()),
                "low_ratio": float((ratings <= 2.0).mean()),
            }
        return stats


# ─────────────────────────────────────────────────────────
# 模型组件
# ─────────────────────────────────────────────────────────

class DeepFMModel(nn.Module):
    """
    DeepFM 模型。

    结构:
        Sparse Embedding (每个 sparse feat 一个 Embedding 表)
        Multi-Hot Embedding (每个 multi-hot feat 一个 Embedding 表，sum pool)
        Continuous → Dense 映射 (1维 Embedding 的等价实现)

        FM 分支（一阶 + 二阶交互）
        DNN 分支 (MLP)
        最终预测 = sigmoid(FM_out + DNN_out)
    """

    def __init__(
        self,
        sparse_vocab_sizes: Dict[str, int],
        multi_hot_vocab_sizes: Dict[str, int],
        n_cont_feats: int,
        sparse_embed_dim: int = 8,
        deep_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2,
    ):
        super().__init__()

        self.sparse_embed_dim = sparse_embed_dim
        self.n_sparse = len(sparse_vocab_sizes)
        self.n_multi_hot = len(multi_hot_vocab_sizes)
        self.n_cont = n_cont_feats

        # Sparse Embeddings
        self.sparse_embs = nn.ModuleDict({
            name: nn.Embedding(vocab_size, sparse_embed_dim)
            for name, vocab_size in sparse_vocab_sizes.items()
        })

        # Multi-Hot Embeddings（每个类型一个 Embedding，Pool 时用 mask）
        self.mh_embs = nn.ModuleDict({
            name: nn.Embedding(vocab_size, sparse_embed_dim)
            for name, vocab_size in multi_hot_vocab_sizes.items()
        })

        # Continuous → 等维 Embedding（用 Linear 代替 1维 Embedding）
        self.cont_fc = nn.Linear(n_cont_feats, n_cont_feats * sparse_embed_dim)

        # 总 Embedding 数（FM 所需的 field 数）
        self.n_fields = self.n_sparse + self.n_multi_hot + n_cont_feats
        self.total_embed_dim = self.n_fields * sparse_embed_dim

        # ── FM 一阶 ──
        self.fm_linear_sparse = nn.ModuleDict({
            name: nn.Embedding(vocab_size, 1)
            for name, vocab_size in sparse_vocab_sizes.items()
        })
        self.fm_linear_mh = nn.ModuleDict({
            name: nn.Embedding(vocab_size, 1)
            for name, vocab_size in multi_hot_vocab_sizes.items()
        })
        self.fm_linear_cont = nn.Linear(n_cont_feats, 1)

        self.fm_global_bias = nn.Parameter(torch.zeros(1))

        # ── DNN 分支 ──
        dnn_input_dim = self.total_embed_dim
        layers = []
        for hidden_dim in deep_layers:
            layers.append(nn.Linear(dnn_input_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            dnn_input_dim = hidden_dim
        self.dnn = nn.Sequential(*layers)
        self.dnn_out = nn.Linear(deep_layers[-1], 1)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for emb in self.sparse_embs.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.mh_embs.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.fm_linear_sparse.values():
            nn.init.xavier_uniform_(emb.weight)
        for emb in self.fm_linear_mh.values():
            nn.init.xavier_uniform_(emb.weight)

    def forward(self, features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: {
                "sparse_indices": (B, n_sparse)
                "multi_hot_indices": (B, multi_hot_max_len)
                "multi_hot_mask": (B, multi_hot_max_len)
                "continuous": (B, n_cont)
            }

        Returns:
            pred: (B,) logits
        """
        sparse_idx = features["sparse_indices"]  # (B, 2)
        mh_idx = features["multi_hot_indices"]   # (B, max_len)
        mh_mask = features["multi_hot_mask"]      # (B, max_len)
        cont = features["continuous"]             # (B, n_cont)

        B = sparse_idx.shape[0]
        all_embs = []  # 收集所有 Embedding 用于 FM 二阶

        # ── Sparse Embedding ──
        for i, (name, emb) in enumerate(self.sparse_embs.items()):
            idx = sparse_idx[:, i]  # (B,)
            e = emb(idx)  # (B, embed_dim)
            all_embs.append(e)

        # ── Multi-Hot Embedding (sum pooling with mask) ──
        for name, emb in self.mh_embs.items():
            e = emb(mh_idx)  # (B, max_len, embed_dim)
            e = e * mh_mask.unsqueeze(-1)  # mask
            e = e.sum(dim=1)  # (B, embed_dim) - sum pooling
            # 归一化（防止激活值过大）
            count = mh_mask.sum(dim=1, keepdim=True).clamp(min=1)
            e = e / count
            all_embs.append(e)

        # ── Continuous → Dense Embedding ──
        cont_emb = self.cont_fc(cont)  # (B, n_cont * embed_dim)
        cont_emb = cont_emb.view(B, self.n_cont, self.sparse_embed_dim)  # (B, n_cont, embed_dim)
        # 把每个连续特征看作一个 Embedding 加入
        for j in range(self.n_cont):
            all_embs.append(cont_emb[:, j, :])

        # Stack: (B, n_fields, embed_dim)
        emb_stack = torch.stack(all_embs, dim=1)

        # ── FM 一阶 ──
        fm_first_order = self.fm_global_bias
        for i, (name, linear_emb) in enumerate(self.fm_linear_sparse.items()):
            fm_first_order = fm_first_order + linear_emb(sparse_idx[:, i]).squeeze(-1)
        # Multi-Hot 一阶
        for name, linear_emb in self.fm_linear_mh.items():
            e = linear_emb(mh_idx).squeeze(-1)  # (B, max_len)
            e = (e * mh_mask).sum(dim=1)
            fm_first_order = fm_first_order + e
        fm_first_order = fm_first_order + self.fm_linear_cont(cont).squeeze(-1)

        # ── FM 二阶交互 ──
        # 公式: Σ(Σv_i)^2 - Σ(v_i)^2
        sum_square = emb_stack.sum(dim=1).pow(2)  # (B, embed_dim)
        square_sum = emb_stack.pow(2).sum(dim=1)  # (B, embed_dim)
        fm_second_order = 0.5 * (sum_square - square_sum).sum(dim=1)  # (B,)

        fm_out = fm_first_order + fm_second_order

        # ── DNN ──
        dnn_input = emb_stack.reshape(B, -1)  # (B, n_fields * embed_dim)
        dnn_hidden = self.dnn(dnn_input)
        dnn_out = self.dnn_out(dnn_hidden).squeeze(-1)  # (B,)

        # ── 最终输出 ──
        return fm_out + dnn_out


# ─────────────────────────────────────────────────────────
# DeepFM 推荐器（继承 BaseRecommender）
# ─────────────────────────────────────────────────────────

class DeepFMRecommender(BaseRecommender):
    """
    DeepFM 精排推荐器。

    在候选集上打分排序，通常接收 LightGCN 或 Popularity 的召回结果作为输入。
    """

    def __init__(
        self,
        name: str = "deepfm",
        sparse_embed_dim: int = 8,
        deep_layers: List[int] = [128, 64, 32],
        dropout: float = 0.2,
        lr: float = 1e-3,
        n_epochs: int = 50,
        batch_size: int = 256,
        rating_pos_threshold: float = 4.0,
        rating_neg_threshold: float = 2.0,
        early_stop_patience: int = 5,
        device: str = "cuda",
    ):
        super().__init__(name=name)
        self.sparse_embed_dim = sparse_embed_dim
        self.deep_layers = deep_layers
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.rating_pos_threshold = rating_pos_threshold
        self.rating_neg_threshold = rating_neg_threshold
        self.early_stop_patience = early_stop_patience
        self.dev = torch.device(device if torch.cuda.is_available() else "cpu")

        # 模型和特征处理器
        self.model: DeepFMModel | None = None
        self.feature_processor: FeatureProcessor | None = None

        # 缓存特征
        self._user_seen: dict[str, set[str]] = {}
        self._movies_df: pd.DataFrame | None = None
        self._ratings_df: pd.DataFrame | None = None
        self._features_cache: Dict[str, torch.Tensor] | None = None

    def _build_dataset(self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> "CTRDataset":
        """构建 CTR 训练数据集"""
        return CTRDataset(
            ratings_df,
            movies_df,
            self.feature_processor,
            self.rating_pos_threshold,
            self.rating_neg_threshold,
        )

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame | None = None,
        val_ratings_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """
        训练 DeepFM 模型。

        Args:
            ratings_df: 训练集评分
            movies_df:  电影元数据
            val_ratings_df: 验证集评分
        """
        if movies_df is None:
            raise ValueError("DeepFM requires movies_df")

        self._movies_df = movies_df
        self._ratings_df = ratings_df

        self._build_index_maps(ratings_df)

        # 记录用户已看
        self._user_seen = {}
        for _, row in ratings_df.iterrows():
            uid, mid = row["userId"], row["movieId"]
            self._user_seen.setdefault(uid, set()).add(mid)

        # 特征工程
        print(f"[DeepFM] Building feature processor...")
        self.feature_processor = FeatureProcessor()
        self.feature_processor.fit(ratings_df, movies_df)
        print(f"[DeepFM] Features: {len(self.feature_processor.sparse_feats)} sparse, "
              f"{len(self.feature_processor.multi_hot_feats)} multi-hot, "
              f"{len(self.feature_processor.cont_feats)} continuous")
        print(f"[DeepFM] Total fields: {self.feature_processor.n_fields}")

        # 缓存全量特征
        print(f"[DeepFM] Transforming features...")
        self._features_cache = self.feature_processor.transform(ratings_df, movies_df)

        # 构建模型
        self.model = DeepFMModel(
            sparse_vocab_sizes=self.feature_processor.sparse_vocab_sizes,
            multi_hot_vocab_sizes=self.feature_processor.multi_hot_vocab_sizes,
            n_cont_feats=len(self.feature_processor.cont_feats),
            sparse_embed_dim=self.sparse_embed_dim,
            deep_layers=self.deep_layers,
            dropout=self.dropout,
        ).to(self.dev)

        print(f"[DeepFM] Model params: {sum(p.numel() for p in self.model.parameters()):,}")

        # 数据集
        dataset = self._build_dataset(ratings_df, movies_df)
        if val_ratings_df is not None:
            val_dataset = self._build_dataset(val_ratings_df, movies_df)

        train_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        if val_ratings_df is not None:
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size * 2, shuffle=False)
        else:
            val_loader = None

        # 训练
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        print(f"[DeepFM] Training {self.n_epochs} epochs on {self.dev}...")

        for epoch in range(1, self.n_epochs + 1):
            self.model.train()
            total_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                batch = {k: v.to(self.dev) for k, v in batch.items()}
                labels = batch.pop("label")

                optimizer.zero_grad()
                logits = self.model(batch)
                loss = criterion(logits, labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)

            # 验证
            val_loss_str = ""
            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                val_n = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = {k: v.to(self.dev) for k, v in batch.items()}
                        labels = batch.pop("label")
                        logits = self.model(batch)
                        val_loss += criterion(logits, labels).item() * len(labels)
                        val_n += len(labels)
                val_loss = val_loss / max(val_n, 1)
                val_loss_str = f" | Val Loss: {val_loss:.4f}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

            if epoch == 1 or epoch % 5 == 0:
                print(f"[DeepFM] Epoch {epoch:3d}/{self.n_epochs} | Train Loss: {avg_loss:.4f}{val_loss_str}")

            if patience_counter >= self.early_stop_patience:
                print(f"[DeepFM] Early stopping at epoch {epoch}")
                break

        # 恢复最佳模型
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(f"[DeepFM] Loaded best model (val_loss={best_val_loss:.4f})")

        self.is_fitted = True

    def predict(self, user_id: str, movie_id: str) -> float:
        """预测用户对电影的偏好分数（0~1，接近 1 表示喜欢）"""
        self._check_fitted()
        if self._movies_df is None or self._ratings_df is None:
            return 0.5

        # 构造一条样本
        row = pd.DataFrame([{
            "userId": user_id,
            "movieId": movie_id,
            "rating": 3.0,  # dummy
        }])
        features = self.feature_processor.transform(row, self._movies_df)
        features = {k: v.to(self.dev) for k, v in features.items()}

        self.model.eval()
        with torch.no_grad():
            logit = self.model(features)
            prob = torch.sigmoid(logit).item()

        return prob

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        对候选电影集打分排序。

        如果未提供 candidate_movies，则在全量电影上打分（开销大，不推荐）。
        """
        self._check_fitted()

        if candidate_movies is not None:
            candidates = candidate_movies
        else:
            candidates = self.all_movie_ids

        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        candidates = [m for m in candidates if m not in seen]

        if not candidates:
            return []

        # 批量打分
        rows = pd.DataFrame([
            {"userId": user_id, "movieId": mid, "rating": 3.0}
            for mid in candidates
        ])
        features = self.feature_processor.transform(rows, self._movies_df)
        features = {k: v.to(self.dev) for k, v in features.items()}

        self.model.eval()
        with torch.no_grad():
            logits = self.model(features)
            probs = torch.sigmoid(logits).cpu().numpy()

        scored = list(zip(candidates, probs))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:n]

    def save(self, path: str) -> None:
        """保存模型"""
        self._check_fitted()
        state = {
            "model_state": self.model.state_dict(),
            "feature_processor": self.feature_processor,
            "sparse_embed_dim": self.sparse_embed_dim,
            "deep_layers": self.deep_layers,
            "dropout": self.dropout,
            "user_id_to_idx": self.user_id_to_idx,
            "movie_id_to_idx": self.movie_id_to_idx,
            "idx_to_user_id": self.idx_to_user_id,
            "idx_to_movie_id": self.idx_to_movie_id,
            "all_movie_ids": self.all_movie_ids,
            "_user_seen": self._user_seen,
        }
        torch.save(state, path)
        print(f"[DeepFM] Saved to {path}")

    def load(self, path: str, movies_df: pd.DataFrame) -> None:
        """加载模型"""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.feature_processor = state["feature_processor"]
        self.sparse_embed_dim = state["sparse_embed_dim"]
        self.deep_layers = state["deep_layers"]
        self.dropout = state["dropout"]

        self.model = DeepFMModel(
            sparse_vocab_sizes=self.feature_processor.sparse_vocab_sizes,
            multi_hot_vocab_sizes=self.feature_processor.multi_hot_vocab_sizes,
            n_cont_feats=len(self.feature_processor.cont_feats),
            sparse_embed_dim=self.sparse_embed_dim,
            deep_layers=self.deep_layers,
            dropout=self.dropout,
        )
        self.model.load_state_dict(state["model_state"])
        self.model.to(self.dev)

        self.user_id_to_idx = state["user_id_to_idx"]
        self.movie_id_to_idx = state["movie_id_to_idx"]
        self.idx_to_user_id = state["idx_to_user_id"]
        self.idx_to_movie_id = state["idx_to_movie_id"]
        self.all_movie_ids = state["all_movie_ids"]
        self._user_seen = state.get("_user_seen", {})
        self._movies_df = movies_df
        self.is_fitted = True
        print(f"[DeepFM] Loaded from {path}")


# ─────────────────────────────────────────────────────────
# CTR Dataset
# ─────────────────────────────────────────────────────────

class CTRDataset(Dataset):
    """
    CTR 二分类训练数据集。

    正样本: rating >= pos_threshold
    负样本: rating <= neg_threshold
    中间评分不参与训练（模糊区域）
    """

    def __init__(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame,
        feature_processor: FeatureProcessor,
        pos_threshold: float = 4.0,
        neg_threshold: float = 2.0,
    ):
        # 过滤出正负样本
        mask = (
            (ratings_df["rating"] >= pos_threshold) |
            (ratings_df["rating"] <= neg_threshold)
        )
        self.df = ratings_df[mask].reset_index(drop=True)
        self.labels = (self.df["rating"] >= pos_threshold).astype(np.float32)

        # 特征变换
        self.features = feature_processor.transform(self.df, movies_df)

        print(f"[CTRDataset] Samples: {len(self.df)} "
              f"(pos={self.labels.sum():.0f}, neg={len(self.df) - self.labels.sum():.0f})")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "sparse_indices": self.features["sparse_indices"][idx],
            "multi_hot_indices": self.features["multi_hot_indices"][idx],
            "multi_hot_mask": self.features["multi_hot_mask"][idx],
            "continuous": self.features["continuous"][idx],
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        return item
