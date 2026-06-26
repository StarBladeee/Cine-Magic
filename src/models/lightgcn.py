"""
LightGCN 图协同过滤召回模型

基于 He et al. (SIGIR 2020) "LightGCN: Simplifying and Powering Graph Convolution
Network for Recommendation" 的 PyTorch + PyG 实现。

核心思想:
- 去掉 GCN 中的特征变换矩阵和非线性激活函数
- 只保留邻居聚合的归一化求和（对称归一化拉普拉斯）
- K 层聚合后取各层 Embedding 的加权平均作为最终表示
- BPR Loss（Bayesian Personalized Ranking）训练

图结构:
- User-Item 二部图（无向边）
- 节点数 = n_users + n_items
- 边 = 每条评分记录是一条无向边

References:
    https://arxiv.org/abs/2002.02126
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import LGConv

from typing import List, Tuple
import pandas as pd
from scipy.sparse import csr_matrix

from src.models.base import BaseRecommender


class LightGCN(BaseRecommender, nn.Module):
    """
    LightGCN 召回模型。

    使用 PyG 的 LGConv 模块逐层传播，BPR Loss 训练，输出用户/物品 Embedding。
    """

    def __init__(
        self,
        name: str = "lightgcn",
        embed_dim: int = 64,
        n_layers: int = 3,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 2048,
        reg_weight: float = 1e-4,
        top_k: int = 50,
        early_stop_patience: int = 10,
        num_neg_samples: int = 1,
        device: str = "cuda",
    ):
        BaseRecommender.__init__(self, name=name)
        nn.Module.__init__(self)
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.top_k = top_k
        self.early_stop_patience = early_stop_patience
        self.num_neg_samples = num_neg_samples
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 训练后填充
        self.user_emb: torch.Tensor | None = None
        self.item_emb: torch.Tensor | None = None

        # 图数据
        self.n_users = 0
        self.n_items = 0
        self.edge_index: torch.Tensor | None = None
        self.edge_weight: torch.Tensor | None = None

        # 训练历史
        self.train_losses: list[float] = []
        self.val_metrics: list[float] = []

        # 用户已看集合（推荐时排除）
        self._user_seen: dict[str, set[str]] = {}

    # ── 模型定义 ──

    def _init_model(self) -> None:
        """初始化 Embedding 表和 LGConv 层"""
        self.user_embedding = nn.Embedding(self.n_users, self.embed_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embed_dim)

        # 用 Xavier 初始化
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)

        # LGConv 轻量卷积层（无特征变换、无激活）
        self.convs = nn.ModuleList([
            LGConv() for _ in range(self.n_layers)
        ])

    def get_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行 K 层图卷积，返回最终的 User/Item Embedding。
        """
        # 确保 edge_index 在正确设备上
        if self.edge_index.device != self.user_embedding.weight.device:
            self.edge_index = self.edge_index.to(self.user_embedding.weight.device)
        if self.edge_weight is not None and self.edge_weight.device != self.user_embedding.weight.device:
            self.edge_weight = self.edge_weight.to(self.user_embedding.weight.device)

        ego_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embs = [ego_emb]

        for conv in self.convs:
            ego_emb = conv(ego_emb, self.edge_index, edge_weight=self.edge_weight)
            all_embs.append(ego_emb)

        # 各层取平均
        final_emb = torch.stack(all_embs, dim=0).mean(dim=0)

        user_emb, item_emb = torch.split(final_emb, [self.n_users, self.n_items])
        return user_emb, item_emb

    # ── 图构建 ──

    def _build_bipartite_graph(self, ratings_df: pd.DataFrame) -> None:
        """
        从评分数据构建 User-Item 二部图。

        节点编号: 0..n_users-1 是用户, n_users..n_users+n_items-1 是物品
        边: (user_idx, item_idx + n_users)，包含每条评分记录
        """
        # 映射 ID → 内部索引
        users = ratings_df["userId"].unique()
        items = ratings_df["movieId"].unique()
        self.n_users = len(users)
        self.n_items = len(items)

        self.user_id_to_idx = {uid: i for i, uid in enumerate(users)}
        self.idx_to_user_id = {i: uid for uid, i in self.user_id_to_idx.items()}
        self.movie_id_to_idx = {mid: i for i, mid in enumerate(items)}
        self.idx_to_movie_id = {i: mid for mid, i in self.movie_id_to_idx.items()}
        self.all_movie_ids = list(items)

        # 记录用户已看
        self._user_seen = {}
        for _, row in ratings_df.iterrows():
            uid, mid = row["userId"], row["movieId"]
            self._user_seen.setdefault(uid, set()).add(mid)

        # 构建边
        src = []
        dst = []
        for _, row in ratings_df.iterrows():
            u = self.user_id_to_idx[row["userId"]]
            i = self.movie_id_to_idx[row["movieId"]]
            src.append(u)
            dst.append(self.n_users + i)

        # 无向图：加反向边
        src_all = torch.tensor(src + [d for d in dst], dtype=torch.long)
        dst_all = torch.tensor(dst + [s for s in src], dtype=torch.long)

        self.edge_index = torch.stack([src_all, dst_all], dim=0).to(self.device)

        # GCN 归一化边权重
        self.edge_index, self.edge_weight = gcn_norm(self.edge_index, num_nodes=self.n_users + self.n_items)

    def _build_adj_matrix(self, ratings_df: pd.DataFrame) -> csr_matrix:
        """构建稀疏邻接矩阵（备用，用于可能的稀疏操作）"""
        rows = [self.user_id_to_idx[uid] for uid in ratings_df["userId"]]
        cols = [self.movie_id_to_idx[mid] for mid in ratings_df["movieId"]]
        data = np.ones(len(ratings_df))

        adj = csr_matrix(
            (data, (rows, cols)),
            shape=(self.n_users, self.n_items),
            dtype=np.float32,
        )
        return adj

    # ── BPR 数据加载 ──

    def _sample_bpr_pairs(self, ratings_df: pd.DataFrame) -> List[Tuple[int, int, int]]:
        """
        为 BPR Loss 采样三元组 (user, pos_item, neg_item)。

        对每条评分 (user, pos_item, rating)，从 user 未评分的物品中随机采样 1 个作为负样本。
        """
        # 构建每个 user 的已评分集合
        user_pos_items: dict[int, set[int]] = {}
        for _, row in ratings_df.iterrows():
            u = self.user_id_to_idx[row["userId"]]
            i = self.movie_id_to_idx[row["movieId"]]
            user_pos_items.setdefault(u, set()).add(i)

        # 所有物品的快速负采样
        all_items_set = set(range(self.n_items))

        pairs = []
        for u, pos_set in user_pos_items.items():
            neg_pool = list(all_items_set - pos_set)
            if not neg_pool:
                continue
            for pos_i in pos_set:
                for _ in range(self.num_neg_samples):
                    neg_i = int(np.random.choice(neg_pool))
                    pairs.append((u, pos_i, neg_i))

        return pairs

    def _sample_bpr_batch(self, pairs: List[Tuple[int, int, int]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """从三元组列表中随机采样一个 batch"""
        batch_pairs = np.random.choice(len(pairs), size=min(self.batch_size, len(pairs)), replace=False)
        users = torch.tensor([pairs[i][0] for i in batch_pairs], dtype=torch.long)
        pos_items = torch.tensor([pairs[i][1] for i in batch_pairs], dtype=torch.long)
        neg_items = torch.tensor([pairs[i][2] for i in batch_pairs], dtype=torch.long)
        return users.to(self.device), pos_items.to(self.device), neg_items.to(self.device)

    # ── 训练 ──

    def fit(
        self,
        ratings_df: pd.DataFrame,
        movies_df: pd.DataFrame | None = None,
        val_ratings_df: pd.DataFrame | None = None,
        **kwargs,
    ) -> None:
        """
        训练 LightGCN 模型。

        Args:
            ratings_df:     训练集评分
            val_ratings_df: 验证集评分（用于早停），如果为 None 则从训练集中留出 5%
        """
        # 构建图
        print(f"[LightGCN] Building bipartite graph...")
        self._build_bipartite_graph(ratings_df)
        print(f"[LightGCN] Graph: {self.n_users} users, {self.n_items} items, "
              f"{len(self.edge_index[0])//2} edges")

        # 如果不提供验证集，从训练集中采样一部分作为验证
        if val_ratings_df is None:
            n_val = int(len(ratings_df) * 0.05)
            val_df = ratings_df.sample(n=n_val, random_state=42)
            train_df = ratings_df.drop(val_df.index)
        else:
            train_df = ratings_df
            val_df = val_ratings_df

        # BPR 三元组
        print(f"[LightGCN] Sampling BPR triplets...")
        train_pairs = self._sample_bpr_pairs(train_df)
        print(f"[LightGCN] Training triplets: {len(train_pairs)}")

        # 验证 ground truth（每个用户的测试集正样本）
        val_user_items: dict[int, set[int]] = {}
        for _, row in val_df.iterrows():
            uid = row["userId"]
            mid = row["movieId"]
            if uid in self.user_id_to_idx and mid in self.movie_id_to_idx:
                u = self.user_id_to_idx[uid]
                i = self.movie_id_to_idx[mid]
                val_user_items.setdefault(u, set()).add(i)

        # 初始化模型
        self._init_model()
        self.to(self.device)

        # 确保边索引在正确设备上
        self.edge_index = self.edge_index.to(self.device)
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(self.device)

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        best_ndcg = 0.0
        best_epoch = 0
        patience_counter = 0

        print(f"[LightGCN] Training {self.n_epochs} epochs on {self.device}...")
        if self.device.type == "cuda":
            print(f"[LightGCN] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        for epoch in range(1, self.n_epochs + 1):
            self.train()

            # Shuffle 三元组
            np.random.shuffle(train_pairs)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(train_pairs), self.batch_size):
                end = min(start + self.batch_size, len(train_pairs))
                batch_pairs = train_pairs[start:end]

                users = torch.tensor([p[0] for p in batch_pairs], dtype=torch.long, device=self.device)
                pos_items = torch.tensor([p[1] for p in batch_pairs], dtype=torch.long, device=self.device)
                neg_items = torch.tensor([p[2] for p in batch_pairs], dtype=torch.long, device=self.device)

                optimizer.zero_grad()

                # 获取 Embedding（确保在正确的设备上）
                user_emb, item_emb = self.get_embeddings()
                user_emb = user_emb.to(self.device)
                item_emb = item_emb.to(self.device)

                # BPR Loss
                u_emb = user_emb[users]
                pos_emb = item_emb[pos_items]
                neg_emb = item_emb[neg_items]

                pos_scores = (u_emb * pos_emb).sum(dim=1)
                neg_scores = (u_emb * neg_emb).sum(dim=1)

                bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()

                # L2 正则化
                reg_loss = 0.5 * self.reg_weight * (
                    u_emb.norm(2).pow(2) +
                    pos_emb.norm(2).pow(2) +
                    neg_emb.norm(2).pow(2)
                ) / len(users)

                loss = bpr_loss + reg_loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            self.train_losses.append(avg_loss)

            # 每 5 个 epoch 评估一次
            if epoch % 5 == 0 or epoch == 1:
                self.eval()
                with torch.no_grad():
                    user_emb, item_emb = self.get_embeddings()
                    ndcg = self._compute_recall_ndcg(user_emb, item_emb, val_user_items, k=20)
                    self.val_metrics.append(ndcg)

                status = f"[LightGCN] Epoch {epoch:3d}/{self.n_epochs} | Loss: {avg_loss:.4f} | Val NDCG@20: {ndcg:.4f}"

                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    best_epoch = epoch
                    patience_counter = 0
                    status += " *"
                    # 保存最佳 Embedding
                    self.user_emb = user_emb.clone().detach().cpu()
                    self.item_emb = item_emb.clone().detach().cpu()
                else:
                    patience_counter += 1

                print(status)

                if patience_counter >= self.early_stop_patience:
                    print(f"[LightGCN] Early stopping at epoch {epoch}")
                    break

        print(f"[LightGCN] Best val NDCG@20: {best_ndcg:.4f} (epoch {best_epoch})")

        # 如果因为早停等原因没有保存最终 Embedding，保存当前的
        if self.user_emb is None:
            self.eval()
            with torch.no_grad():
                user_emb, item_emb = self.get_embeddings()
                self.user_emb = user_emb.clone().detach().cpu()
                self.item_emb = item_emb.clone().detach().cpu()

        self.is_fitted = True

    def _compute_recall_ndcg(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        val_user_items: dict[int, set[int]],
        k: int = 20,
    ) -> float:
        """
        用验证集计算 Recall@K 和 NDCG@K 的均值。
        简化实现：用内积对全部物品排序，计算命中率。
        """
        ndcg_sum = 0.0
        n_users = 0

        for u, true_items in val_user_items.items():
            if not true_items:
                continue

            u_vec = user_emb[u]  # (embed_dim,)
            scores = torch.matmul(item_emb, u_vec)  # (n_items,)

            # 排除训练集中已评分的物品？
            # 这里不过滤，因为验证就是验证是否能找回测试集的物品
            _, top_indices = torch.topk(scores, k)

            hits = [1 if idx.item() in true_items else 0 for idx in top_indices]

            # DCG
            dcg = sum(
                hit / np.log2(i + 2)
                for i, hit in enumerate(hits)
            )
            # IDCG（理想情况下所有 true_items 排在前面）
            ideal_hits = min(len(true_items), k)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))

            ndcg_sum += dcg / idcg if idcg > 0 else 0.0
            n_users += 1

        return ndcg_sum / max(n_users, 1)

    def to(self, device: torch.device):
        """将模型参数移到指定设备"""
        self.device = device
        if self.edge_index is not None:
            self.edge_index = self.edge_index.to(device)
        if self.edge_weight is not None:
            self.edge_weight = self.edge_weight.to(device)
        return self

    # ── 推荐接口 ──

    def predict(self, user_id: str, movie_id: str) -> float:
        """预测评分（用 Embedding 内积），映射到 1-5 区间"""
        self._check_fitted()
        if user_id not in self.user_id_to_idx or movie_id not in self.movie_id_to_idx:
            return 2.5
        u = self.user_id_to_idx[user_id]
        i = self.movie_id_to_idx[movie_id]
        score = float(self.user_emb[u] @ self.item_emb[i])
        # 将内积映射到 1-5
        return 1.0 + 4.0 * torch.sigmoid(torch.tensor(score)).item()

    def recommend(
        self,
        user_id: str,
        n: int = 10,
        exclude_seen: bool = True,
        candidate_movies: List[str] | None = None,
    ) -> list[tuple[str, float]]:
        """
        为指定用户召回 Top-K 电影。

        使用 Embedding 内积在全量物品上计算得分，取 Top-K。
        """
        self._check_fitted()

        if user_id not in self.user_id_to_idx:
            return self._cold_start_recommend(n)

        u = self.user_id_to_idx[user_id]
        u_vec = self.user_emb[u]

        if candidate_movies is not None:
            valid = [mid for mid in candidate_movies if mid in self.movie_id_to_idx]
            if not valid:
                return []
            indices = torch.tensor([self.movie_id_to_idx[mid] for mid in valid])
            item_vecs = self.item_emb[indices]
            scores = torch.matmul(item_vecs, u_vec).cpu()
            top_local_idx = torch.topk(scores, min(n, len(valid))).indices
            return [(valid[i], float(scores[j])) for j, i in enumerate(top_local_idx)]

        seen = self._user_seen.get(user_id, set()) if exclude_seen else set()
        seen_indices = {self.movie_id_to_idx[mid] for mid in seen if mid in self.movie_id_to_idx}

        # 全量内积计算
        scores = torch.matmul(self.item_emb, u_vec)  # (n_items,)

        # 获取 Top-(n + len(seen)) 再过滤
        top_k = min(n + len(seen_indices) + 10, self.n_items)
        _, top_indices = torch.topk(scores, top_k)

        results = []
        for idx in top_indices.tolist():
            mid = self.idx_to_movie_id[idx]
            if exclude_seen and mid in seen:
                continue
            results.append((mid, float(scores[idx])))
            if len(results) >= n:
                break

        return results

    def _cold_start_recommend(self, n: int) -> list[tuple[str, float]]:
        """冷启动用户：返回平均 Embedding Top-K（没有用户信号的兜底）"""
        avg_user_vec = self.user_emb.mean(dim=0)
        scores = torch.matmul(self.item_emb, avg_user_vec)
        _, top_indices = torch.topk(scores, min(n, self.n_items))
        return [
            (self.idx_to_movie_id[i], float(scores[i]))
            for i in top_indices.tolist()
        ]

    def get_similar_movies(
        self,
        movie_id: str,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        """通过 Item Embedding 余弦相似度找相似电影"""
        self._check_fitted()
        if movie_id not in self.movie_id_to_idx:
            return []

        idx = self.movie_id_to_idx[movie_id]
        target_vec = self.item_emb[idx]

        # 余弦相似度
        sims = F.cosine_similarity(target_vec.unsqueeze(0), self.item_emb).cpu()

        # 排除自身
        sims[idx] = -1.0

        _, top_indices = torch.topk(sims, min(n, len(sims) - 1))
        return [
            (self.idx_to_movie_id[int(i)], float(sims[int(i)]))
            for i in top_indices
            if self.idx_to_movie_id[int(i)] != movie_id
        ][:n]

    def parameters(self):
        """返回所有可优化参数"""
        params = list(self.user_embedding.parameters()) + list(self.item_embedding.parameters())
        for conv in self.convs:
            params += list(conv.parameters())
        return params

    def save(self, path: str) -> None:
        """保存模型到文件"""
        self._check_fitted()
        state = {
            "user_emb": self.user_emb,
            "item_emb": self.item_emb,
            "user_id_to_idx": self.user_id_to_idx,
            "movie_id_to_idx": self.movie_id_to_idx,
            "idx_to_user_id": self.idx_to_user_id,
            "idx_to_movie_id": self.idx_to_movie_id,
            "all_movie_ids": self.all_movie_ids,
            "_user_seen": self._user_seen,
            "config": {
                "embed_dim": self.embed_dim,
                "n_layers": self.n_layers,
                "n_users": self.n_users,
                "n_items": self.n_items,
            },
        }
        torch.save(state, path)
        print(f"[LightGCN] Saved to {path}")

    def load(self, path: str) -> None:
        """从文件加载模型"""
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.user_emb = state["user_emb"]
        self.item_emb = state["item_emb"]
        self.user_id_to_idx = state["user_id_to_idx"]
        self.movie_id_to_idx = state["movie_id_to_idx"]
        self.idx_to_user_id = state["idx_to_user_id"]
        self.idx_to_movie_id = state["idx_to_movie_id"]
        self.all_movie_ids = state["all_movie_ids"]
        self._user_seen = state.get("_user_seen", {})
        self.n_users = state["config"]["n_users"]
        self.n_items = state["config"]["n_items"]
        self.is_fitted = True
        print(f"[LightGCN] Loaded from {path}")
