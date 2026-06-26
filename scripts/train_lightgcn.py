#!/usr/bin/env python3
"""
CineMagic LightGCN 训练脚本

Usage:
    python scripts/train_lightgcn.py
    python scripts/train_lightgcn.py --epochs 200 --embed-dim 128
"""

import argparse
import sys
import os
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import get_config
from src.data.loader import load_movies, load_ratings
from src.data.preprocess import filter_sparse, train_test_split
from src.models.lightgcn import LightGCN


def parse_args():
    parser = argparse.ArgumentParser(description="CineMagic — LightGCN Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--reg-weight", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    model_cfg = cfg.get("models", {}).get("lightgcn", {})
    preproc_cfg = cfg.get("preprocessing", {})
    data_cfg = cfg["data"]

    # 加载数据
    movies_path = os.path.join(data_cfg["douban_dir"], data_cfg["movies_file"])
    ratings_path = os.path.join(data_cfg["douban_dir"], data_cfg["ratings_file"])

    print(f"Loading: {movies_path}")
    movies = load_movies(movies_path)
    print(f"Loading: {ratings_path}")
    ratings = load_ratings(ratings_path)
    print(f"  Movies: {len(movies)}, Ratings: {len(ratings)}")

    # 预处理
    ratings = filter_sparse(
        ratings,
        min_user_ratings=preproc_cfg.get("min_user_ratings", 5),
        min_item_ratings=preproc_cfg.get("min_item_ratings", 5),
    )
    print(f"  After filter: {len(ratings)} ratings, "
          f"{ratings['userId'].nunique()} users, "
          f"{ratings['movieId'].nunique()} movies")

    train, test = train_test_split(
        ratings,
        test_size=preproc_cfg.get("test_size", 0.2),
        method=preproc_cfg.get("test_split_method", "random"),
        random_seed=preproc_cfg.get("random_seed", 42),
    )
    print(f"  Train: {len(train)}, Test: {len(test)}")

    # 构建模型（命令行参数覆盖配置文件）
    kwargs = {
        "embed_dim": int(args.embed_dim or model_cfg.get("embed_dim", 64)),
        "n_layers": int(args.n_layers or model_cfg.get("n_layers", 3)),
        "lr": float(args.lr or model_cfg.get("lr", 1e-3)),
        "n_epochs": int(args.epochs or model_cfg.get("n_epochs", 100)),
        "batch_size": int(args.batch_size or model_cfg.get("batch_size", 2048)),
        "reg_weight": float(args.reg_weight or model_cfg.get("reg_weight", 1e-4)),
        "top_k": int(model_cfg.get("top_k", 50)),
        "early_stop_patience": int(model_cfg.get("early_stop_patience", 10)),
        "num_neg_samples": int(model_cfg.get("num_neg_samples", 1)),
        "device": args.device or model_cfg.get("device", "cuda"),
    }

    print(f"\n[LightGCN] Config: {kwargs}")

    model = LightGCN(**kwargs)

    # 训练（用测试集作为验证集来早停）
    model.fit(train, val_ratings_df=test)

    # 保存
    output_dir = args.output or cfg.get("output", {}).get("model_dir", "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "lightgcn.pt")
    model.save(model_path)

    # 测试推荐
    test_user = test["userId"].iloc[0]
    recs = model.recommend(test_user, n=10)
    print(f"\n[LightGCN] Test recommendations for user {test_user[:12]}...:")
    for mid, score in recs[:5]:
        name = movies.at[mid, "NAME"] if mid in movies.index else "?"
        print(f"  {name[:40]:<40} score={score:.3f}")

    print(f"\n[LightGCN] Done. Model saved to {model_path}")


if __name__ == "__main__":
    main()
