#!/usr/bin/env python3
"""
CineMagic DeepFM 训练脚本

Usage:
    python scripts/train_deepfm.py
    python scripts/train_deepfm.py --epochs 100 --deep-layers 256,128,64
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
from src.models.deepfm import DeepFMRecommender


def parse_args():
    parser = argparse.ArgumentParser(description="CineMagic — DeepFM Training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--deep-layers", type=str, default=None,
                        help="Comma-separated hidden dims, e.g. 256,128,64")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    model_cfg = cfg.get("models", {}).get("deepfm", {})
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

    # 构建模型
    deep_layers = model_cfg.get("deep_layers", [128, 64, 32])
    if args.deep_layers:
        deep_layers = [int(d) for d in args.deep_layers.split(",")]

    kwargs = {
        "sparse_embed_dim": int(args.embed_dim or model_cfg.get("sparse_embed_dim", 8)),
        "deep_layers": deep_layers,
        "dropout": float(args.dropout or model_cfg.get("dropout", 0.2)),
        "lr": float(args.lr or model_cfg.get("lr", 1e-3)),
        "n_epochs": int(args.epochs or model_cfg.get("n_epochs", 50)),
        "batch_size": int(args.batch_size or model_cfg.get("batch_size", 256)),
        "rating_pos_threshold": float(model_cfg.get("rating_pos_threshold", 4.0)),
        "rating_neg_threshold": float(model_cfg.get("rating_neg_threshold", 2.0)),
        "early_stop_patience": int(model_cfg.get("early_stop_patience", 5)),
        "device": args.device or model_cfg.get("device", "cuda"),
    }

    print(f"\n[DeepFM] Config: {kwargs}")

    model = DeepFMRecommender(**kwargs)

    # 训练
    model.fit(train, movies_df=movies, val_ratings_df=test)

    # 保存
    output_dir = args.output or cfg.get("output", {}).get("model_dir", "models")
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "deepfm.pt")
    model.save(model_path)

    # 测试推荐
    test_user = test["userId"].iloc[0]
    test_movie = test["movieId"].iloc[0]
    prob = model.predict(test_user, test_movie)
    actual = test[test["movieId"] == test_movie]["rating"].iloc[0]
    print(f"\n[DeepFM] Test: user={test_user[:12]}..., movie={test_movie}, "
          f"prob={prob:.3f}, actual_rating={actual}")

    recs = model.recommend(test_user, n=10)
    print(f"[DeepFM] Recommendations for user {test_user[:12]}...:")
    for mid, score in recs[:5]:
        name = movies.at[mid, "NAME"] if mid in movies.index else "?"
        print(f"  {name[:40]:<40} score={score:.3f}")

    print(f"\n[DeepFM] Done. Model saved to {model_path}")


if __name__ == "__main__":
    main()
