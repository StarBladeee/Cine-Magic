#!/usr/bin/env python3
"""
CineMagic 模型训练脚本

Usage:
    python scripts/train.py --model svd
    python scripts/train.py --model popularity
    python scripts/train.py --model user_knn
    python scripts/train.py --model item_knn
    python scripts/train.py --model all
    python scripts/train.py --model all --full          # 使用全量数据
"""

import argparse
import pickle
import sys
import os

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.data.loader import load_movies, load_ratings
from src.data.preprocess import filter_sparse, train_test_split
from src.models.base import BaseRecommender
from src.models.popularity import PopularityRecommender
from src.models.collaborative import SVDRecommender, KNNRecommender


def parse_args():
    parser = argparse.ArgumentParser(description="CineMagic Model Training")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["popularity", "svd", "user_knn", "item_knn", "all"],
        help="Model type to train",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full dataset instead of top200",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: config value)",
    )
    return parser.parse_args()


def load_data(cfg: dict, use_full: bool):
    """加载并预处理数据"""
    data_cfg = cfg["data"]
    if use_full:
        movies_path = os.path.join(data_cfg["douban_dir"], data_cfg["full_movies_file"])
        ratings_path = os.path.join(data_cfg["douban_dir"], data_cfg["full_ratings_file"])
    else:
        movies_path = os.path.join(data_cfg["douban_dir"], data_cfg["movies_file"])
        ratings_path = os.path.join(data_cfg["douban_dir"], data_cfg["ratings_file"])

    print(f"Loading movies from: {movies_path}")
    movies = load_movies(movies_path)
    print(f"  Loaded {len(movies)} movies")

    print(f"Loading ratings from: {ratings_path}")
    ratings = load_ratings(ratings_path)
    print(f"  Loaded {len(ratings)} ratings")

    # 过滤稀疏
    preproc = cfg.get("preprocessing", {})
    min_user = preproc.get("min_user_ratings", 5)
    min_item = preproc.get("min_item_ratings", 5)
    print(f"Filtering: min_user_ratings={min_user}, min_item_ratings={min_item}")
    ratings = filter_sparse(ratings, min_user_ratings=min_user, min_item_ratings=min_item)
    print(f"  After filter: {len(ratings)} ratings, "
          f"{ratings['userId'].nunique()} users, {ratings['movieId'].nunique()} movies")

    # 划分
    test_size = preproc.get("test_size", 0.2)
    split_method = preproc.get("test_split_method", "random")
    seed = preproc.get("random_seed", 42)
    print(f"Splitting: test_size={test_size}, method={split_method}")
    train, test = train_test_split(ratings, test_size=test_size, method=split_method, random_seed=seed)
    print(f"  Train: {len(train)} ratings, Test: {len(test)} ratings")

    return movies, ratings, train, test


def build_model(model_type: str, cfg: dict) -> BaseRecommender:
    """根据配置构建模型"""
    model_cfg = cfg.get("models", {})

    if model_type == "popularity":
        pop_cfg = model_cfg.get("popularity", {})
        return PopularityRecommender(**pop_cfg)

    elif model_type == "svd":
        svd_cfg = model_cfg.get("svd", {})
        return SVDRecommender(**svd_cfg)

    elif model_type == "user_knn":
        knn_cfg = model_cfg.get("user_knn", {}).copy()
        knn_cfg.setdefault("name", "user_knn")
        knn_cfg.setdefault("user_based", True)
        # 展平 sim_options
        sim = knn_cfg.pop("sim_options", {})
        knn_cfg.setdefault("sim_name", sim.get("name", "pearson_baseline"))
        knn_cfg.setdefault("shrinkage", sim.get("shrinkage", 100))
        return KNNRecommender(**knn_cfg)

    elif model_type == "item_knn":
        knn_cfg = model_cfg.get("item_knn", {}).copy()
        knn_cfg.setdefault("name", "item_knn")
        knn_cfg.setdefault("user_based", False)
        sim = knn_cfg.pop("sim_options", {})
        knn_cfg.setdefault("sim_name", sim.get("name", "pearson_baseline"))
        knn_cfg.setdefault("shrinkage", sim.get("shrinkage", 100))
        return KNNRecommender(**knn_cfg)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_one_model(
    model_type: str,
    cfg: dict,
    movies,
    train: "pd.DataFrame",
    output_dir: str,
) -> BaseRecommender:
    """训练单个模型并保存"""
    model = build_model(model_type, cfg)
    print(f"\n{'='*50}")
    print(f"Training: {model_type.upper()}")
    print(f"{'='*50}")

    if model_type == "popularity":
        model.fit(train, movies_df=movies)
    else:
        model.fit(train)

    # 保存
    model_path = os.path.join(output_dir, f"{model_type}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

    return model


def main():
    args = parse_args()
    cfg = get_config()

    output_dir = args.output or cfg.get("output", {}).get("model_dir", "models")
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    movies, ratings, train, test = load_data(cfg, args.full)

    model_types = (
        ["popularity", "svd", "user_knn", "item_knn"]
        if args.model == "all"
        else [args.model]
    )

    models = {}
    for mt in model_types:
        models[mt] = train_one_model(mt, cfg, movies, train, output_dir)

    print(f"\n{'='*50}")
    print(f"Training complete! {len(models)} model(s) trained.")
    print(f"Models saved to: {output_dir}/")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
