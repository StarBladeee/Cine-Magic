#!/usr/bin/env python3
"""
CineMagic 模型评估脚本

Usage:
    python scripts/evaluate.py --model svd
    python scripts/evaluate.py --model all
    python scripts/evaluate.py --model all --topk 5,10,20
"""

import argparse
import pickle
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from src.config import get_config
from src.data.loader import load_movies, load_ratings
from src.data.preprocess import filter_sparse, train_test_split
from src.evaluation.metrics import evaluate_model, compare_models


def parse_args():
    parser = argparse.ArgumentParser(description="CineMagic Model Evaluation")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["popularity", "svd", "user_knn", "item_knn", "all"],
        help="Model type to evaluate",
    )
    parser.add_argument(
        "--topk", type=str, default="5,10,20",
        help="Comma-separated K values for Precision/Recall/NDCG (default: 5,10,20)",
    )
    parser.add_argument(
        "--model-dir", type=str, default=None,
        help="Directory containing saved model .pkl files",
    )
    parser.add_argument(
        "--rating-threshold", type=float, default=3.5,
        help="Rating threshold for 'liked' classification (default: 3.5)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = get_config()

    model_dir = args.model_dir or cfg.get("output", {}).get("model_dir", "models")
    k_values = [int(k.strip()) for k in args.topk.split(",")]

    # 加载数据
    data_cfg = cfg["data"]
    movies_path = os.path.join(data_cfg["douban_dir"], data_cfg["movies_file"])
    ratings_path = os.path.join(data_cfg["douban_dir"], data_cfg["ratings_file"])

    print(f"Loading data...")
    movies = load_movies(movies_path)
    ratings = load_ratings(ratings_path)

    # 预处理
    preproc = cfg.get("preprocessing", {})
    ratings = filter_sparse(
        ratings,
        min_user_ratings=preproc.get("min_user_ratings", 5),
        min_item_ratings=preproc.get("min_item_ratings", 5),
    )
    train, test = train_test_split(
        ratings,
        test_size=preproc.get("test_size", 0.2),
        method=preproc.get("test_split_method", "random"),
        random_seed=preproc.get("random_seed", 42),
    )
    print(f"Train: {len(train)} ratings, Test: {len(test)} ratings")

    # 确定要评估的模型
    model_types = (
        ["popularity", "svd", "user_knn", "item_knn"]
        if args.model == "all"
        else [args.model]
    )

    # 加载模型
    models_loaded = {}
    for mt in model_types:
        model_path = os.path.join(model_dir, f"{mt}.pkl")
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found: {model_path}, skipping.")
            continue
        with open(model_path, "rb") as f:
            models_loaded[mt] = pickle.load(f)
        print(f"Loaded model: {mt}")

    if not models_loaded:
        print("No models found. Please run train.py first.")
        sys.exit(1)

    # 评估
    if len(models_loaded) == 1:
        name, model = next(iter(models_loaded.items()))
        results = evaluate_model(
            model, train, test, movies,
            k_values=k_values,
            rating_threshold=args.rating_threshold,
        )
        # 保存结果
        output_dir = cfg.get("output", {}).get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        result_path = os.path.join(output_dir, f"eval_{name}.json")
        import json
        with open(result_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {result_path}")
    else:
        comp_df = compare_models(
            models_loaded, train, test, movies,
            k_values=k_values,
            rating_threshold=args.rating_threshold,
        )
        print("\n" + "=" * 70)
        print("Model Comparison")
        print("=" * 70)
        print(comp_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

        # 保存
        output_dir = cfg.get("output", {}).get("output_dir", "output")
        os.makedirs(output_dir, exist_ok=True)
        comp_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
        print(f"\nComparison saved to {output_dir}/model_comparison.csv")


if __name__ == "__main__":
    main()
