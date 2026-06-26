#!/usr/bin/env python3
"""
CineMagic 命令行推荐入口

核心交互方式：通过命令行获取电影推荐、查看用户历史、找相似电影。

Usage:
    # 为用户推荐电影
    python scripts/recommend.py --user <user_id> --model svd --topk 10

    # 使用热门推荐兜底
    python scripts/recommend.py --user <user_id> --model popularity --topk 20

    # 查找相似电影
    python scripts/recommend.py --similar-to <movie_id> --topk 10

    # 查看用户观影历史
    python scripts/recommend.py --user <user_id> --history

    # 查看电影详情
    python scripts/recommend.py --movie <movie_id>

    # 列出可用模型
    python scripts/recommend.py --list-models

    # 列出热门电影
    python scripts/recommend.py --hot --genre <类型> --topk 10
"""

import argparse
import pickle
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_config
from src.data.loader import (
    load_movies,
    load_ratings,
    get_movie_by_id,
    get_user_history,
    get_movies_by_ids,
)
from src.data.preprocess import filter_sparse
from src.models.base import BaseRecommender


def parse_args():
    parser = argparse.ArgumentParser(
        description="CineMagic — 电影推荐系统命令行接口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/recommend.py --user abc123 --model svd --topk 10
  python scripts/recommend.py --similar-to 1292052 --topk 10
  python scripts/recommend.py --user abc123 --history
  python scripts/recommend.py --movie 1292052
  python scripts/recommend.py --hot --genre 动画 --topk 10
  python scripts/recommend.py --list-models
        """,
    )
    parser.add_argument("--user", type=str, help="User ID for personalized recommendation")
    parser.add_argument("--model", type=str, default="svd",
                        help="Model type (svd, user_knn, item_knn, popularity, "
                             "lightgcn, lightgcn+deepfm)")
    parser.add_argument("--topk", type=int, default=10, help="Number of recommendations (default: 10)")
    parser.add_argument("--similar-to", type=str, help="Movie ID to find similar movies")
    parser.add_argument("--movie", type=str, help="Movie ID to view details")
    parser.add_argument("--history", action="store_true", help="Show user rating history")
    parser.add_argument("--hot", action="store_true", help="Show popular movies")
    parser.add_argument("--genre", type=str, help="Filter by genre (e.g., 动画, 剧情)")
    parser.add_argument("--list-models", action="store_true", help="List available trained models")
    parser.add_argument("--model-dir", type=str, default=None,
                        help="Directory of saved model .pkl files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output format: table (default), json, csv")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device for deep models (cuda/cpu, default: cuda)")
    return parser.parse_args()

# DeepFMRecommender 延迟导入（避免 torch import 拖慢非深度学习场景）


def load_model(model_type: str, model_dir: str) -> BaseRecommender | None:
    """加载已训练的模型"""
    model_path = os.path.join(model_dir, f"{model_type}.pkl")
    if not os.path.exists(model_path):
        print(f"❌ Model '{model_type}' not found at {model_path}")
        available = _list_models(model_dir)
        if available:
            print(f"   Available: {', '.join(available)}")
        else:
            print(f"   Run 'python scripts/train.py --model {model_type}' first.")
        return None
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def _list_models(model_dir: str) -> list[str]:
    """列出可用模型文件"""
    if not os.path.isdir(model_dir):
        return []
    return [
        f.replace(".pkl", "")
        for f in os.listdir(model_dir)
        if f.endswith(".pkl")
    ]


def _format_score(score: float) -> str:
    """将预测得分格式化为星级"""
    stars = int(round(score))
    return "★" * min(stars, 5) + "☆" * max(5 - stars, 0)


def print_recommendations(
    recs: list[tuple[str, float]],
    movies,
    title: str = "Recommendations",
) -> None:
    """格式化打印推荐列表"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'#':<4} {'Name':<40} {'Score':<8} {'Genres':<25} {'Douban':<8}")
    print(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*25} {'-'*8}")

    for i, (movie_id, score) in enumerate(recs, 1):
        info = get_movie_by_id(movies, movie_id)
        if info:
            name = (info.get("NAME") or "Unknown")[:38]
            genres = (info.get("GENRES") or "")[:23]
            douban = info.get("DOUBAN_SCORE", 0)
            d_str = f"{douban:.1f}" if douban and douban > 0 else "N/A"
        else:
            name = "Unknown"
            genres = ""
            d_str = "N/A"

        print(f"  {i:<4} {name:<40} {_format_score(score):<8} {genres:<25} {d_str:<8}")

    print(f"{'='*80}\n")


def print_movie_detail(movie: dict) -> None:
    """格式化打印单部电影详情"""
    print(f"\n{'='*80}")
    print(f"  🎬  {movie.get('NAME', 'Unknown')}")
    print(f"{'='*80}")
    print(f"  ID:           {movie.get('MOVIE_ID', movie.get('movie_id', ''))}")
    print(f"  Director:     {movie.get('DIRECTORS', 'N/A')}")
    print(f"  Actors:       {str(movie.get('ACTORS', ''))[:80]}")
    print(f"  Genres:       {movie.get('GENRES', 'N/A')}")
    print(f"  Year:         {movie.get('YEAR', 'N/A')}")
    print(f"  Region:       {movie.get('REGIONS', 'N/A')}")
    print(f"  Duration:     {movie.get('MINS', 'N/A')} min")
    print(f"  Language:     {movie.get('LANGUAGES', 'N/A')}")
    print(f"  Douban Score: {movie.get('DOUBAN_SCORE', 'N/A')} "
          f"({int(movie.get('DOUBAN_VOTES', 0))} votes)")
    print(f"  Alias:        {movie.get('ALIAS', 'N/A')}")
    storyline = movie.get('STORYLINE', '')
    if storyline and not pd.isna(storyline):
        print(f"  Storyline:    {str(storyline)[:200]}")
    print(f"{'='*80}\n")


def print_user_history(history, movies) -> None:
    """打印用户观影历史"""
    print(f"\n{'='*80}")
    print(f"  User Rating History ({len(history)} movies)")
    print(f"{'='*80}")
    print(f"  {'#':<4} {'Name':<40} {'Rating':<8} {'Genres':<25}")
    print(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*25}")

    for i, (_, row) in enumerate(history.iterrows(), 1):
        movie_id = row["movieId"]
        rating = row["rating"]
        info = get_movie_by_id(movies, movie_id)
        name = (info.get("NAME") or "Unknown")[:38] if info else "Unknown"
        genres = (info.get("GENRES") or "")[:23] if info else ""

        print(f"  {i:<4} {name:<40} {_format_score(rating):<8} {genres:<25}")

    print(f"{'='*80}\n")


def main():
    args = parse_args()
    cfg = get_config()
    model_dir = args.model_dir or cfg.get("output", {}).get("model_dir", "models")

    # 加载电影数据
    data_cfg = cfg["data"]
    movies_path = os.path.join(data_cfg["douban_dir"], data_cfg["movies_file"])
    ratings_path = os.path.join(data_cfg["douban_dir"], data_cfg["ratings_file"])

    if not os.path.exists(movies_path):
        print(f"❌ Movies file not found: {movies_path}")
        sys.exit(1)

    movies = load_movies(movies_path)

    # ── --list-models ──
    if args.list_models:
        available = _list_models(model_dir)
        if available:
            print(f"Available models in '{model_dir}':")
            for m in available:
                print(f"  - {m}")
        else:
            print(f"No models found in '{model_dir}'.")
            print("Run: python scripts/train.py --model all")
        return

    # ── --movie <id> ──
    if args.movie:
        info = get_movie_by_id(movies, args.movie)
        if info:
            info["MOVIE_ID"] = args.movie
            print_movie_detail(info)
        else:
            print(f"❌ Movie '{args.movie}' not found.")
        return

    # ── --hot ──
    if args.hot:
        model = load_model("popularity", model_dir)
        if model is None:
            return
        if args.genre:
            recs = model.get_popular_by_genre(args.genre, movies, n=args.topk)
            title = f"Top {args.topk} Popular Movies — Genre: {args.genre}"
        else:
            recs = model.recommend("dummy_user", n=args.topk, exclude_seen=False)
            title = f"Top {args.topk} Popular Movies"
        print_recommendations(recs, movies, title=title)
        return

    # ── --similar-to <movie_id> ──
    if args.similar_to:
        model = load_model(args.model, model_dir)
        if model is None:
            return
        try:
            recs = model.get_similar_movies(args.similar_to, n=args.topk)
        except NotImplementedError:
            print(f"❌ Model '{args.model}' does not support similar movie search.")
            print("   Try: --model item_knn or --model svd")
            return
        movie_info = get_movie_by_id(movies, args.similar_to)
        ref_name = movie_info.get("NAME") if movie_info else args.similar_to
        print_recommendations(recs, movies, title=f"Similar to: {ref_name} (Top {args.topk})")
        return

    # ── --user <id> --history ──
    if args.user and args.history:
        ratings = load_ratings(ratings_path)
        history = get_user_history(ratings, args.user)
        if history.empty:
            print(f"❌ No ratings found for user '{args.user}'.")
            return
        print_user_history(history, movies)
        return

    # ── --user <id> (recommendation) ──
    if args.user:
        model = load_model(args.model, model_dir)
        if model is None:
            return
        recs = model.recommend(args.user, n=args.topk, exclude_seen=True)
        print_recommendations(
            recs, movies,
            title=f"Recommendations for user '{args.user}' (Model: {args.model}, Top {args.topk})",
        )
        return

    # ── no action specified ──
    print("No action specified. Use --help for usage.")
    print("\nQuick examples:")
    print("  python scripts/recommend.py --user <id> --model svd --topk 10")
    print("  python scripts/recommend.py --similar-to <movie_id> --topk 10")
    print("  python scripts/recommend.py --hot --genre 动画")
    print("  python scripts/recommend.py --user <id> --history")
    print("  python scripts/recommend.py --movie <id>")


if __name__ == "__main__":
    main()
