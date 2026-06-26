#!/usr/bin/env python3
"""
CineMagic 豆瓣电影数据爬虫

使用 Playwright 无头浏览器绕过豆瓣 JavaScript 反爬挑战，从电影页面抓取准确的元数据。
修复现有数据集中 46% 的电影豆瓣评分为 0.0 的问题。

依赖:
    pip install playwright beautifulsoup4 lxml
    python -m playwright install chromium

使用方式:
    # 修复 top200.csv 中缺失的评分
    python scripts/scraper.py

    # 只抓取指定电影
    python scripts/scraper.py --movie-id 1292052

    # 只修复评分缺失的电影
    python scripts/scraper.py --fix-score-only

    # 输出为 JSON 格式
    python scripts/scraper.py --format json --output data/douban/fixed.json
"""

import argparse
import csv
import json
import os
import random
import re
import sys
import time
from pathlib import Path

from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


# ── 豆瓣页面 URL ──
DOUBAN_MOVIE_URL = "https://movie.douban.com/subject/{}/"

# ── 页面等待策略 ──
# 豆瓣页面有三个关键等待标志:
# 1. 第一个页面的 JS 挑战会自动提交（~2秒）
# 2. rating_num 出现表示真实页面加载完成
WAIT_FOR_SELECTOR = "strong.ll.rating_num"  # 评分元素，真实页面必有
LOAD_TIMEOUT = 30_000  # 页面加载超时（ms）
WAIT_AFTER_LOAD = 2000  # 加载后额外等待（ms），让 JS 渲染完成

# 输出 CSV 列（与 top200.csv 一致）
OUTPUT_COLUMNS = [
    "MOVIE_ID", "NAME", "ALIAS", "ACTORS", "COVER",
    "DIRECTORS", "DOUBAN_SCORE", "DOUBAN_VOTES",
    "GENRES", "IMDB_ID", "LANGUAGES", "MINS",
    "OFFICIAL_SITE", "REGIONS", "RELEASE_DATE",
    "SLUG", "STORYLINE", "TAGS", "YEAR",
    "ACTOR_IDS", "DIRECTOR_IDS",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="CineMagic — 豆瓣电影数据爬虫 (Playwright)"
    )
    parser.add_argument(
        "--movie-id", type=str, default=None,
        help="只抓取指定电影ID"
    )
    parser.add_argument(
        "--fix-score-only", action="store_true",
        help="只抓取豆瓣评分为 0 的电影"
    )
    parser.add_argument(
        "--input", type=str, default=None,
        help="输入 CSV 路径（默认 data/douban/top200.csv）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出文件路径（默认 data/douban/top200_fixed.csv）"
    )
    parser.add_argument(
        "--format", type=str, choices=["csv", "json"], default="csv",
        help="输出格式（默认 csv）"
    )
    parser.add_argument(
        "--delay-min", type=float, default=2.0,
        help="请求最小间隔秒数（默认 2.0）"
    )
    parser.add_argument(
        "--delay-max", type=float, default=5.0,
        help="请求最大间隔秒数（默认 5.0）"
    )
    parser.add_argument(
        "--retries", type=int, default=3,
        help="每个请求的重试次数（默认 3）"
    )
    parser.add_argument(
        "--headless", type=bool, default=True,
        help="是否使用无头模式（默认 True）"
    )
    parser.add_argument(
        "--cache", type=str,
        default="data/douban/.scraper_cache.json",
        help="抓取缓存文件路径（支持断点续抓）"
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────
# 核心抓取
# ─────────────────────────────────────────────────────────

class DoubanScraper:
    """使用 Playwright 的豆瓣电影数据抓取器"""

    def __init__(
        self,
        delay_min: float = 2.0,
        delay_max: float = 5.0,
        max_retries: int = 3,
        headless: bool = True,
        cache_path: str | None = None,
    ):
        self.delay_min = delay_min
        self.delay_max = delay_max
        self.max_retries = max_retries
        self.headless = headless
        self.cache_path = cache_path
        self.cache = self._load_cache() if cache_path else {}
        self._last_request = 0.0
        self._browser = None
        self._context = None

    def start(self):
        """启动浏览器实例（所有请求共用同一个浏览器）"""
        self._playwright = sync_playwright().start()
        self._browser = self._playwright.chromium.launch(headless=self.headless)
        self._context = self._browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            locale="zh-CN",
        )

    def stop(self):
        """关闭浏览器"""
        if self._context:
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._playwright:
            self._playwright.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()

    # ── 公共接口 ──

    def fetch_movie(self, movie_id: str) -> dict | None:
        """获取单部电影的完整元数据"""
        if movie_id in self.cache:
            cached = self.cache[movie_id]
            if cached.get("DOUBAN_SCORE") and cached["DOUBAN_SCORE"] != 0.0:
                print(f"  [CACHE] {movie_id}")
                return cached

        data = self._fetch_page(movie_id)
        if data:
            self._save_cache(movie_id, data)
        return data

    def fetch_score_only(self, movie_id: str) -> dict | None:
        """只抓取评分（DOUBAN_SCORE, DOUBAN_VOTES）。其他字段抓到了也存。"""
        cached = self.cache.get(movie_id, {})
        if cached.get("DOUBAN_SCORE") and cached["DOUBAN_SCORE"] != 0.0:
            print(f"  [CACHE] {movie_id}")
            return cached

        data = self._fetch_page(movie_id)
        if data:
            self._save_cache(movie_id, data)
        return data

    # ── 页面抓取 ──

    def _fetch_page(self, movie_id: str) -> dict | None:
        """使用 Playwright 加载并解析豆瓣电影页面"""
        url = DOUBAN_MOVIE_URL.format(movie_id)
        page = None

        for attempt in range(self.max_retries):
            self._respect_delay()

            try:
                page = self._context.new_page()

                # 第一步：访问页面
                page.goto(url, timeout=LOAD_TIMEOUT, wait_until="domcontentloaded")

                # 第二步：等待真实页面加载（豆瓣有两层页面）
                try:
                    page.wait_for_selector(
                        WAIT_FOR_SELECTOR,
                        timeout=20_000,
                        state="attached",
                    )
                except PlaywrightTimeout:
                    # 可能是 JS 挑战页面，再等一会儿
                    page.wait_for_timeout(5_000)
                    try:
                        page.wait_for_selector(
                            WAIT_FOR_SELECTOR,
                            timeout=15_000,
                            state="attached",
                        )
                    except PlaywrightTimeout:
                        # 检查是否被拦截
                        current_url = page.url
                        if "sec.douban.com" in current_url or "deny" in current_url:
                            print(f"  [BLOCKED] {movie_id} — 豆瓣返回了拦截页面")
                        else:
                            print(f"  [TIMEOUT] {movie_id} — 评分元素未出现")
                        page.close()
                        continue

                # 第三步：额外等待 JS 渲染完成
                page.wait_for_timeout(WAIT_AFTER_LOAD)

                # 第四步：获取完整 HTML
                html = page.content()
                page.close()

                if len(html) < 5000:
                    print(f"  [SHORT] {movie_id} — HTML 过短 (可能仍被拦截)")
                    continue

                # 第五步：解析
                return self._parse_page(movie_id, html)

            except PlaywrightTimeout:
                print(f"  [TIMEOUT] {movie_id} — 页面加载超时")
                if page:
                    page.close()
            except Exception as e:
                print(f"  [ERROR] {movie_id} — {e}")
                if page:
                    try:
                        page.close()
                    except Exception:
                        pass

            if attempt < self.max_retries - 1:
                wait = (attempt + 1) * 3
                print(f"  [RETRY {attempt+1}/{self.max_retries}] {movie_id} — waiting {wait}s")
                time.sleep(wait)

        print(f"  [FAIL] {movie_id} — all {self.max_retries} attempts exhausted")
        return None

    # ── HTML 解析 ──

    def _parse_page(self, movie_id: str, html: str) -> dict:
        """
        解析豆瓣电影页面 HTML。

        关键数据位置:
        - 评分:   <strong class="ll rating_num" property="v:average">9.7</strong>
        - 评分数: <span property="v:votes">3298591</span>
        - 标题:   <meta property="og:title" content="肖申克的救赎 The Shawshank Redemption">
        - 年份:   <span class="year">(1994)</span>
        - 类型/演/时长等:  <div id="info"> 区域
        - 剧情:   <span property="v:summary"> 或 <div id="link-report-intra">
        - 标签:   <div class="tags-body">
        - JSON-LD: <script type="application/ld+json"> (结构化元数据)
        """
        soup = BeautifulSoup(html, "lxml")

        # ── JSON-LD 结构化数据（最可靠的数据源）──
        ld_data = {}
        ld_script = soup.find("script", type="application/ld+json")
        if ld_script:
            try:
                ld_data = json.loads(ld_script.string)
            except (json.JSONDecodeError, TypeError):
                pass

        # ── 评分 ──
        score_el = soup.find("strong", class_="ll rating_num")
        if not score_el:
            score_el = soup.find(attrs={"property": "v:average"})
        score = _parse_float(score_el.text) if score_el else 0.0

        # ── 评分人数 ──
        votes_el = soup.find(attrs={"property": "v:votes"})
        votes = _parse_float(votes_el.text) if votes_el else 0.0

        # ── 标题 ──
        name = ""
        og_title = soup.find("meta", attrs={"property": "og:title"})
        if og_title:
            raw_title = og_title.get("content", "").strip()
            # "肖申克的救赎 The Shawshank Redemption" → 去掉英文副标题
            # 或用 LD-JSON 的 name
            if ld_data:
                name = ld_data.get("name", "").strip()
            if not name:
                # 取纯中文部分
                name = raw_title
        else:
            title_el = soup.find("title")
            if title_el:
                name = re.sub(r"\s*\(豆瓣\).*", "", title_el.text).strip()
            elif ld_data:
                name = ld_data.get("name", "")

        # ── 别名 ──
        alias = ""
        if ld_data:
            aka = ld_data.get("alternateName", "")
            if isinstance(aka, list):
                alias = " / ".join(aka)
            else:
                alias = aka

        # ── IMDb ID ──
        imdb_id = ld_data.get("imdb_id", "")

        # ── 年份 ──
        year_el = soup.find("span", class_="year")
        year = 0.0
        if year_el:
            year = _parse_float(year_el.text.strip("()"))
        if year == 0.0:
            # 从 og:title 或 LD-JSON 提取
            if og_title:
                ym = re.search(r"\((\d{4})\)", og_title.get("content", ""))
                if ym:
                    year = float(ym.group(1))

        # ── #info 区域 ──
        info_text = ""
        info_el = soup.find("div", id="info")
        if info_el:
            info_text = info_el.get_text("\n", strip=True)
        info_dict = _parse_info_block(info_text)

        # ── 演员 ──
        actors = info_dict.get("主演", "")
        # 如果 LD-JSON 有更完整的列表
        if ld_data and "actor" in ld_data:
            actor_list = ld_data["actor"]
            if isinstance(actor_list, list):
                actors = " / ".join(
                    a.get("name", "") for a in actor_list if isinstance(a, dict)
                )

        # ── 导演 ──
        directors = info_dict.get("导演", "")
        if ld_data and "director" in ld_data:
            dir_list = ld_data["director"]
            if isinstance(dir_list, list):
                directors = " / ".join(
                    d.get("name", "") for d in dir_list if isinstance(d, dict)
                )

        # ── 剧情简介 ──
        storyline = self._extract_storyline(soup)

        # ── 标签 ──
        tags = self._extract_tags(soup)

        return {
            "MOVIE_ID": movie_id,
            "NAME": name,
            "ALIAS": alias or info_dict.get("又名", ""),
            "ACTORS": actors,
            "COVER": ld_data.get("image", ""),
            "DIRECTORS": directors,
            "DOUBAN_SCORE": score,
            "DOUBAN_VOTES": votes,
            "GENRES": info_dict.get("类型", ""),
            "IMDB_ID": imdb_id or info_dict.get("IMDb", ""),
            "LANGUAGES": info_dict.get("语言", ""),
            "MINS": _parse_mins(info_dict.get("片长", "")),
            "OFFICIAL_SITE": info_dict.get("官方网站", ""),
            "REGIONS": info_dict.get("制片国家/地区", ""),
            "RELEASE_DATE": info_dict.get("上映日期", ""),
            "SLUG": "",
            "STORYLINE": storyline,
            "TAGS": tags,
            "YEAR": year,
            "ACTOR_IDS": "",
            "DIRECTOR_IDS": "",
        }

    def _extract_storyline(self, soup: BeautifulSoup) -> str:
        """提取剧情简介"""
        # 优先从 property="v:summary"
        story_el = soup.find(attrs={"property": "v:summary"})
        if story_el:
            return _clean_text(story_el.get_text())

        # 从 #link-report-intra
        report_el = soup.find("div", id="link-report-intra")
        if report_el:
            # 有"展开"按钮的情况
            full_span = report_el.find("span", class_="all hidden")
            target = full_span if full_span else report_el.find("span")
            if target:
                return _clean_text(target.get_text(" ", strip=True))

        # LD-JSON
        ld_script = soup.find("script", type="application/ld+json")
        if ld_script:
            try:
                ld_data = json.loads(ld_script.string)
                desc = ld_data.get("description", "")
                if desc:
                    return _clean_text(desc)
            except (json.JSONDecodeError, TypeError):
                pass

        return ""

    def _extract_tags(self, soup: BeautifulSoup) -> str:
        """提取用户标签"""
        tags_body = soup.find("div", class_="tags-body")
        if not tags_body:
            return ""
        tag_texts = []
        for a in tags_body.find_all("a"):
            text = a.get_text(strip=True)
            if text:
                tag_texts.append(text)
        return " / ".join(tag_texts)

    # ── 缓存与延迟 ──

    def _load_cache(self) -> dict:
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save_cache(self, movie_id: str, data: dict):
        self.cache[movie_id] = data
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _respect_delay(self):
        """请求间隔随机化"""
        elapsed = time.time() - self._last_request
        if elapsed < self.delay_min:
            delay = random.uniform(self.delay_min, self.delay_max)
            time.sleep(delay - elapsed)


# ─────────────────────────────────────────────────────────
# HTML 解析辅助
# ─────────────────────────────────────────────────────────

def _parse_info_block(text: str) -> dict:
    """解析豆瓣 #info 区域的键值对文本"""
    result = {}
    field_map = {
        "导演": "导演",
        "编剧": "编剧",
        "主演": "主演",
        "类型": "类型",
        "制片国家/地区": "制片国家/地区",
        "语言": "语言",
        "上映日期": "上映日期",
        "片长": "片长",
        "又名": "又名",
        "IMDb": "IMDb",
        "官方网站": "官方网站",
    }

    for line in text.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        parts = line.split(":", 1)
        if len(parts) != 2:
            continue
        key = parts[0].strip()
        value = parts[1].strip()

        for cn_label in field_map:
            if cn_label in key:
                if result.get(cn_label):
                    result[cn_label] += " / " + value
                else:
                    result[cn_label] = value
                break

    return result


def _parse_float(text: str) -> float:
    """从文本中提取浮点数"""
    if not text:
        return 0.0
    cleaned = re.sub(r"[,\s ]", "", text.strip())
    try:
        return float(cleaned)
    except ValueError:
        match = re.search(r"[\d.]+", cleaned)
        return float(match.group()) if match else 0.0


def _parse_mins(text: str) -> float:
    """解析片长: '142分钟' → 142.0, '2小时30分钟' → 150.0"""
    if not text:
        return 0.0
    hours_match = re.search(r"(\d+)\s*小时", text)
    mins_match = re.search(r"(\d+)\s*分钟", text)
    total = 0.0
    if hours_match:
        total += float(hours_match.group(1)) * 60
    if mins_match:
        total += float(mins_match.group(1))
    if total == 0.0:
        total = _parse_float(text)
    return total


def _clean_text(text: str) -> str:
    """清洗文本：合并空白、去首尾空格"""
    return re.sub(r"\s+", " ", text).strip()


# ─────────────────────────────────────────────────────────
# 合并与输出
# ─────────────────────────────────────────────────────────

def merge_data(original_row: dict, scraped: dict) -> dict:
    """
    合并原始数据和抓取数据。

    规则:
    - DOUBAN_SCORE: 原始值 == 0 且抓取值 > 0 → 覆盖
    - DOUBAN_VOTES: 原始值 == 0 且抓取值 > 0 → 覆盖
    - 其他字段: 抓取值非空则覆盖原始空值
    """
    merged = dict(original_row)

    # 评分修复
    orig_score = float(original_row.get("DOUBAN_SCORE", 0) or 0)
    scraped_score = float(scraped.get("DOUBAN_SCORE", 0) or 0)
    if orig_score == 0.0 and scraped_score > 0:
        merged["DOUBAN_SCORE"] = str(scraped_score)

    orig_votes = float(original_row.get("DOUBAN_VOTES", 0) or 0)
    scraped_votes = float(scraped.get("DOUBAN_VOTES", 0) or 0)
    if scraped_votes > orig_votes:
        merged["DOUBAN_VOTES"] = str(scraped_votes)

    # 一般字段：新值非空则覆盖
    text_fields = [
        "NAME", "ALIAS", "ACTORS", "DIRECTORS",
        "GENRES", "IMDB_ID", "LANGUAGES", "MINS",
        "OFFICIAL_SITE", "REGIONS", "RELEASE_DATE",
        "SLUG", "STORYLINE", "TAGS", "YEAR",
        "ACTOR_IDS", "DIRECTOR_IDS", "COVER",
    ]
    for field in text_fields:
        new_val = scraped.get(field, "")
        if new_val and not str(new_val).strip() in ("", "0", "0.0"):
            merged[field] = str(new_val)

    return merged


def load_csv_dict(path: str) -> dict[str, dict]:
    """加载 CSV 为 {movie_id: row_dict}"""
    rows = {}
    with open(path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row.get("MOVIE_ID", "").strip()
            if mid:
                rows[mid] = row
    return rows


def save_csv(output_path: str, movies: list[dict]):
    """保存为 CSV"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for movie in movies:
            writer.writerow(movie)


def save_json(output_path: str, movies: list[dict]):
    """保存为 JSON"""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(movies, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────
# 主流程
# ─────────────────────────────────────────────────────────

def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    input_path = args.input or str(project_root / "data" / "douban" / "top200.csv")

    if args.format == "json":
        default_output = str(project_root / "data" / "douban" / "top200_fixed.json")
    else:
        default_output = str(project_root / "data" / "douban" / "top200_fixed.csv")
    output_path = args.output or default_output

    cache_path = args.cache or str(
        project_root / "data" / "douban" / ".scraper_cache.json"
    )

    print(f"{'='*60}")
    print(f"CineMagic — 豆瓣电影数据爬虫 (Playwright)")
    print(f"{'='*60}")
    print(f"Input:   {input_path}")
    print(f"Output:  {output_path} ({args.format})")
    print(f"Cache:   {cache_path}")
    print(f"Delay:   {args.delay_min}–{args.delay_max}s")
    print()

    # 加载原始数据
    original_rows = load_csv_dict(input_path)
    print(f"Loaded {len(original_rows)} movies from CSV")

    # 确定要抓取的电影ID
    if args.movie_id:
        movie_ids = [args.movie_id]
    elif args.fix_score_only:
        movie_ids = [
            mid for mid, row in original_rows.items()
            if float(row.get("DOUBAN_SCORE", 1) or 1) == 0.0
            and float(row.get("DOUBAN_VOTES", 0) or 0) > 50_000
        ]
    else:
        movie_ids = list(original_rows.keys())

    bad_count = sum(
        1 for mid in movie_ids
        if float(original_rows[mid].get("DOUBAN_SCORE", 1) or 1) == 0.0
        and float(original_rows[mid].get("DOUBAN_VOTES", 0) or 0) > 50_000
    )
    print(f"Movies to scrape: {len(movie_ids)}")
    print(f"With bad scores:  {bad_count}")
    print()

    # 抓取
    results = []
    success = 0
    fail = 0
    fixed = 0

    with DoubanScraper(
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        max_retries=args.retries,
        headless=args.headless,
        cache_path=cache_path,
    ) as scraper:

        for i, mid in enumerate(movie_ids):
            orig = original_rows.get(mid, {})
            name = (orig.get("NAME") or f"movie_{mid}")[:35]

            print(f"[{i+1}/{len(movie_ids)}] {mid} — {name}")

            if args.fix_score_only:
                scraped = scraper.fetch_score_only(mid)
            else:
                scraped = scraper.fetch_movie(mid)

            if scraped:
                merged = merge_data(orig, scraped) if orig else scraped
                results.append(merged)

                # 报告修复
                old_score = float(orig.get("DOUBAN_SCORE", 1) or 1) if orig else 1
                new_score = float(merged.get("DOUBAN_SCORE", 0) or 0)
                if old_score == 0.0 and new_score > 0:
                    print(f"  ✓ Score: 0.0 → {new_score}")
                    fixed += 1
                else:
                    print(f"  ✓ Done (Score: {new_score})")
                success += 1
            else:
                results.append(dict(orig) if orig else {"MOVIE_ID": mid})
                fail += 1
            print()

    # 保存
    if args.format == "json":
        save_json(output_path, results)
    else:
        save_csv(output_path, results)

    print(f"{'='*60}")
    print(f"DONE — {success} succeeded, {fail} failed, {fixed} scores fixed")
    print(f"Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
