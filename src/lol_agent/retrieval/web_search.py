"""联网搜索兜底：解决本地知识库天然滞后的问题。

触发策略必须保守 —— 每次联网都要花钱、都要等。
本项目只在两种情况下联网：
    1. 问题命中"最新版本/这次补丁/现在强度"等时效性信号；
    2. 本地检索为空，或向量相似度低于阈值（说明本地确实没有）。

另外：搜索结果会落本地缓存（默认 24 小时），同一个问题不重复计费。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import requests

import config

from ..types import Evidence

logger = logging.getLogger(__name__)


class SearchError(RuntimeError):
    pass


class NoSearchConfigured(SearchError):
    pass


# ------------------------------------------------------------------ 缓存
def _cache_path(query: str) -> Path:
    digest = hashlib.md5(f"{config.SEARCH_PROVIDER}::{query}".encode("utf-8")).hexdigest()
    return config.CACHE_DIR / f"search_{digest}.json"


def _read_cache(query: str) -> list[Evidence] | None:
    path = _cache_path(query)
    if not path.exists():
        return None
    if time.time() - path.stat().st_mtime > config.SEARCH_CACHE_TTL:
        path.unlink(missing_ok=True)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return [Evidence(**item) for item in data]
    except Exception:
        return None


def _write_cache(query: str, evidences: list[Evidence]) -> None:
    path = _cache_path(query)
    payload = [
        {
            "content": e.content,
            "source": e.source,
            "kind": e.kind,
            "score": e.score,
            "title": e.title,
            "url": e.url,
            "fetched_at": e.fetched_at,
        }
        for e in evidences
    ]
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ 供应商
def _search_tavily(query: str, max_results: int) -> list[Evidence]:
    resp = requests.post(
        "https://api.tavily.com/search",
        json={
            "api_key": config.SEARCH_API_KEY,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": True,
        },
        timeout=config.SEARCH_TIMEOUT,
    )
    if resp.status_code != 200:
        raise SearchError(f"Tavily 返回 {resp.status_code}：{resp.text[:200]}")
    data = resp.json()
    evidences: list[Evidence] = []
    answer = (data.get("answer") or "").strip()
    if answer:
        evidences.append(
            Evidence(content=answer, source="Tavily 摘要", kind="web", title="联网摘要", score=None)
        )
    for item in data.get("results") or []:
        evidences.append(
            Evidence(
                content=(item.get("content") or "").strip(),
                source=item.get("url", "联网结果"),
                kind="web",
                title=item.get("title", ""),
                url=item.get("url", ""),
            )
        )
    return evidences


def _search_serper(query: str, max_results: int) -> list[Evidence]:
    resp = requests.post(
        "https://google.serper.dev/search",
        headers={"X-API-KEY": config.SEARCH_API_KEY, "Content-Type": "application/json"},
        json={"q": query, "num": max_results},
        timeout=config.SEARCH_TIMEOUT,
    )
    if resp.status_code != 200:
        raise SearchError(f"Serper 返回 {resp.status_code}：{resp.text[:200]}")
    data = resp.json()
    evidences: list[Evidence] = []
    for item in data.get("organic") or []:
        evidences.append(
            Evidence(
                content=(item.get("snippet") or "").strip(),
                source=item.get("link", "联网结果"),
                kind="web",
                title=item.get("title", ""),
                url=item.get("link", ""),
            )
        )
    return evidences


def _search_bocha(query: str, max_results: int) -> list[Evidence]:
    resp = requests.post(
        "https://api.bochaai.com/v1/web-search",
        headers={"Authorization": f"Bearer {config.SEARCH_API_KEY}", "Content-Type": "application/json"},
        json={"query": query, "count": max_results, "summary": True},
        timeout=config.SEARCH_TIMEOUT,
    )
    if resp.status_code != 200:
        raise SearchError(f"博查返回 {resp.status_code}：{resp.text[:200]}")
    data = resp.json()
    pages = (((data.get("data") or {}).get("webPages") or {}).get("value")) or []
    evidences: list[Evidence] = []
    for item in pages:
        content = (item.get("summary") or item.get("snippet") or "").strip()
        evidences.append(
            Evidence(
                content=content,
                source=item.get("url", "联网结果"),
                kind="web",
                title=item.get("name", ""),
                url=item.get("url", ""),
            )
        )
    return evidences


PROVIDERS = {"tavily": _search_tavily, "serper": _search_serper, "bocha": _search_bocha}


# ------------------------------------------------------------------ 对外
def search(query: str, max_results: int = 5) -> list[Evidence]:
    """执行联网搜索。未配置时抛 NoSearchConfigured，由调用方降级处理。"""
    if not config.has_search():
        raise NoSearchConfigured(
            f"未配置联网搜索（SEARCH_PROVIDER={config.SEARCH_PROVIDER or '未设置'}），"
            "可在 .env 中填入 SEARCH_API_KEY"
        )

    cached = _read_cache(query)
    if cached is not None:
        logger.info("命中搜索缓存：%s", query)
        return cached

    provider = PROVIDERS.get(config.SEARCH_PROVIDER)
    if provider is None:
        raise SearchError(f"不支持的搜索服务：{config.SEARCH_PROVIDER}")

    fetched_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    evidences = provider(query, max_results)
    for ev in evidences:
        ev.fetched_at = fetched_at
    _write_cache(query, evidences)
    logger.info("联网搜索完成，获得 %d 条结果", len(evidences))
    return evidences
