"""采集层：把拳头官方 Data Dragon 静态数据写入 SQLite。

为什么数值必须来自官方接口，而不是爬攻略或让模型回答？
    - 攻略里的数值常常过期；模型凭记忆回答更是会编造。
    - Data Dragon 免费、免 Key、按版本发布，且官方出中文包，
      天然带版本号 —— 这一条同时解决了"数值准确"和"版本时效性"两个需求。

可用接口（已实测）：
    /api/versions.json                                 版本列表
    /cdn/{v}/data/zh_CN/champion.json                  英雄清单（不含技能）
    /cdn/{v}/data/zh_CN/champion/{ChampionId}.json      英雄详情（含技能/属性/背景）
    /cdn/{v}/data/zh_CN/item.json                       装备
    /cdn/{v}/data/zh_CN/runesReforged.json              符文
    /cdn/{v}/data/zh_CN/summoner.json                   召唤师技能
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests

import config

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).resolve().parent.parent / "store" / "schema.sql"

# Data Dragon 的定位标签是英文，转成中文方便检索与展示
TAG_CN = {
    "Fighter": "战士",
    "Tank": "坦克",
    "Mage": "法师",
    "Assassin": "刺客",
    "Marksman": "射手",
    "Support": "辅助",
}

SLOT_BY_INDEX = ["Q", "W", "E", "R"]


# ------------------------------------------------------------------ 抓取
def fetch_json(url: str, retries: int = 3, timeout: float = 20) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()
            last_error = RuntimeError(f"HTTP {resp.status_code}")
        except requests.RequestException as exc:
            last_error = exc
    raise RuntimeError(f"抓取失败 {url}：{last_error}")


def list_versions() -> list[str]:
    return fetch_json(f"{config.DD_BASE}/api/versions.json")


def resolve_version(version: str | None = None) -> str:
    """'latest'（默认）解析为官方最新版本。"""
    version = version or config.GAME_VERSION
    if version and version.lower() != "latest":
        return version
    return list_versions()[0]


def _clean_html(text: str | None) -> str:
    if not text:
        return ""
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    text = text.replace("&lt;", "<").replace("&gt;", ">")
    return re.sub(r"[ \t]+", " ", text).strip()


def _resolve_tokens(text: str | None, spell: dict) -> str:
    """把 tooltip 里的 {{ e1 }} / {{ qdamage }} 占位符替换成真实数值。

    不替换的话，用户看到的是"造成 {{ qdamage }} 物理伤害"，这是不能接受的。
    """
    if not text:
        return ""
    var_map: dict[str, str] = {}
    for var in spell.get("vars") or []:
        key, coeff = var.get("key"), var.get("coeff")
        if key is None or coeff is None:
            continue
        var_map[str(key).lower()] = "/".join(str(c) for c in coeff) if isinstance(coeff, list) else str(coeff)
    effect_burn = spell.get("effectBurn") or []

    def repl(match: re.Match) -> str:
        token = match.group(1).strip().lower()
        if token.startswith("e") and token[1:].isdigit():
            idx = int(token[1:])
            if idx < len(effect_burn) and effect_burn[idx] is not None:
                return str(effect_burn[idx])
            return ""  # 无法解析的占位符直接去掉，避免用户看到 {{ }} 这种原始标记
        return var_map.get(token, "")

    return re.sub(r"\{\{\s*([^}]+?)\s*\}\}", repl, text)


# ------------------------------------------------------------------ 入库
def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))


def _save_champion(conn: sqlite3.Connection, version: str, detail: dict, with_lore: bool) -> None:
    champ_id = detail["id"]
    tags = [TAG_CN.get(t, t) for t in (detail.get("tags") or [])]

    conn.execute(
        """INSERT OR REPLACE INTO champions
           (version, champ_id, champ_key, name, title, tags, partype, info, stats, blurb, lore)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (
            version,
            champ_id,
            detail.get("key", ""),
            detail.get("name", ""),
            detail.get("title", ""),
            json.dumps(tags, ensure_ascii=False),
            detail.get("partype", ""),
            json.dumps(detail.get("info") or {}, ensure_ascii=False),
            json.dumps(detail.get("stats") or {}, ensure_ascii=False),
            _clean_html(detail.get("blurb")),
            _clean_html(detail.get("lore")) if with_lore else "",
        ),
    )

    passive = detail.get("passive") or {}
    if passive.get("name"):
        conn.execute(
            """INSERT OR REPLACE INTO champion_skills
               (version, champ_id, slot, name, description, tooltip, cooldown, cost, range)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                version,
                champ_id,
                "passive",
                passive.get("name", ""),
                _clean_html(passive.get("description")),
                "",
                "",
                "",
                "",
            ),
        )

    for idx, spell in enumerate(detail.get("spells") or []):
        slot = SLOT_BY_INDEX[idx] if idx < len(SLOT_BY_INDEX) else f"S{idx + 1}"
        conn.execute(
            """INSERT OR REPLACE INTO champion_skills
               (version, champ_id, slot, name, description, tooltip, cooldown, cost, range)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                version,
                champ_id,
                slot,
                spell.get("name", ""),
                _clean_html(spell.get("description")),
                _resolve_tokens(_clean_html(spell.get("tooltip")), spell),
                spell.get("cooldownBurn", ""),
                spell.get("costBurn", ""),
                spell.get("rangeBurn", ""),
            ),
        )


def _save_items(conn: sqlite3.Connection, version: str, data: dict) -> int:
    count = 0
    for item_id, item in (data.get("data") or {}).items():
        name = (item.get("name") or "").strip()
        if not name:
            continue
        gold = item.get("gold") or {}
        conn.execute(
            """INSERT OR REPLACE INTO items
               (version, item_id, name, description, plaintext, gold_total, gold_sell,
                tags, stats, from_items, into_items)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                version,
                item_id,
                name,
                _clean_html(item.get("description")),
                _clean_html(item.get("plaintext")),
                gold.get("total"),
                gold.get("sell"),
                json.dumps(item.get("tags") or [], ensure_ascii=False),
                json.dumps(item.get("stats") or {}, ensure_ascii=False),
                json.dumps(item.get("from") or [], ensure_ascii=False),
                json.dumps(item.get("into") or [], ensure_ascii=False),
            ),
        )
        count += 1
    return count


def _save_runes(conn: sqlite3.Connection, version: str, families: list) -> int:
    count = 0
    for family in families or []:
        family_name = family.get("name", "")
        for slot in family.get("slots") or []:
            for rune in slot.get("runes") or []:
                conn.execute(
                    """INSERT OR REPLACE INTO runes
                       (version, rune_id, key_name, name, family, short_desc, long_desc)
                       VALUES (?,?,?,?,?,?,?)""",
                    (
                        version,
                        rune.get("id"),
                        rune.get("key", ""),
                        rune.get("name", ""),
                        family_name,
                        _clean_html(rune.get("shortDesc")),
                        _clean_html(rune.get("longDesc")),
                    ),
                )
                count += 1
    return count


def _save_summoner_spells(conn: sqlite3.Connection, version: str, data: dict) -> int:
    count = 0
    for spell_id, spell in (data.get("data") or {}).items():
        conn.execute(
            """INSERT OR REPLACE INTO summoner_spells
               (version, spell_id, name, description, cooldown) VALUES (?,?,?,?,?)""",
            (
                version,
                spell_id,
                spell.get("name", ""),
                _clean_html(spell.get("description")),
                spell.get("cooldownBurn", ""),
            ),
        )
        count += 1
    return count


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value, updated_at) VALUES (?,?,datetime('now')) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
        (key, value),
    )


# ------------------------------------------------------------------ 主流程
def ingest(version: str | None = None, with_lore: bool = True, workers: int = 10) -> dict:
    """执行一次完整采集，返回统计信息。"""
    version = resolve_version(version)
    logger.info("开始采集 Data Dragon 版本 %s", version)

    summary = fetch_json(f"{config.DD_BASE}/cdn/{version}/data/{config.DD_LANG}/champion.json")
    champ_ids = sorted((summary.get("data") or {}).keys())
    logger.info("英雄共 %d 个，开始拉取详情（并发 %d）", len(champ_ids), workers)

    details: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                fetch_json,
                f"{config.DD_BASE}/cdn/{version}/data/{config.DD_LANG}/champion/{cid}.json",
            ): cid
            for cid in champ_ids
        }
        for future in as_completed(futures):
            cid = futures[future]
            try:
                details[cid] = future.result()["data"][cid]
            except Exception as exc:  # 单个英雄失败不影响整体
                logger.warning("英雄 %s 详情拉取失败：%s", cid, exc)

    items = fetch_json(f"{config.DD_BASE}/cdn/{version}/data/{config.DD_LANG}/item.json")
    runes = fetch_json(f"{config.DD_BASE}/cdn/{version}/data/{config.DD_LANG}/runesReforged.json")
    summoner = fetch_json(f"{config.DD_BASE}/cdn/{version}/data/{config.DD_LANG}/summoner.json")

    conn = sqlite3.connect(config.DB_PATH)
    try:
        _init_db(conn)
        for champ_id in champ_ids:
            if champ_id in details:
                _save_champion(conn, version, details[champ_id], with_lore)
        item_count = _save_items(conn, version, items)
        rune_count = _save_runes(conn, version, runes)
        spell_count = _save_summoner_spells(conn, version, summoner)
        _set_meta(conn, "game_version", version)
        _set_meta(conn, "ingested_at", "")
        conn.commit()
    finally:
        conn.close()

    result = {
        "version": version,
        "champions": len(details),
        "items": item_count,
        "runes": rune_count,
        "summoner_spells": spell_count,
        "db": str(config.DB_PATH),
    }
    logger.info("采集完成：%s", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="采集 Data Dragon 官方数据到 SQLite")
    parser.add_argument("--version", default=None, help="游戏版本，默认取 .env 的 GAME_VERSION（latest）")
    parser.add_argument("--no-lore", action="store_true", help="不抓背景故事，速度更快")
    parser.add_argument("--workers", type=int, default=10, help="并发数")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = ingest(version=args.version, with_lore=not args.no_lore, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
