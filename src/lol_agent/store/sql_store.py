"""结构化知识库访问层：所有数值查询都走这里。

设计讲究（这是本项目最关键的一条）：
    数值绝不交给向量检索，也不交给模型记忆，而是"先解析出实体 → 参数化查库 →
    把查到的真实结果交给模型组织语言"。
    这样即使模型想编造，它手里也只有真实数据，从工程上根除了数值幻觉。

    另外：所有查询都用 ? 占位符传参，不做字符串拼接 SQL。
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import config

from ..types import Evidence

# 玩家常用但官方中英文名里都没有的俗称（Data Dragon 只提供官方名，俗称需人工补）
NICKNAMES = {
    "Darius": "诺手",
    "Fiora": "剑姬",
    "Tryndamere": "蛮王",
    "MasterYi": "剑圣",
    "MissFortune": "女枪",
    "Irelia": "刀妹",
    "Yone": "永恩",
    "Sett": "腕豪",
    "Camille": "青钢影",
    "Renekton": "鳄鱼",
}


# 属性字段的中文标签（Data Dragon 的 key 是英文，直接展示对新手不友好）
STAT_LABELS = {
    "hp": "生命值",
    "hpperlevel": "每级生命值",
    "mp": "法力值",
    "mpperlevel": "每级法力值",
    "movespeed": "移动速度",
    "armor": "护甲",
    "armorperlevel": "每级护甲",
    "spellblock": "魔法抗性",
    "spellblockperlevel": "每级魔法抗性",
    "attackrange": "攻击距离",
    "hpregen": "生命回复",
    "hpregenperlevel": "每级生命回复",
    "mpregen": "法力回复",
    "mpregenperlevel": "每级法力回复",
    "crit": "暴击几率",
    "critperlevel": "每级暴击几率",
    "attackdamage": "攻击力",
    "attackdamageperlevel": "每级攻击力",
    "attackspeed": "攻击速度",
    "attackspeedperlevel": "每级攻击速度",
}


# 装备属性字段的中文标签（Data Dragon 的 key 是内部命名，直接展示看不懂）
ITEM_STAT_LABELS = {
    "FlatPhysicalDamageMod": "攻击力",
    "FlatMagicDamageMod": "法术强度",
    "FlatHPPoolMod": "生命值",
    "FlatMPPoolMod": "法力值",
    "FlatArmorMod": "护甲",
    "FlatSpellBlockMod": "魔法抗性",
    "FlatCritChanceMod": "暴击几率",
    "FlatCritDamageMod": "暴击伤害",
    "FlatAttackSpeedMod": "攻击速度",
    "PercentAttackSpeedMod": "攻击速度",
    "FlatMovementSpeedMod": "移动速度",
    "PercentMovementSpeedMod": "移动速度",
    "FlatHPRegenMod": "生命回复",
    "FlatMPRegenMod": "法力回复",
    "PercentLifeStealMod": "生命偷取",
    "FlatSpellVampMod": "法术吸血",
    "PercentMagicPenetrationMod": "法术穿透",
    "FlatMagicPenetrationMod": "法术穿透",
    "PercentArmorPenetrationMod": "护甲穿透",
    "FlatArmorPenetrationMod": "护甲穿透",
    "FlatBlockMod": "格挡",
    "PercentBaseHPRegenMod": "基础生命回复",
    "PercentHealingMod": "治疗与护盾强度",
    "FlatTenacityMod": "韧性",
}


def _format_item_stat(key: str, value) -> str:
    label = ITEM_STAT_LABELS.get(key, key)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return f"{label} {value}"
    if key.startswith("Percent") or key in ("FlatCritChanceMod", "FlatSpellCritChanceMod"):
        return f"{label} +{number * 100:g}%"
    return f"{label} +{number:g}"


def _json_loads(text: str | None, default):
    if not text:
        return default
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return default


class SqlStore:
    """SQLite 只读访问封装。"""

    def __init__(self, db_path: Path | str = config.DB_PATH):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    # -------------------------------------------------------------- 基础
    def close(self) -> None:
        self.conn.close()

    def _query(self, sql: str, params: tuple = ()) -> list[dict]:
        cur = self.conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]

    def version(self) -> str | None:
        rows = self._query("SELECT value FROM meta WHERE key = 'game_version'")
        return rows[0]["value"] if rows else None

    def is_ready(self) -> bool:
        return self.version() is not None

    # -------------------------------------------------------------- 实体解析
    def resolve_champion(self, keyword: str, limit: int = 5) -> list[dict]:
        """把用户口语里的称呼解析成英雄记录。

        匹配顺序：中文名精确 → 称号精确 → 英文 id → 中文名包含 → 称号包含。
        """
        keyword = (keyword or "").strip()
        if not keyword:
            return []
        v = self.version()
        if not v:
            return []

        for column in ("name", "title", "champ_id", "champ_key"):
            rows = self._query(
                f"SELECT * FROM champions WHERE version = ? AND {column} = ? COLLATE NOCASE LIMIT ?",
                (v, keyword, limit),
            )
            if rows:
                return rows
        rows = self._query(
            "SELECT * FROM champions WHERE version = ? AND (name LIKE ? OR title LIKE ?) LIMIT ?",
            (v, f"%{keyword}%", f"%{keyword}%", limit),
        )
        if rows:
            return rows
        return self._query(
            "SELECT * FROM champions WHERE version = ? AND champ_id LIKE ? LIMIT ?",
            (v, f"%{keyword}%", limit),
        )

    def resolve_item(self, keyword: str, limit: int = 5) -> list[dict]:
        keyword = (keyword or "").strip()
        if not keyword:
            return []
        v = self.version()
        if not v:
            return []
        rows = self._query(
            "SELECT * FROM items WHERE version = ? AND name = ? LIMIT ?", (v, keyword, limit)
        )
        if rows:
            return rows
        return self._query(
            "SELECT * FROM items WHERE version = ? AND name LIKE ? ORDER BY gold_total LIMIT ?",
            (v, f"%{keyword}%", limit),
        )

    def resolve_rune(self, keyword: str, limit: int = 5) -> list[dict]:
        keyword = (keyword or "").strip()
        if not keyword or not self.version():
            return []
        rows = self._query(
            "SELECT * FROM runes WHERE version = ? AND name = ? LIMIT ?", (self.version(), keyword, limit)
        )
        if rows:
            return rows
        return self._query(
            "SELECT * FROM runes WHERE version = ? AND (name LIKE ? OR family LIKE ?) LIMIT ?",
            (self.version(), f"%{keyword}%", f"%{keyword}%", limit),
        )

    def resolve_summoner(self, keyword: str, limit: int = 5) -> list[dict]:
        keyword = (keyword or "").strip()
        if not keyword or not self.version():
            return []
        return self._query(
            "SELECT * FROM summoner_spells WHERE version = ? AND name LIKE ? LIMIT ?",
            (self.version(), f"%{keyword}%", limit),
        )

    def champion_names(self) -> list[str]:
        """全部英雄的可检索别名（中文名 + 称号 + 英文 id），供实体识别用。"""
        if not self.version():
            return []
        rows = self._query(
            "SELECT champ_id, name, title FROM champions WHERE version = ?", (self.version(),)
        )
        names: list[str] = []
        for row in rows:
            names.extend([row["name"], row["title"], row["champ_id"], NICKNAMES.get(row["champ_id"], "")])
        return [n for n in names if n]

    def champion_entities(self) -> list[dict]:
        """英雄实体的全部可匹配名（中文名 / 称号 / 英文 id），供规则识别用。

        返回内部 champ_id，避免"亚索"和"疾风剑豪"被当成两个英雄。
        """
        if not self.version():
            return []
        rows = self._query(
            "SELECT champ_id, name, title FROM champions WHERE version = ?", (self.version(),)
        )
        for row in rows:
            row["nickname"] = NICKNAMES.get(row["champ_id"], "")
        return rows

    def item_names(self) -> list[str]:
        if not self.version():
            return []
        rows = self._query("SELECT name FROM items WHERE version = ?", (self.version(),))
        return [r["name"] for r in rows if r["name"]]

    def rune_names(self) -> list[str]:
        if not self.version():
            return []
        rows = self._query("SELECT name FROM runes WHERE version = ?", (self.version(),))
        return [r["name"] for r in rows if r["name"]]

    def summoner_names(self) -> list[str]:
        if not self.version():
            return []
        rows = self._query("SELECT name FROM summoner_spells WHERE version = ?", (self.version(),))
        return [r["name"] for r in rows if r["name"]]

    # -------------------------------------------------------------- 数值查询
    def get_champion(self, champ_id: str) -> dict | None:
        rows = self._query(
            "SELECT * FROM champions WHERE version = ? AND champ_id = ?", (self.version(), champ_id)
        )
        return rows[0] if rows else None

    def get_skills(self, champ_id: str, slot: str | None = None) -> list[dict]:
        sql = "SELECT * FROM champion_skills WHERE version = ? AND champ_id = ?"
        params: list = [self.version(), champ_id]
        if slot:
            sql += " AND slot = ? COLLATE NOCASE"
            params.append(slot)
        sql += " ORDER BY CASE slot WHEN 'passive' THEN 0 WHEN 'Q' THEN 1 WHEN 'W' THEN 2 WHEN 'E' THEN 3 ELSE 4 END"
        return self._query(sql, tuple(params))

    def get_items_by_ids(self, item_ids: list[str]) -> list[dict]:
        if not item_ids:
            return []
        placeholders = ",".join("?" * len(item_ids))
        return self._query(
            f"SELECT * FROM items WHERE version = ? AND item_id IN ({placeholders})",
            (self.version(), *item_ids),
        )


# ------------------------------------------------------------------ 格式化
def _source(version: str | None) -> str:
    return f"DataDragon {version}" if version else "DataDragon"


def champion_to_evidence(row: dict, version: str | None) -> Evidence:
    """英雄基础属性 → 证据。"""
    stats = _json_loads(row.get("stats"), {})
    tags = _json_loads(row.get("tags"), [])
    info = _json_loads(row.get("info"), {})
    lines = [
        f"英雄：{row['name']}（{row['title']} / {row['champ_id']}）",
        f"定位：{'、'.join(tags) if tags else '未知'}",
        f"资源类型：{row.get('partype') or '未知'}",
    ]
    if info:
        lines.append(
            "能力倾向：" + "，".join(f"{k} {v}/10" for k, v in info.items())
        )
    if stats:
        lines.append("基础属性（1级）与成长：")
        for key in ("hp", "mp", "movespeed", "armor", "spellblock", "attackdamage", "attackspeed", "attackrange", "hpregen", "mpregen"):
            if key in stats:
                label = STAT_LABELS.get(key, key)
                grow_key = f"{key}perlevel"
                grow = f"，每级 +{stats[grow_key]}" if grow_key in stats else ""
                lines.append(f"  - {label}：{stats[key]}{grow}")
    return Evidence(
        content="\n".join(lines),
        source=_source(version),
        kind="structured",
        title=f"{row['name']} 基础属性",
    )


def skill_to_evidence(row: dict, champ_name: str, version: str | None) -> Evidence:
    slot_cn = {"passive": "被动", "Q": "Q", "W": "W", "E": "E", "R": "R"}.get(row["slot"], row["slot"])
    lines = [f"{champ_name} {slot_cn} 技能：{row['name']}"]
    if row.get("cooldown"):
        lines.append(f"冷却时间：{row['cooldown']}")
    if row.get("cost") not in (None, "", "0"):
        lines.append(f"消耗：{row['cost']}")
    if row.get("range") not in (None, "", "0"):
        lines.append(f"技能射程：{row['range']}")
    if row.get("description"):
        lines.append(f"说明：{row['description']}")
    if row.get("tooltip"):
        lines.append(f"详细数值：{row['tooltip']}")
    return Evidence(
        content="\n".join(lines),
        source=_source(version),
        kind="structured",
        title=f"{champ_name} {slot_cn}：{row['name']}",
    )


def item_to_evidence(row: dict, version: str | None) -> Evidence:
    stats = _json_loads(row.get("stats"), {})
    lines = [
        f"装备：{row['name']}",
        f"总价格：{row.get('gold_total')} 金币（售价 {row.get('gold_sell')}）",
    ]
    if row.get("plaintext"):
        lines.append(f"一句话：{row['plaintext']}")
    if stats:
        lines.append("属性：" + "，".join(_format_item_stat(k, v) for k, v in stats.items()))
    if row.get("description"):
        lines.append(f"说明：{row['description']}")
    from_items = _json_loads(row.get("from_items"), [])
    if from_items:
        lines.append(f"合成所需组件 ID：{', '.join(from_items)}")
    into_items = _json_loads(row.get("into_items"), [])
    if into_items:
        lines.append(f"可合成上级装备 ID：{', '.join(into_items)}")
    return Evidence(
        content="\n".join(lines),
        source=_source(version),
        kind="structured",
        title=f"装备 {row['name']}",
    )


def rune_to_evidence(row: dict, version: str | None) -> Evidence:
    lines = [f"符文：{row['name']}（{row['family']}系）"]
    if row.get("short_desc"):
        lines.append(f"效果：{row['short_desc']}")
    if row.get("long_desc"):
        lines.append(f"详细说明：{row['long_desc']}")
    return Evidence(
        content="\n".join(lines), source=_source(version), kind="structured", title=f"符文 {row['name']}"
    )


def summoner_to_evidence(row: dict, version: str | None) -> Evidence:
    lines = [f"召唤师技能：{row['name']}", f"冷却：{row.get('cooldown')} 秒"]
    if row.get("description"):
        lines.append(f"说明：{row['description']}")
    return Evidence(
        content="\n".join(lines),
        source=_source(version),
        kind="structured",
        title=f"召唤师技能 {row['name']}",
    )
