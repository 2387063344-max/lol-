"""检索路由：决定一个问题该去哪类知识库找答案。

为什么要先路由？
    如果所有问题都"向量检索一把梭"，会出现两类典型事故：
    - 问数值时召回了攻略里的一段旧数值，模型照着念，数值错误；
    - 问策略时命中了一堆无关英雄的数值表，答非所问。
    先分类再检索，是混合知识库能不能真正发挥作用的分水岭。

实现上采用「规则信号 + 模型分类」双保险：
    - 规则信号稳定、零成本、可解释（实体命中、技能位、时效性关键词）；
    - 模型负责处理规则覆盖不到的模糊表达；
    - 模型不可用或返回异常时，自动退回规则结果，保证系统永远有输出。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

import config

from .. import llm
from ..prompts import ROUTER_SYSTEM, ROUTER_USER_TEMPLATE
from ..store.sql_store import SqlStore
from ..types import Evidence

logger = logging.getLogger(__name__)


class Route(str, Enum):
    """知识来源类型。"""

    STRUCTURED = "structured"   # 数值/事实：走 SQLite
    VECTOR = "vector"           # 策略/克制：走 Chroma
    WEB = "web"                 # 最新改动：走联网
    HYBRID = "hybrid"           # 多路径合并


# 命中即认为"在问一个有标准答案的数值"
NUMERIC_SIGNALS = [
    "冷却", "CD", "cd", "多少", "几秒", "价格", "多少钱", "金币", "售价",
    "被动", "叫什么", "名称是", "数值是",
    "伤害", "基础", "成长", "攻速", "护甲", "魔抗", "生命值", "法力", "射程",
    "移速", "暴击", "属性", "数值", "加成", "百分比", "消耗", "耗蓝", "面板",
]

# 命中即认为"本地知识库可能过期，需要联网"
LATEST_SIGNALS = [
    "最新", "这版本", "当前版本", "新版本", "这赛季", "当前赛季", "补丁",
    "改了", "改动", "加强", "削弱", "削了", "增强了", "现在", "刚更新",
    "版本答案", "强势", "强度", "排行", "T0", "t0", "T1", "t1",
]

SLOT_RE = re.compile(r"(?<![A-Za-z])([QWERqwer])(?![A-Za-z])")


@dataclass
class RouteDecision:
    """路由决策结果。champions/items 用于精确查库，也用于向量库元数据过滤。"""

    route: Route = Route.HYBRID
    champions: list[str] = field(default_factory=list)
    items: list[str] = field(default_factory=list)
    runes: list[str] = field(default_factory=list)
    summoners: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)   # stats/skill/item/rune/summoner/lore
    slots: list[str] = field(default_factory=list)    # Q/W/E/R/passive
    needs_latest: bool = False
    confidence: float = 0.0
    reason: str = ""


class Router:
    def __init__(self, sql: SqlStore | None = None):
        self.sql = sql or SqlStore()
        self._champions: list[dict] | None = None
        self._items: list[str] | None = None
        self._runes: list[str] | None = None
        self._summoners: list[str] | None = None

    # ------------------------------------------------------------ 实体识别
    def _ensure_entities(self) -> None:
        if self._champions is None:
            try:
                self._champions = self.sql.champion_entities()
            except Exception as exc:
                logger.warning("加载英雄索引失败：%s", exc)
                self._champions = []
        if self._items is None:
            try:
                self._items = self.sql.item_names()
            except Exception as exc:
                logger.warning("加载装备索引失败：%s", exc)
                self._items = []
        if self._runes is None:
            try:
                self._runes = self.sql.rune_names()
            except Exception as exc:
                logger.warning("加载符文索引失败：%s", exc)
                self._runes = []
        if self._summoners is None:
            try:
                self._summoners = self.sql.summoner_names()
            except Exception as exc:
                logger.warning("加载召唤师技能索引失败：%s", exc)
                self._summoners = []

    def match_entities(self, question: str) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
        """规则匹配实体：英雄（内部 id）、装备名、技能位、符文名、召唤师技能名。"""
        self._ensure_entities()
        champions: list[str] = []
        for row in self._champions or []:
            for key in ("name", "title", "champ_id", "nickname"):
                value = row.get(key) or ""
                if value and value in question:
                    champions.append(row["champ_id"])
                    break

        items = [name for name in (self._items or []) if name and name in question]
        # 长名优先，避免"无尽"与"无尽之刃"重复；最多保留 3 个
        items = sorted(set(items), key=len, reverse=True)[:3]

        runes = [name for name in (self._runes or []) if name and name in question]
        summoners = [name for name in (self._summoners or []) if name and name in question]

        slots: list[str] = []
        if "被动" in question:
            slots.append("passive")
        for match in SLOT_RE.finditer(question):
            slots.append(match.group(1).upper())
        return champions, items, list(dict.fromkeys(slots)), runes[:2], summoners[:2]

    # ------------------------------------------------------------ 决策
    def decide(self, question: str) -> RouteDecision:
        champions, items, slots, runes, summoners = self.match_entities(question)
        needs_latest = any(signal in question for signal in LATEST_SIGNALS)
        numeric = any(signal in question for signal in NUMERIC_SIGNALS)

        decision = RouteDecision(
            champions=champions,
            items=items,
            slots=slots,
            runes=runes,
            summoners=summoners,
            needs_latest=needs_latest,
        )

        # ---- 规则兜底（无论如何都会先算出来）
        if numeric and slots and champions:
            decision.route, decision.fields, decision.confidence = Route.STRUCTURED, ["skill"], 0.7
        elif numeric and items:
            decision.route, decision.fields, decision.confidence = Route.STRUCTURED, ["item"], 0.7
        elif numeric and champions:
            decision.route, decision.fields, decision.confidence = Route.STRUCTURED, ["stats", "skill"], 0.6
        elif runes or summoners:
            decision.route, decision.confidence = Route.STRUCTURED, 0.6
            decision.fields = (["rune"] if runes else []) + (["summoner"] if summoners else [])
        elif needs_latest:
            decision.route, decision.confidence = Route.WEB, 0.6
        else:
            decision.route, decision.confidence = Route.VECTOR, 0.5
        decision.reason = "规则判定"

        # ---- 模型分类（可失败，失败不影响主流程）
        if config.has_llm():
            try:
                data = llm.chat_json(
                    [
                        {"role": "system", "content": ROUTER_SYSTEM},
                        {
                            "role": "user",
                            "content": ROUTER_USER_TEMPLATE.format(
                                question=question,
                                champions="、".join(champions) or "无",
                                items="、".join(items) or "无",
                                slots="、".join(slots) or "无",
                            ),
                        },
                    ]
                )
                self._merge_llm_result(decision, data)
            except Exception as exc:
                logger.warning("模型路由失败，沿用规则判定：%s", exc)
        return decision

    @staticmethod
    def _merge_llm_result(decision: RouteDecision, data: dict) -> None:
        route = str(data.get("route", "")).lower()
        if route in {r.value for r in Route}:
            decision.route = Route(route)
            decision.confidence = max(decision.confidence, float(data.get("confidence") or 0))
        for key, attr in (
            ("champions", "champions"),
            ("items", "items"),
            ("fields", "fields"),
            ("slots", "slots"),
            ("runes", "runes"),
            ("summoners", "summoners"),
        ):
            value = data.get(key)
            if isinstance(value, list) and value:
                setattr(decision, attr, [str(v) for v in value])
        decision.needs_latest = bool(decision.needs_latest or data.get("needs_latest"))
        if data.get("reason"):
            decision.reason = f"{decision.reason} + 模型：{data['reason']}"


def describe_evidence(evidences: list[Evidence]) -> str:
    """把证据列表渲染成给模型看的上下文，带 [n] 编号以便引用。"""
    if not evidences:
        return "（无参考资料）"
    blocks = []
    for idx, ev in enumerate(evidences, start=1):
        score = f"，相似度 {ev.score:.2f}" if ev.score is not None else ""
        blocks.append(f"[{idx}] 来源：{ev.citation()}{score}\n{ev.content}")
    return "\n\n".join(blocks)
