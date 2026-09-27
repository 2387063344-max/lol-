"""问答主流程：路由 → 检索 → 上下文合并 → 生成 → 附来源引用。

整个系统的产品契约都在这一层体现：
    - 数值只能来自 SQLite，模型只负责组织语言；
    - 本地检索不到就如实说明，而不是让模型"编一个像样的答案"；
    - 每条回答都带来源，用户可以自己核验（这对"用来学习"的场景是刚需）。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import config

from . import llm
from .prompts import NO_EVIDENCE_ANSWER, NO_LLM_NOTICE, SYSTEM_QA, USER_QA_TEMPLATE
from .retrieval import web_search
from .retrieval.router import Route, Router, RouteDecision, describe_evidence
from .store.sql_store import (
    SqlStore,
    champion_to_evidence,
    item_to_evidence,
    rune_to_evidence,
    skill_to_evidence,
    summoner_to_evidence,
)
from .store.vector_store import VectorStore, embed_model_mismatch
from .types import Evidence

logger = logging.getLogger(__name__)


@dataclass
class Answer:
    question: str
    text: str
    evidences: list[Evidence] = field(default_factory=list)
    route: Route = Route.HYBRID
    decision: RouteDecision | None = None
    used_web: bool = False
    elapsed_ms: int = 0
    degraded: bool = False          # 未配置模型，只给出检索结果
    notes: list[str] = field(default_factory=list)

    def sources_text(self) -> str:
        if not self.evidences:
            return "（无来源）"
        lines = []
        for idx, ev in enumerate(self.evidences, start=1):
            score = f"（相似度 {ev.score:.2f}）" if ev.score is not None else ""
            lines.append(f"[{idx}] {ev.citation()} {score}")
        return "\n".join(lines)


class LoLAgent:
    """英雄联盟混合知识库问答智能体。"""

    def __init__(self, sql: SqlStore | None = None, vector: VectorStore | None = None):
        self.sql = sql or SqlStore()
        self.vector = vector or VectorStore()
        self.router = Router(self.sql)

    # ------------------------------------------------------------------ 主入口
    def ask(self, question: str) -> Answer:
        started = time.time()
        question = (question or "").strip()
        notes: list[str] = []

        if not question:
            return Answer(question=question, text="请输入一个关于英雄联盟的问题。", notes=["空问题"])

        # 1) 路由：决定去哪类知识库找
        decision = self.router.decide(question)
        logger.info("路由决策：%s（%s）", decision.route.value, decision.reason)

        # 2) 检索
        evidences = self._retrieve(question, decision)
        if not self.sql.is_ready():
            notes.append("结构化库未初始化，请先运行 `python ingest_data.py`")

        # 3) 是否需要联网兜底
        used_web = False
        if self._need_web(decision, evidences):
            try:
                web_evidences = web_search.search(question)
                if web_evidences:
                    evidences.extend(web_evidences)
                    used_web = True
            except web_search.NoSearchConfigured as exc:
                notes.append(str(exc))
            except Exception as exc:
                notes.append(f"联网搜索失败，已降级为仅本地知识库：{exc}")

        # 4) 生成
        text, degraded = self._generate(question, evidences)

        return Answer(
            question=question,
            text=text,
            evidences=evidences,
            route=decision.route,
            decision=decision,
            used_web=used_web,
            elapsed_ms=int((time.time() - started) * 1000),
            degraded=degraded,
            notes=notes,
        )

    # ------------------------------------------------------------------ 检索
    def _retrieve(self, question: str, decision: RouteDecision) -> list[Evidence]:
        evidences: list[Evidence] = []
        if decision.route in (Route.STRUCTURED, Route.HYBRID, Route.WEB):
            evidences.extend(self._retrieve_structured(decision))
        if decision.route in (Route.VECTOR, Route.HYBRID, Route.WEB):
            evidences.extend(self.vector.search(question, champions=decision.champions or None))
        return evidences

    def _retrieve_structured(self, decision: RouteDecision) -> list[Evidence]:
        """所有数值都在这里查出来 —— 模型拿到的永远是真实数据。"""
        if not self.sql.is_ready():
            return []
        version = self.sql.version()
        fields = decision.fields or ["stats", "skill"]
        evidences: list[Evidence] = []

        for champ_ref in decision.champions[:2]:
            row = self.sql.get_champion(champ_ref)
            if row is None:  # 模型可能给的是中文名而非内部 id
                rows = self.sql.resolve_champion(champ_ref, limit=1)
                if not rows:
                    continue
                row = rows[0]

            if "stats" in fields:
                evidences.append(champion_to_evidence(row, version))
            if "skill" in fields:
                skills = self.sql.get_skills(row["champ_id"])
                targets = {s.upper() for s in decision.slots} if decision.slots else None
                for skill in skills:
                    if targets is None or skill["slot"].upper() in targets:
                        evidences.append(skill_to_evidence(skill, row["name"], version))
            if "lore" in fields and row.get("lore"):
                evidences.append(
                    Evidence(
                        content=row["lore"][:1500],
                        source=f"DataDragon {version}",
                        kind="structured",
                        title=f"{row['name']} 背景故事",
                    )
                )

        for item_name in decision.items[:2]:
            rows = self.sql.resolve_item(item_name, limit=1)
            if rows:
                evidences.append(item_to_evidence(rows[0], version))
        for rune_name in decision.runes[:2]:
            rows = self.sql.resolve_rune(rune_name, limit=1)
            if rows:
                evidences.append(rune_to_evidence(rows[0], version))
        for spell_name in decision.summoners[:2]:
            rows = self.sql.resolve_summoner(spell_name, limit=1)
            if rows:
                evidences.append(summoner_to_evidence(rows[0], version))
        return evidences

    @staticmethod
    def _need_web(decision: RouteDecision, evidences: list[Evidence]) -> bool:
        """联网触发条件：宁可保守，也不让每次提问都花钱。"""
        if decision.needs_latest or decision.route == Route.WEB:
            return True
        if not evidences:
            return True
        if any(ev.kind == "structured" for ev in evidences):
            return False  # 已经查到精确数值，不必联网
        best = max((ev.score or 0.0) for ev in evidences if ev.kind == "vector")
        return best < config.SCORE_THRESHOLD

    # ------------------------------------------------------------------ 生成
    def _generate(self, question: str, evidences: list[Evidence]) -> tuple[str, bool]:
        """返回 (答案文本, 是否为降级输出)。"""
        if not evidences:
            return NO_EVIDENCE_ANSWER, True

        if not config.has_llm():
            degraded_text = NO_LLM_NOTICE + "\n\n".join(
                f"[{i}] {ev.citation()}\n{ev.content}" for i, ev in enumerate(evidences, start=1)
            )
            return degraded_text, True

        context = describe_evidence(evidences)
        try:
            answer = llm.chat(
                [
                    {"role": "system", "content": SYSTEM_QA},
                    {
                        "role": "user",
                        "content": USER_QA_TEMPLATE.format(question=question, context=context),
                    },
                ]
            )
        except Exception as exc:
            logger.warning("生成失败，降级为直接展示检索结果：%s", exc)
            fallback = f"（模型调用失败：{exc}）\n\n" + "\n\n".join(
                f"[{i}] {ev.citation()}\n{ev.content}" for i, ev in enumerate(evidences, start=1)
            )
            return fallback, True
        return answer, False

    # ------------------------------------------------------------------ 体检
    def health(self) -> dict:
        """一次性检查三层知识库的状态，便于定位"为什么答不好"。"""
        mismatch = embed_model_mismatch()
        return {
            "game_version": self.sql.version() if self.sql.is_ready() else None,
            "structured_ready": self.sql.is_ready(),
            "vector_ready": self.vector.is_ready(),
            "vector_chunks": self.vector.count(),
            "llm_ready": config.has_llm(),
            "embed_model": llm.embed_model_name(),
            "search_ready": config.has_search(),
            "embed_model_mismatch": mismatch,
        }
