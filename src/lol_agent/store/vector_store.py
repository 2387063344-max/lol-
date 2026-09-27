"""非结构化知识库访问层：Chroma 向量检索。

为什么策略类知识必须走向量库？
    "上单诺手怎么打"这类问题没有唯一正确答案，靠的是语义相近。
    关键词匹配（BM25/数据库 LIKE）无法处理"克制关系""节奏"这类表达，
    只有把文本变成向量后比相似度才召回得到。

讲究：
    - 检索时先按元数据（英雄）过滤，再做向量相似度 —— 减少跨英雄的噪声召回。
    - 返回的相似度用于兜底判断：低于阈值说明本地知识库没覆盖，该转联网了。
"""

from __future__ import annotations

import logging

import config

from .. import llm
from ..types import Evidence

logger = logging.getLogger(__name__)


class VectorStore:
    """Chroma 向量库封装。库不存在时所有方法安全返回空，不让主流程崩掉。"""

    def __init__(self, collection_name: str = config.CHROMA_COLLECTION):
        self.collection_name = collection_name
        self._collection = None

    # -------------------------------------------------------------- 状态
    def _ensure(self):
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
            )
            self._collection = client.get_collection(self.collection_name)
        return self._collection

    def is_ready(self) -> bool:
        try:
            return self._ensure().count() > 0
        except Exception:  # 库未构建 / chromadb 未安装
            return False

    def count(self) -> int:
        try:
            return self._ensure().count()
        except Exception:
            return 0

    # -------------------------------------------------------------- 检索
    def search(
        self, query: str, top_k: int | None = None, champions: list[str] | None = None
    ) -> list[Evidence]:
        """语义检索。champions 用于元数据预过滤。"""
        try:
            collection = self._ensure()
        except Exception as exc:
            logger.info("向量库不可用（可能尚未构建索引）：%s", exc)
            return []

        top_k = top_k or config.TOP_K
        try:
            query_vec = llm.embed_one(query)
        except Exception as exc:
            logger.warning("生成查询向量失败：%s", exc)
            return []

        results = self._query(collection, query_vec, top_k, champions)
        if not results and champions:
            # 预过滤可能把所有切片都过滤掉了，退化为不带过滤再查一次
            results = self._query(collection, query_vec, top_k, None)

        evidences: list[Evidence] = []
        for doc, meta, distance in results:
            # Chroma 使用 cosine 空间时 distance = 1 - 余弦相似度
            similarity = max(0.0, min(1.0, 1.0 - float(distance)))
            evidences.append(
                Evidence(
                    content=doc,
                    source=meta.get("source", "攻略文档"),
                    kind="vector",
                    score=similarity,
                    title=meta.get("title", ""),
                )
            )
        return evidences

    def _query(self, collection, query_vec, top_k: int, champions: list[str] | None):
        where = None
        if champions:
            # 始终保留空字符串，保证"通用攻略"不被过滤掉
            values = list(dict.fromkeys([c for c in champions if c] + [""]))
            where = {"champion_primary": {"$in": values}}
        try:
            res = collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("带过滤检索失败，退化为全量检索：%s", exc)
            res = collection.query(
                query_embeddings=[query_vec], n_results=top_k, include=["documents", "metadatas", "distances"]
            )

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        return list(zip(docs, metas, dists))


def embed_model_mismatch() -> str | None:
    """检查当前嵌入模型与建库时是否一致，不一致返回提示语。"""
    import sqlite3

    try:
        conn = sqlite3.connect(config.DB_PATH)
        row = conn.execute("SELECT value FROM index_meta WHERE name = 'embed_model'").fetchone()
        conn.close()
    except sqlite3.Error:
        return None
    if not row:
        return None
    current = llm.embed_model_name()
    if row[0] != current:
        return f"建库用的是 {row[0]}，当前配置是 {current}，请执行 python build_kb.py --rebuild"
    return None
