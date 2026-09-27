"""构建向量索引：切片 → 嵌入 → 写入 Chroma。

为什么要单独一个建索引的步骤，而不是边查边算？
    - 嵌入要花钱/花时间，只做一次；查询时只嵌入问题那一条。
    - 索引是"快照"，必须能被整体重建：换嵌入模型、换切片策略都要重建，
      否则库里会混进语义空间不一致的旧向量，检索结果不可信。

用法：
    python build_kb.py            # 增量构建（模型没变就追加）
    python build_kb.py --rebuild  # 全量重建（换嵌入模型后必须执行）
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3

import config

from .. import llm
from ..store.sql_store import SqlStore
from .docs_loader import load_chunks

logger = logging.getLogger(__name__)


def _read_index_meta(key: str) -> str | None:
    try:
        conn = sqlite3.connect(config.DB_PATH)
        row = conn.execute("SELECT value FROM index_meta WHERE name = ?", (key,)).fetchone()
        conn.close()
        return row[0] if row else None
    except sqlite3.Error:
        return None


def _write_index_meta(items: dict) -> None:
    conn = sqlite3.connect(config.DB_PATH)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS index_meta ("
            "name TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL DEFAULT (datetime('now')))"
        )
        for key, value in items.items():
            conn.execute(
                "INSERT INTO index_meta (name, value, updated_at) VALUES (?,?,datetime('now')) "
                "ON CONFLICT(name) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                (key, str(value)),
            )
        conn.commit()
    finally:
        conn.close()


def build(rebuild: bool = False) -> dict:
    """构建（或重建）向量知识库。"""
    import chromadb
    from chromadb.config import Settings

    store = SqlStore()
    try:
        version = store.version()
        aliases = store.champion_names()
        entities = store.champion_entities()
    finally:
        store.close()

    chunks = load_chunks(config.RAW_DOCS_DIR, champion_aliases=aliases, default_version=version or "")
    if not chunks:
        return {"ok": False, "reason": f"{config.RAW_DOCS_DIR} 下没有可索引的文档（.md/.txt）", "chunks": 0}

    # 统一把英雄别名归一到官方 champ_id：
    # 否则"亚索"（文档里写的）和 "Yasuo"（路由识别出的）在元数据过滤时对不上。
    alias_map: dict[str, str] = {}
    for row in entities or []:
        for key in ("name", "title", "champ_id", "champ_key"):
            value = row.get(key)
            if value:
                alias_map[value] = row["champ_id"]
    for chunk in chunks:
        chunk.champions = list(dict.fromkeys(alias_map.get(c, c) for c in chunk.champions))

    model_name = llm.embed_model_name()

    # 一致性校验：换嵌入模型必须重建，否则新旧向量混检
    old_model = _read_index_meta("embed_model")
    if old_model and old_model != model_name and not rebuild:
        return {
            "ok": False,
            "reason": f"嵌入模型已从 {old_model} 变为 {model_name}，向量空间不一致，必须加 --rebuild 全量重建",
            "chunks": 0,
        }

    client = chromadb.PersistentClient(
        path=str(config.CHROMA_DIR), settings=Settings(anonymized_telemetry=False)
    )
    if rebuild:
        try:
            client.delete_collection(config.CHROMA_COLLECTION)
        except Exception:  # 集合不存在时忽略
            pass
    collection = client.get_or_create_collection(
        name=config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"}
    )

    logger.info("开始嵌入 %d 个切片（模型：%s）", len(chunks), model_name)
    texts = [c.text for c in chunks]
    vectors = llm.embed_texts(texts)

    ids, documents, metadatas = [], [], []
    for idx, chunk in enumerate(chunks):
        ids.append(f"{chunk.source}::{idx}")
        documents.append(chunk.text)
        metadatas.append(
            {
                "source": chunk.source,
                # Chroma 的 where 只支持标量，主英雄单独存一个字段用于预过滤
                "champion_primary": chunk.champions[0] if chunk.champions else "",
                "champions": ",".join(chunk.champions),
                "roles": ",".join(chunk.roles),
                "version": chunk.version or "",
                "title": chunk.title,
            }
        )
    collection.upsert(ids=ids, documents=documents, embeddings=vectors, metadatas=metadatas)

    _write_index_meta(
        {
            "embed_model": model_name,
            "embed_dim": len(vectors[0]) if vectors else 0,
            "chunk_count": len(chunks),
            "game_version": version or "",
            "collection": config.CHROMA_COLLECTION,
        }
    )

    result = {
        "ok": True,
        "chunks": len(chunks),
        "embed_model": model_name,
        "dim": len(vectors[0]) if vectors else 0,
        "game_version": version,
        "rebuild": rebuild,
        "persist_dir": str(config.CHROMA_DIR),
    }
    logger.info("索引构建完成：%s", json.dumps(result, ensure_ascii=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="构建攻略文档的向量索引")
    parser.add_argument("--rebuild", action="store_true", help="删除旧集合后全量重建")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = build(rebuild=args.rebuild)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result.get("ok"):
        raise SystemExit(result.get("reason", "构建失败"))


if __name__ == "__main__":
    main()
