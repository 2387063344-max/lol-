"""全局配置：所有"会随供应商 / 版本变化"的参数集中在这里。

为什么要单独一个 config？
    知识库选型的一个隐藏坑是：模型名、版本号、阈值散落在代码各处，
    改一处漏一处。尤其是"嵌入模型名"——它决定了向量的语义空间，
    一旦更换，整个向量索引必须全量重建，否则新旧向量混在一起检索结果全错。
    因此这里统一定义，并由 build_index / vector_store 写入数据库做一致性校验。
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- 路径
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_DOCS_DIR = DATA_DIR / "raw" / "docs"
DB_PATH = DATA_DIR / "lol.db"
CHROMA_DIR = DATA_DIR / "chroma"
CACHE_DIR = DATA_DIR / "cache"

for _d in (DATA_DIR, RAW_DOCS_DIR, CHROMA_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


# ---------------------------------------------------------------- 数据源
DD_BASE = "https://ddragon.leagueoflegends.com"
DD_LANG = "zh_CN"
GAME_VERSION = _env("GAME_VERSION", "latest")  # "latest" 或写死如 "16.19.1"

# ---------------------------------------------------------------- 大模型
LLM_BASE_URL = _env("LLM_BASE_URL", "https://api.deepseek.com/v1")
LLM_API_KEY = _env("LLM_API_KEY")
# DeepSeek 的 V4.1 Flash 在 API 里的模型名就是 deepseek-flash（base_url 不变）
LLM_MODEL = _env("LLM_MODEL", "deepseek-flash")
LLM_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.2)
LLM_TIMEOUT = _env_float("LLM_TIMEOUT", 60)

# ---------------------------------------------------------------- 嵌入模型
# 百度千帆：OpenAI 兼容地址为 https://qianfan.baidubce.com/v2，
# 向量接口是 POST /v2/embeddings（注意不是 /v2/chat/completions）
EMBED_BASE_URL = _env("EMBED_BASE_URL", "https://qianfan.baidubce.com/v2")
EMBED_API_KEY = _env("EMBED_API_KEY")
# 千帆上的千问向量模型，注意必须全小写（实测驼峰写法 Qwen3-Embedding-0.6B 会报 no_such_model）：
#   qwen3-embedding-0.6b  -> 1024 维，便宜
#   qwen3-embedding-4b    -> 2560 维，效果更好
#   bge-large-zh          -> 1024 维，百度自研中文向量模型
EMBED_MODEL = _env("EMBED_MODEL", "qwen3-embedding-0.6b")
EMBED_TIMEOUT = _env_float("EMBED_TIMEOUT", 60)
# 千帆限制：除 tao-8k 外，单次请求的文本数量不能超过 16 条
EMBED_BATCH_SIZE = _env_int("EMBED_BATCH_SIZE", 16)
# 没配 Key 时的离线降级：用哈希构造稀疏向量，仅用于跑通流程，检索效果很差
EMBED_OFFLINE_DIM = _env_int("EMBED_OFFLINE_DIM", 512)

# ---------------------------------------------------------------- 联网搜索
SEARCH_PROVIDER = _env("SEARCH_PROVIDER", "tavily").lower()
SEARCH_API_KEY = _env("SEARCH_API_KEY")
SEARCH_TIMEOUT = _env_float("SEARCH_TIMEOUT", 20)
SEARCH_CACHE_TTL = _env_int("SEARCH_CACHE_TTL", 86400)

# ---------------------------------------------------------------- 检索参数
TOP_K = _env_int("TOP_K", 4)
# 余弦相似度阈值，低于它认为"本地知识库没找到"
SCORE_THRESHOLD = _env_float("SCORE_THRESHOLD", 0.35)
CHUNK_SIZE = _env_int("CHUNK_SIZE", 600)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 80)
CHROMA_COLLECTION = _env("CHROMA_COLLECTION", "lol_docs")


# ---------------------------------------------------------------- 能力探测
def has_llm() -> bool:
    """是否配置了生成模型。未配置时问答走"只检索不生成"的降级模式。"""
    return bool(LLM_API_KEY)


def has_embed_api() -> bool:
    """是否配置了在线嵌入模型。未配置时用离线哈希向量降级。"""
    return bool(EMBED_API_KEY)


def has_search() -> bool:
    return SEARCH_PROVIDER not in ("", "none") and bool(SEARCH_API_KEY)
