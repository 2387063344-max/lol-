"""模型访问层：统一封装「生成」与「嵌入」两类调用（OpenAI 兼容协议）。

设计要点：
1. 不绑定任何具体厂商。只要接口是 OpenAI 兼容的（DeepSeek / 通义 / 智谱 / OpenAI），
   改 .env 里的 BASE_URL + KEY + MODEL 即可切换，代码零改动。
2. 密钥只从配置读取，日志里绝不打印请求体或密钥。
3. 提供离线降级：没配 Key 时用哈希向量代替真实嵌入，
   让初学者能先把整条链路跑通（效果很差，仅用于理解流程）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Iterable, Sequence

import numpy as np
import requests

import config

logger = logging.getLogger(__name__)


class ModelError(RuntimeError):
    """模型调用失败。调用方应捕获后降级，而不是让整个问答崩掉。"""


class NoModelConfigured(ModelError):
    """未配置对应的模型 Key。"""


# --------------------------------------------------------------------- 生成
def chat(messages: list[dict], temperature: float | None = None) -> str:
    """调用聊天补全接口，返回文本。"""
    if not config.has_llm():
        raise NoModelConfigured("未配置 LLM_API_KEY，无法生成答案（可先配置后重试）")

    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": config.LLM_TEMPERATURE if temperature is None else temperature,
    }
    data = _post_json(f"{config.LLM_BASE_URL.rstrip('/')}/chat/completions", payload, config.LLM_API_KEY, config.LLM_TIMEOUT)
    try:
        return data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise ModelError(f"聊天接口返回结构异常：{str(data)[:200]}") from exc


def chat_json(messages: list[dict], temperature: float = 0.0) -> dict:
    """要求模型返回 JSON，并容错处理 ```json 代码围栏。"""
    text = chat(messages, temperature=temperature)
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end == -1:
        raise ModelError(f"模型未返回 JSON：{text[:200]}")
    try:
        return json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ModelError(f"JSON 解析失败：{cleaned[:200]}") from exc


# --------------------------------------------------------------------- 嵌入
def embed_texts(texts: Sequence[str], batch_size: int | None = None) -> list[list[float]]:
    """把一批文本转成向量。未配置在线嵌入模型时自动降级为离线哈希向量。"""
    texts = [t if t.strip() else " " for t in texts]
    if not config.has_embed_api():
        logger.warning("未配置 EMBED_API_KEY，使用离线哈希向量（仅供跑通流程，检索质量很差）")
        return [_offline_embed(t, config.EMBED_OFFLINE_DIM) for t in texts]

    url = f"{config.EMBED_BASE_URL.rstrip('/')}/embeddings"
    # 各家对单次请求条数有限制（千帆除 tao-8k 外最多 16 条），取配置值并按上限保护
    batch_size = min(batch_size or config.EMBED_BATCH_SIZE, 16)
    vectors: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        payload = {"model": config.EMBED_MODEL, "input": batch}
        data = _post_json(url, payload, config.EMBED_API_KEY, config.EMBED_TIMEOUT)
        try:
            vectors.extend([item["embedding"] for item in data["data"]])
        except (KeyError, TypeError) as exc:
            raise ModelError(f"嵌入接口返回结构异常：{str(data)[:200]}") from exc
    if len(vectors) != len(texts):
        raise ModelError(f"嵌入数量不匹配：期望 {len(texts)}，实际 {len(vectors)}")
    return vectors


def embed_one(text: str) -> list[float]:
    return embed_texts([text])[0]


def embed_model_name() -> str:
    """当前实际使用的嵌入模型名（离线降级时有特殊标记，用于索引一致性校验）。"""
    return config.EMBED_MODEL if config.has_embed_api() else f"offline-hash-{config.EMBED_OFFLINE_DIM}"


def _offline_embed(text: str, dim: int) -> list[float]:
    """离线降级向量：按字/词哈希到固定维度并归一化。

    它只表达"字面重合度"，没有语义泛化能力，所以「亚索」和「疾风剑豪」不会相似。
    这正是用来体会"真实嵌入模型为什么重要"的对照组。
    """
    vec = np.zeros(dim, dtype=np.float32)
    lowered = text.lower()
    tokens: Iterable[str] = re.findall(r"[a-z0-9]+", lowered)
    han = re.findall(r"[\u4e00-\u9fff]", lowered)
    tokens = list(tokens) + han + [han[i] + han[i + 1] for i in range(len(han) - 1)]
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        num = int(digest, 16)
        vec[num % dim] += 1.0 if (num >> 8) & 1 else -1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec.tolist()


# --------------------------------------------------------------------- 底层
def _post_json(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise ModelError(f"请求模型服务失败：{exc}") from exc

    if resp.status_code != 200:
        # 只记录状态码与简短原因，避免把密钥或完整报文写进日志
        raise ModelError(f"模型服务返回 {resp.status_code}：{resp.text[:200]}")
    try:
        return resp.json()
    except ValueError as exc:
        raise ModelError("模型服务返回非 JSON 内容") from exc
