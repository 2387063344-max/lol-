"""攻略文档加载与切片。

切片是 RAG 效果的第一杀手，讲究有三条：
1. **按语义边界切，不按固定字数硬切**。
   优先用 Markdown 标题切分（一个标题下的内容是一个完整语义单元），
   只有超长段落才退化成滑窗切分，且保留 overlap 避免句子被拦腰截断。
2. **每个切片必须带元数据**（英雄 / 分路 / 版本 / 出处）。
   检索时先按元数据过滤再做相似度，能显著减少"张冠李戴"。
3. **切片不是越小越好**。切太碎会丢失上下文，模型拿到了碎片也答不对。

支持的文档格式：Markdown / 纯文本，放在 data/raw/docs 下。
可用 YAML 风格 front-matter 显式声明元数据：

    ---
    champion: 亚索
    role: 中单
    version: 16.19
    ---
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import config

from ..types import Chunk

logger = logging.getLogger(__name__)

ROLE_KEYWORDS = {
    "上单": "上单", "上路": "上单", "top": "上单",
    "中单": "中单", "中路": "中单", "mid": "中单",
    "打野": "打野", "jungle": "打野", "jg": "打野",
    "下路": "下路", "bot": "下路", "adc": "下路", "射手": "下路",
    "辅助": "辅助", "support": "辅助",
}

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_front_matter(text: str) -> tuple[dict, str]:
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    meta: dict = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            meta[key.strip().lower()] = value.strip()
    return meta, text[match.end() :]


def _split_by_heading(body: str) -> list[tuple[str, str]]:
    """按标题切成 (标题路径, 正文) 列表。"""
    matches = list(HEADING_RE.finditer(body))
    if not matches:
        return [("", body.strip())]

    blocks: list[tuple[str, str]] = []
    prefix = body[: matches[0].start()].strip()
    if prefix:
        blocks.append(("", prefix))

    stack: list[tuple[int, str]] = []
    for idx, match in enumerate(matches):
        level, title = len(match.group(1)), match.group(2).strip()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(body)
        content = body[match.end() : end].strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = " > ".join(t for _, t in stack)
        if content:
            blocks.append((path, content))
    return blocks


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    """超长文本退化成滑窗切分，优先在句号/换行处断开。"""
    if len(text) <= size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            for sep in ("\n", "。", "；", "！", "？", ". "):
                pos = text.rfind(sep, start + size // 2, end)
                if pos != -1:
                    end = pos + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def _detect(text: str, aliases: list[str]) -> list[str]:
    """识别切片里出现过的英雄名（去重，保持出现顺序）。"""
    found: list[str] = []
    for alias in aliases:
        if alias and alias in text:
            # 同一英雄的"中文名/称号/英文 id"可能被同时命中，取第一个即可
            if not any(alias in f or f in alias for f in found):
                found.append(alias)
    return found


def _detect_roles(text: str) -> list[str]:
    lowered = text.lower()
    roles: list[str] = []
    for keyword, role in ROLE_KEYWORDS.items():
        if keyword.lower() in lowered and role not in roles:
            roles.append(role)
    return roles


def load_chunks(
    docs_dir: Path | str = config.RAW_DOCS_DIR,
    champion_aliases: list[str] | None = None,
    default_version: str = "",
) -> list[Chunk]:
    """加载目录下所有 .md / .txt 文档并切片。"""
    docs_dir = Path(docs_dir)
    aliases = champion_aliases or []
    chunks: list[Chunk] = []

    if not docs_dir.exists():
        logger.warning("文档目录不存在：%s", docs_dir)
        return chunks

    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in (".md", ".txt", ".markdown"):
            continue
        raw = path.read_text(encoding="utf-8", errors="ignore")
        if not raw.strip():
            continue
        meta, body = _parse_front_matter(raw)
        declared_champions = [c.strip() for c in (meta.get("champion") or "").split("、") if c.strip()]
        declared_roles = [r.strip() for r in (meta.get("role") or "").split("、") if r.strip()]
        version = meta.get("version") or default_version

        for title_path, content in _split_by_heading(body):
            for piece in _split_long(content, config.CHUNK_SIZE, config.CHUNK_OVERLAP):
                full_text = f"{title_path}\n{piece}" if title_path else piece
                champions = declared_champions or _detect(full_text, aliases)
                roles = declared_roles or _detect_roles(full_text)
                chunks.append(
                    Chunk(
                        text=full_text.strip(),
                        source=path.name,
                        title=title_path or path.stem,
                        champions=champions,
                        roles=roles,
                        version=version,
                    )
                )

    logger.info("共切出 %d 个切片（来自 %s）", len(chunks), docs_dir)
    return chunks
