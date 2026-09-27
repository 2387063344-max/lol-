"""跨层共用的数据结构。

Evidence（证据）是整个系统的核心契约：
无论来自 SQLite、Chroma 还是联网搜索，都必须统一成这个结构，
这样上层生成答案时不必关心它从哪来，也天然保证了"回答可溯源"。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evidence:
    """一条检索到的证据。"""

    content: str          # 喂给模型的正文
    source: str           # 来源标识：DataDragon 16.19.1 / 文档名 / 网页 URL
    kind: str             # structured | vector | web
    score: float | None = None   # 相似度（结构化证据为 None，表示"精确命中"）
    title: str = ""       # 人类可读标题，用于界面展示
    url: str = ""         # 联网证据的链接
    fetched_at: str = ""  # 联网证据的获取时间，用于判断时效性

    def citation(self) -> str:
        """界面/答案里展示的来源标注。"""
        if self.kind == "web":
            return f"{self.title or self.source}（{self.url}）"
        if self.kind == "vector":
            return f"{self.title or '攻略文档'}（{self.source}）"
        return self.source


@dataclass
class Chunk:
    """攻略文档切片。

    讲究：切片必须携带元数据（英雄 / 分路 / 版本 / 出处），
    检索时先按元数据过滤再做相似度，能显著减少"张冠李戴"的噪声召回。
    """

    text: str
    source: str = ""
    title: str = ""
    champions: list[str] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    version: str = ""
