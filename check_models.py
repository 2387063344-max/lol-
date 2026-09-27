"""连通性自检脚本：分别验证「生成模型」和「嵌入模型」是否真的能调通。

这是配环境时最容易踩坑的一步 —— 模型名写错、接口地址写错、Key 权限不对，
都会表现为"看起来配好了但一直答非所问"。先跑这个脚本能省掉大量排查时间。

用法：
    python check_models.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402

from src.lol_agent import llm  # noqa: E402

LINE = "-" * 62


def check_embedding() -> bool:
    print(LINE)
    print("【嵌入模型】")
    print(f"  地址：{config.EMBED_BASE_URL}")
    print(f"  模型：{config.EMBED_MODEL}")
    if not config.has_embed_api():
        print("  结果：未配置 EMBED_API_KEY，将使用离线哈希向量（效果很差）")
        return False
    try:
        vectors = llm.embed_texts(["亚索的风墙能挡飞行道具", "诺手叠满被动后伤害很高"])
    except Exception as exc:
        print(f"  结果：调用失败 -> {exc}")
        print("  排查：1) 模型名是否准确 2) 该 Key 是否有此模型权限 3) 接口地址是否带 /v2")
        return False
    dim = len(vectors[0]) if vectors else 0
    print(f"  结果：成功，向量维度 {dim}，共 {len(vectors)} 条")
    return True


def check_llm() -> bool:
    print(LINE)
    print("【生成模型】")
    print(f"  地址：{config.LLM_BASE_URL}")
    print(f"  模型：{config.LLM_MODEL}")
    if not config.has_llm():
        print("  结果：未配置 LLM_API_KEY，问答将降级为「仅检索」模式")
        return False
    try:
        text = llm.chat([{"role": "user", "content": "只回复两个字：正常"}])
    except Exception as exc:
        print(f"  结果：调用失败 -> {exc}")
        print("  排查：1) 模型名是否准确 2) Key 是否有效 3) 账户余额是否充足")
        return False
    print(f"  结果：成功，回复：{text[:40]}")
    return True


def main() -> None:
    print("模型连通性自检")
    ok_embed = check_embedding()
    ok_llm = check_llm()
    print(LINE)
    if ok_embed and ok_llm:
        print("全部通过。下一步：python build_kb.py --rebuild && python ask.py \"亚索 W 冷却是多少\"")
    else:
        print("存在未通过项。嵌入模型不通过时不要建索引；")
        print("嵌入模型一旦确认可用，记得执行 python build_kb.py --rebuild 重建向量库。")


if __name__ == "__main__":
    main()
