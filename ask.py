"""命令行问答入口，方便调试与观察路由决策。

用法：
    python ask.py "亚索 W 技能冷却是多少"
    python ask.py                    # 进入交互式提问
    python ask.py "..." --json       # 输出结构化结果（含命中路径与来源）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.lol_agent.agent import LoLAgent  # noqa: E402


def print_answer(answer) -> None:
    print("\n" + "=" * 60)
    print(f"问题：{answer.question}")
    print("-" * 60)
    print(answer.text)
    print("-" * 60)
    print(f"路由：{answer.route.value}")
    if answer.decision:
        print(
            f"识别实体：英雄={answer.decision.champions or '无'} "
            f"装备={answer.decision.items or '无'} "
            f"技能位={answer.decision.slots or '无'}"
        )
        print(f"判定理由：{answer.decision.reason}")
    print(f"是否联网：{'是' if answer.used_web else '否'}    耗时：{answer.elapsed_ms} ms")
    print("\n来源：")
    print(answer.sources_text())
    for note in answer.notes:
        print(f"[提示] {note}")


def main() -> None:
    parser = argparse.ArgumentParser(description="英雄联盟知识问答（命令行）")
    parser.add_argument("question", nargs="?", help="要问的问题；留空进入交互模式")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本处理")
    parser.add_argument("--health", action="store_true", help="只检查三层知识库状态")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    agent = LoLAgent()

    if args.health:
        print(json.dumps(agent.health(), ensure_ascii=False, indent=2))
        return

    if args.question:
        answer = agent.ask(args.question)
        if args.json:
            print(
                json.dumps(
                    {
                        "question": answer.question,
                        "route": answer.route.value,
                        "used_web": answer.used_web,
                        "elapsed_ms": answer.elapsed_ms,
                        "degraded": answer.degraded,
                        "answer": answer.text,
                        "sources": [e.citation() for e in answer.evidences],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        else:
            print_answer(answer)
        return

    print("已进入交互模式，输入 exit 退出。")
    while True:
        try:
            question = input("\n你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("exit", "quit", "退出"):
            break
        if question:
            print_answer(agent.ask(question))


if __name__ == "__main__":
    main()
