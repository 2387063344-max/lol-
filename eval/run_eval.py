"""评测脚本：量化"这个智能体到底好不好"。

AI 产品经理最容易忽略的一步就是评测。没有评测集，你就无法回答：
    - 换了个嵌入模型，效果变好还是变差？
    - 调了阈值，是召回变多了还是噪声变多了？
    - 这次 badcase 是"没召回"还是"召回了但没用上"？

用法：
    python eval/run_eval.py                    # 跑全量
    python eval/run_eval.py --category numeric # 只跑数值类
    python eval/run_eval.py --limit 10

输出：控制台汇总 + eval/reports/report_<时间戳>.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.lol_agent.agent import LoLAgent  # noqa: E402

QA_SET = Path(__file__).resolve().parent / "qa_set.jsonl"
REPORT_DIR = Path(__file__).resolve().parent / "reports"


def load_cases(category: str | None = None, limit: int | None = None) -> list[dict]:
    cases = []
    with QA_SET.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            if category and case.get("category") != category:
                continue
            cases.append(case)
    return cases[:limit] if limit else cases


def evaluate(agent: LoLAgent, cases: list[dict]) -> dict:
    results = []
    for case in cases:
        answer = agent.ask(case["question"])
        haystack = answer.text + "\n" + "\n".join(ev.content for ev in answer.evidences)

        keywords = case.get("keywords") or []
        hit_keywords = [kw for kw in keywords if kw in haystack]

        expected_route = case.get("expected_route") or ""
        route_ok = (not expected_route) or answer.route.value == expected_route

        results.append(
            {
                "id": case.get("id"),
                "category": case.get("category"),
                "question": case["question"],
                "route": answer.route.value,
                "expected_route": expected_route,
                "route_ok": route_ok,
                "keywords": keywords,
                "hit_keywords": hit_keywords,
                "keyword_ok": (not keywords) or bool(hit_keywords),
                "used_web": answer.used_web,
                "requires_web": bool(case.get("requires_web")),
                "evidence_count": len(answer.evidences),
                "elapsed_ms": answer.elapsed_ms,
                "degraded": answer.degraded,
                "answer_preview": answer.text[:160].replace("\n", " "),
                "notes": answer.notes,
            }
        )
    return _summarize(results)


def _summarize(results: list[dict]) -> dict:
    total = len(results)
    route_ok = sum(1 for r in results if r["route_ok"])
    keyword_cases = [r for r in results if r["keywords"]]
    keyword_ok = sum(1 for r in keyword_cases if r["keyword_ok"])
    web_expected = [r for r in results if r["requires_web"]]
    web_triggered = sum(1 for r in web_expected if r["used_web"])
    avg_ms = sum(r["elapsed_ms"] for r in results) / total if total else 0

    by_category: dict[str, dict] = {}
    for r in results:
        bucket = by_category.setdefault(
            r["category"], {"total": 0, "route_ok": 0, "keyword_ok": 0, "keyword_total": 0}
        )
        bucket["total"] += 1
        bucket["route_ok"] += int(r["route_ok"])
        if r["keywords"]:
            bucket["keyword_total"] += 1
            bucket["keyword_ok"] += int(r["keyword_ok"])

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "route_accuracy": round(route_ok / total, 3) if total else 0,
        "keyword_hit_rate": round(keyword_ok / len(keyword_cases), 3) if keyword_cases else None,
        "web_trigger_rate": round(web_triggered / len(web_expected), 3) if web_expected else None,
        "avg_elapsed_ms": round(avg_ms),
        "by_category": by_category,
        "results": results,
    }


def print_summary(report: dict) -> None:
    print("\n" + "=" * 64)
    print("评测汇总")
    print("=" * 64)
    print(f"题量：{report['total']}")
    print(f"路由准确率：{report['route_accuracy']:.1%}")
    if report["keyword_hit_rate"] is not None:
        print(f"关键词命中率：{report['keyword_hit_rate']:.1%}（答案或证据中出现任一预期关键词）")
    if report["web_trigger_rate"] is not None:
        print(f"联网兜底触发率：{report['web_trigger_rate']:.1%}")
    print(f"平均耗时：{report['avg_elapsed_ms']} ms")
    print("-" * 64)
    print("分类明细：")
    for category, bucket in report["by_category"].items():
        acc = bucket["route_ok"] / bucket["total"] if bucket["total"] else 0
        kw = (
            f"{bucket['keyword_ok']}/{bucket['keyword_total']}"
            if bucket["keyword_total"]
            else "-"
        )
        print(f"  {category:<10} 共 {bucket['total']:<3} 路由正确 {bucket['route_ok']:<3} ({acc:.0%})  关键词命中 {kw}")

    print("-" * 64)
    print("未通过样例（badcase，用于归因分析）：")
    bad = [r for r in report["results"] if not (r["route_ok"] and r["keyword_ok"])]
    if not bad:
        print("  无")
    for r in bad:
        reasons = []
        if not r["route_ok"]:
            reasons.append(f"路由应为 {r['expected_route'] or '任意'}，实际 {r['route']}")
        if not r["keyword_ok"]:
            reasons.append(f"未命中关键词 {r['keywords']}（实际命中 {r['hit_keywords']}）")
        print(f"  #{r['id']} [{r['category']}] {r['question']}")
        print(f"      -> {'；'.join(reasons)}")
        print(f"      -> 证据数 {r['evidence_count']}，联网 {r['used_web']}，耗时 {r['elapsed_ms']}ms")


def main() -> None:
    parser = argparse.ArgumentParser(description="运行英雄联盟问答智能体评测")
    parser.add_argument("--category", default=None, help="只跑某个分类：numeric/strategy/matchup/latest/unknown")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 条")
    parser.add_argument("--save", action="store_true", help="保存完整报告到 eval/reports/")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s")

    agent = LoLAgent()
    health = agent.health()
    print("知识库状态：" + json.dumps(health, ensure_ascii=False))
    if not health["structured_ready"]:
        print("\n[错误] 结构化库未就绪，请先运行：python ingest_data.py")
        raise SystemExit(1)

    cases = load_cases(args.category, args.limit)
    print(f"开始评测，共 {len(cases)} 题...\n")
    started = time.time()
    report = evaluate(agent, cases)
    report["total_elapsed_ms"] = int((time.time() - started) * 1000)
    print_summary(report)

    if args.save:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        path = REPORT_DIR / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n完整报告已保存：{path}")


if __name__ == "__main__":
    main()
