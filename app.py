"""Gradio 对话界面。

界面刻意把"路由命中路径 + 来源引用"显示出来：
一是让答案可核验，二是让使用者直观看到"这个问题到底走的哪类知识库"，
这正是理解混合知识库最好的方式。

启动：python app.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

from src.lol_agent.agent import LoLAgent  # noqa: E402

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

agent = LoLAgent()

ROUTE_LABEL = {
    "structured": "结构化库（SQLite·官方数值）",
    "vector": "向量库（Chroma·攻略经验）",
    "web": "联网搜索（最新版本）",
    "hybrid": "混合检索",
}


def health_markdown() -> str:
    info = agent.health()
    rows = [
        ("游戏版本", info.get("game_version") or "未采集"),
        ("结构化知识库", "就绪" if info.get("structured_ready") else "未采集"),
        ("向量知识库", f"就绪（{info.get('vector_chunks')} 个切片）" if info.get("vector_ready") else "未构建"),
        ("嵌入模型", info.get("embed_model")),
        ("生成模型", "已配置" if info.get("llm_ready") else "未配置（仅检索模式）"),
        ("联网兜底", "已配置" if info.get("search_ready") else "未配置"),
    ]
    body = "\n".join(f"| {k} | {v} |" for k, v in rows)
    warning = ""
    if info.get("embed_model_mismatch"):
        warning = f"\n\n> ⚠️ {info['embed_model_mismatch']}"
    return f"### 知识库状态\n\n| 项目 | 状态 |\n| --- | --- |\n{body}{warning}"


def render_panel(answer) -> str:
    lines = [
        f"**命中路径**：{ROUTE_LABEL.get(answer.route.value, answer.route.value)}",
        f"**耗时**：{answer.elapsed_ms} ms　|　**联网**：{'是' if answer.used_web else '否'}",
    ]
    if answer.decision:
        entities = []
        if answer.decision.champions:
            entities.append("英雄：" + "、".join(answer.decision.champions))
        if answer.decision.items:
            entities.append("装备：" + "、".join(answer.decision.items))
        if answer.decision.slots:
            entities.append("技能位：" + "、".join(answer.decision.slots))
        if entities:
            lines.append("**识别实体**：" + "　".join(entities))
        if answer.decision.reason:
            lines.append(f"**判定理由**：{answer.decision.reason}")
    if answer.notes:
        lines.append("\n" + "\n".join(f"> 提示：{n}" for n in answer.notes))
    lines.append("\n**来源引用**\n")
    lines.append(answer.sources_text())
    return "\n\n".join(lines)


def respond(message: str, history: list):
    if not message.strip():
        return history, "", "请输入问题。"
    answer = agent.ask(message)
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer.text},
    ]
    return history, "", render_panel(answer)


with gr.Blocks(title="英雄联盟知识问答智能体", fill_height=True) as demo:
    gr.Markdown(
        "# 英雄联盟知识问答智能体\n"
        "混合知识库示例：**官方数值走 SQLite，攻略经验走 Chroma 向量库，最新改动走联网兜底**。"
        "每条回答都会标注来源，便于自行核验。"
    )
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=520, label="对话")
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="例如：亚索 W 技能冷却是多少？／上单诺手对线怎么打？",
                    scale=5,
                    show_label=False,
                )
                send = gr.Button("发送", variant="primary", scale=1)
            gr.Examples(
                examples=[
                    "亚索 W 技能的冷却时间是多少？",
                    "无尽之刃多少钱？",
                    "上单诺手对线应该怎么打？",
                    "什么类型的英雄克制亚索？",
                    "当前版本亚索被加强了吗？",
                ],
                inputs=txt,
            )
        with gr.Column(scale=2):
            status = gr.Markdown("等待提问…", label="检索诊断")
            gr.Markdown(health_markdown())

    txt.submit(respond, [txt, chatbot], [chatbot, txt, status])
    send.click(respond, [txt, chatbot], [chatbot, txt, status])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
