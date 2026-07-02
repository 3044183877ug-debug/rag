"""
web_demo.py
----------
Streamlit 可视化网页 — 类似 ChatGPT 的税务 RAG 对话界面。

特性:
    - 流式逐字输出（st.write_stream + 原生 SSE 流）
    - 对话历史记录（st.session_state 持久化）
    - 检索来源透视（每个回答附带 Top-6 来源的 distance & 元数据）
    - 置信度路由可视化（HIGH / LOW / NOHIT 彩色标签）

用法:
    streamlit run web_demo.py
"""

from __future__ import annotations

import sys
import os
import time

# ── 必须在 import rag_agent 之前设置，抑制模块顶层 print ─────────
os.environ["RAG_SILENT_IMPORT"] = "1"

import streamlit as st

# ── 页面配置（必须是第一个 st 命令）─────────────────────────────
st.set_page_config(
    page_title="税务 RAG 智能问答",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ════════════════════════════════════════════════════════════════
#  CSS 微调
# ════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    /* 来源卡片紧凑排版 */
    .source-item {
        font-size: 0.82rem;
        padding: 0.3rem 0;
        border-bottom: 1px solid #eee;
    }
    .source-item:last-child { border-bottom: none; }
    .source-distance { color: #888; font-family: monospace; }
    .confidence-badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 4px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.5rem;
    }
    .confidence-high { background: #d4edda; color: #155724; }
    .confidence-low  { background: #fff3cd; color: #856404; }
    .confidence-nohit { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════
#  初始化 RAG 链路（带缓存，仅首次加载时耗时）
# ════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="正在加载 Embedding 模型 & 向量数据库...")
def _init_rag():
    """缓存 RAG 链路的初始化，跨 Streamlit rerun 复用。"""
    import rag_agent as _rag
    # 关闭检索明细打印（避免污染 st 日志）
    _rag.RETRIEVAL_DEBUG = False
    return _rag

try:
    rag = _init_rag()
    chain = rag.rag_chain
except Exception as e:
    st.error(f"❌ RAG 系统初始化失败: {e}")
    st.stop()

# ════════════════════════════════════════════════════════════════
#  会话状态初始化
# ════════════════════════════════════════════════════════════════

if "messages" not in st.session_state:
    st.session_state.messages = []  # 每条: {role, content, sources?, confidence?}

# ════════════════════════════════════════════════════════════════
#  侧边栏 — 系统信息
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.header("📋 系统信息")
    st.metric("向量库规模", f"{rag.vectordb._collection.count()} 条")
    st.caption(f"Embedding: BAAI/bge-small-zh-v1.5")
    st.caption(f"LLM: DeepSeek-V4-Pro")
    st.divider()
    st.caption(f"HIGH ≤ {rag.HIGH_CONFIDENCE_THRESHOLD} | LOW ≤ {rag.LOW_CONFIDENCE_THRESHOLD} | NOHIT > {rag.LOW_CONFIDENCE_THRESHOLD}")
    st.divider()
    if st.button("🗑️ 清空对话", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ════════════════════════════════════════════════════════════════
#  主区域
# ════════════════════════════════════════════════════════════════

st.title("📋 税务 RAG 智能问答")
st.caption("基于中国税法知识库 · 三级置信度路由 · 流式逐字生成")

# ── 渲染历史消息 ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # 置信度标签（仅 assistant 消息）
        if msg["role"] == "assistant" and msg.get("confidence"):
            conf = msg["confidence"]
            labels = {
                "high": ("✅ 高置信度", "confidence-high"),
                "low": ("🟡 低置信度", "confidence-low"),
                "nohit": ("🚫 未命中", "confidence-nohit"),
            }
            label_text, label_css = labels.get(conf, ("", ""))
            st.markdown(
                f'<span class="confidence-badge {label_css}">{label_text}</span>',
                unsafe_allow_html=True,
            )

        # 正文
        st.markdown(msg["content"])

        # 来源折叠面板（仅 assistant 消息）
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 查看检索来源（Top-6）", expanded=False):
                for src in msg["sources"]:
                    dist = src.get("distance", 0)
                    # 距离标记
                    if dist <= rag.HIGH_CONFIDENCE_THRESHOLD:
                        flag = "🟢"
                    elif dist <= rag.LOW_CONFIDENCE_THRESHOLD:
                        flag = "🟡"
                    else:
                        flag = "🔴"

                    st.markdown(
                        f'<div class="source-item">'
                        f'<strong>#{src["rank"]}</strong> {flag} '
                        f'<span class="source-distance">distance={dist:.4f}</span><br>'
                        f'《{src["source_name"]}》'
                        + (f' ({src["source_number"]})' if src["source_number"] else '')
                        + (f' [{src["doc_type"]}]' if src["doc_type"] else '')
                        + (f' {src["chapter"]}' if src["chapter"] else '')
                        + (f' {src["article"]}' if src["article"] else '')
                        + '<br><span style="color:#666;font-size:0.78rem;">'
                        + src["content_preview"].replace("\n", " ")[:120]
                        + '...</span></div>',
                        unsafe_allow_html=True,
                    )

# ── 输入框 ────────────────────────────────────────────────────
if prompt := st.chat_input("请输入您的税务问题，例如：合伙企业需要缴纳企业所得税吗？"):
    # 1. 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 检索 + 路由判定（先于流式回答，用于展示来源）
    with st.spinner("🔍 正在检索相关政策..."):
        docs, distances = chain._single_search(prompt, k=6)
        confidence = rag.classify_confidence(distances)

    # 3. 构造来源数据
    sources_payload = []
    for i, (doc, dist) in enumerate(zip(docs, distances), 1):
        meta = doc.metadata
        sources_payload.append({
            "rank": i,
            "distance": round(dist, 4),
            "source_name": meta.get("source_name", meta.get("source", "")),
            "source_number": meta.get("source_number", ""),
            "doc_type": meta.get("doc_type", ""),
            "chapter": meta.get("chapter", ""),
            "article": meta.get("article", ""),
            "date_effective": meta.get("date_effective", meta.get("date", "")),
            "topic": meta.get("topic", ""),
            "content_preview": doc.page_content,
        })

    # 4. 流式回答
    with st.chat_message("assistant"):
        # 置信度标签
        labels = {
            "high": ("✅ 高置信度 — 完整 RAG 回答", "confidence-high"),
            "low": ("🟡 低置信度 — 回答附带风险警告", "confidence-low"),
            "nohit": ("🚫 未命中 — 知识库未覆盖该问题", "confidence-nohit"),
        }
        label_text, label_css = labels.get(confidence.value, ("", ""))
        st.markdown(
            f'<span class="confidence-badge {label_css}">{label_text}</span>',
            unsafe_allow_html=True,
        )

        # 流式写回答
        answer_container = st.empty()
        try:
            full_answer = st.write_stream(
                chain.stream(prompt, pre_docs=docs, pre_distances=distances)
            )
        except Exception as e:
            full_answer = f"⚠️ 回答生成失败: {e}"

        # 来源折叠面板（事后展示）
        with st.expander("📚 查看检索来源（Top-6）", expanded=False):
            for src in sources_payload:
                dist = src["distance"]
                flag = "🟢" if dist <= rag.HIGH_CONFIDENCE_THRESHOLD else (
                    "🟡" if dist <= rag.LOW_CONFIDENCE_THRESHOLD else "🔴"
                )
                st.markdown(
                    f'<div class="source-item">'
                    f'<strong>#{src["rank"]}</strong> {flag} '
                    f'<span class="source-distance">distance={dist:.4f}</span><br>'
                    f'《{src["source_name"]}》'
                    + (f' ({src["source_number"]})' if src["source_number"] else '')
                    + (f' [{src["doc_type"]}]' if src["doc_type"] else '')
                    + (f' {src["chapter"]}' if src["chapter"] else '')
                    + (f' {src["article"]}' if src["article"] else '')
                    + (f' | 生效: {src["date_effective"]}' if src["date_effective"] else '')
                    + '<br><span style="color:#666;font-size:0.78rem;">'
                    + src["content_preview"].replace("\n", " ")[:150]
                    + '...</span></div>',
                    unsafe_allow_html=True,
                )

    # 5. 存入历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "confidence": confidence.value,
        "sources": sources_payload,
    })
