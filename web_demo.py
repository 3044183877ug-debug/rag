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
import json
from datetime import datetime

from langchain_core.documents import Document

# ── 反馈日志路径 ──────────────────────────────────────────────────
FEEDBACK_LOG_FILE = "data/feedback_logs.jsonl"
os.makedirs(os.path.dirname(os.path.abspath(FEEDBACK_LOG_FILE)), exist_ok=True)

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

def log_user_feedback(prompt: str, target_query: str, dst_state: dict, answer: str, attitude: int) -> None:
    """将用户点赞/点踩的对话上下文写入本地 JSONL 日志，用于后续分析。"""
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "original_prompt": prompt,
        "resolved_query": target_query,
        "dst_state": dst_state,
        "bot_answer": answer,
        "user_attitude": "thumbs_up" if attitude == 1 else "thumbs_down",
    }
    with open(FEEDBACK_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

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

if "dst_state" not in st.session_state:
    st.session_state.dst_state = {
        "tax_type": None,
        "entity": None,
        "action": None,
        "resolved_query": None,
    }

if "temp_file_content" not in st.session_state:
    st.session_state.temp_file_content = None

if "logged_feedback" not in st.session_state:
    st.session_state.logged_feedback = set()

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
        st.session_state.dst_state = {
            "tax_type": None,
            "entity": None,
            "action": None,
            "resolved_query": None,
        }
        st.session_state.temp_file_content = None
        st.rerun()

    st.divider()

    # ── 多轮对话开关（自动开启）──────────────────────────────
    enable_multi_turn = True

    # ── 用户提示 ────────────────────────────────────────────
    if enable_multi_turn:
        st.info(
            "💡 **记忆上限提示**：为保证法律检索的绝对精准，避免话题交叉干扰，"
            "系统最多仅结合您最近 **3 轮**的对话记录进行上下文推演。"
            "若需开启全新税种咨询，建议点击上方「🗑️ 清空对话」。"
        )

    # ── DST 状态可视化（仅多轮模式下显示）─────────────────────
    if enable_multi_turn:
        st.divider()
        st.subheader("🧠 对话状态记忆")
        st.caption("后台实时追踪的对话上下文")
        st.json(st.session_state.dst_state)

    st.divider()

    # ── 临时参考文件上传 ────────────────────────────────────
    uploaded_file = st.file_uploader(
        "📄 上传临时参考文件 (仅限本次对话)",
        type=["txt"],
        help="上传的 .txt 文件内容不会进入向量检索，仅在 LLM 生成回答时作为补充上下文注入。"
             "内容自动截断至前 3000 字符，避免影响响应速度。",
    )
    if uploaded_file is not None:
        try:
            raw_content = uploaded_file.getvalue().decode("utf-8")
            st.session_state.temp_file_content = raw_content[:3000]
            st.caption(
                f"✅ 已加载临时参考文件：`{uploaded_file.name}` "
                f"({len(raw_content[:3000])} / {len(raw_content)} 字符)"
            )
        except Exception as e:
            st.warning(f"⚠️ 文件读取失败: {e}")
            st.session_state.temp_file_content = None
    else:
        st.session_state.temp_file_content = None

# ════════════════════════════════════════════════════════════════
#  主区域
# ════════════════════════════════════════════════════════════════

st.title("📋 税务 RAG 智能问答")
st.caption("基于中国税法知识库 · 三级置信度路由 · 流式逐字生成")

# ── 渲染历史消息 ──────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
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

        # 反馈按钮（每轮 assistant 消息下方）
        if msg["role"] == "assistant":
            fb_key = f"fb_{i}"
            fb_result = st.feedback("thumbs", key=fb_key)
            if fb_result is not None and fb_key not in st.session_state.logged_feedback:
                # 从当前消息中反查上下文信息
                log_user_feedback(
                    prompt=msg.get("original_prompt", ""),
                    target_query=msg.get("resolved_query", ""),
                    dst_state=msg.get("dst_state", {}),
                    answer=msg["content"],
                    attitude=fb_result,
                )
                st.session_state.logged_feedback.add(fb_key)
                st.toast("已记录您的反馈，我们将持续优化系统！")

# ── 输入框 ────────────────────────────────────────────────────
if prompt := st.chat_input("请输入您的税务问题，例如：合伙企业需要缴纳企业所得税吗？"):
    # ── 0. 提取纯文本对话历史（在添加本轮消息之前）─────────────
    chat_history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m.get("role") in ("user", "assistant")
    ]

    # 1. 添加用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ── 1.5 DST 多轮改写（仅在开关开启时触发）─────────────────
    search_query = prompt  # 默认使用原始问题（单轮行为不变）
    target_query = prompt  # 最终发给 LLM 的生成文本（单轮 = 原始输入）
    if enable_multi_turn:
        with st.spinner("🧠 正在分析对话上下文..."):
            st.session_state.dst_state = chain.update_dialogue_state(
                chat_history=chat_history,
                current_state=st.session_state.dst_state,
                current_query=prompt,
            )
        resolved = st.session_state.dst_state["resolved_query"]
        if resolved:
            search_query = resolved
            target_query = resolved  # 多轮：LLM 收到完整改写后的查询

    # 2. 检索 + 路由判定（先于流式回答，用于展示来源）
    with st.spinner("🔍 正在检索相关政策..."):
        docs, distances = chain._hybrid_search(search_query, k=6)
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
            # ── 伪装 Document 注入：临时参考文件（不进向量检索，仅注入 LLM 生成）──
            if st.session_state.temp_file_content:
                temp_doc = Document(
                    page_content=st.session_state.temp_file_content,
                    metadata={
                        "source_name": "用户上传临时参考资料",
                        "doc_type": "临时注入",
                    },
                )
                injected_docs = [temp_doc] + docs
                injected_distances = [0.0] + distances
            else:
                injected_docs = docs
                injected_distances = distances

            full_answer = st.write_stream(
                chain.stream(
                    target_query,
                    pre_docs=injected_docs,
                    pre_distances=injected_distances,
                    bypass_nohit=bool(st.session_state.temp_file_content),
                )
            )
        except Exception as e:
            full_answer = f"⚠️ 回答生成失败: {e}"

        # ── DST 改写提示（仅当多轮改写生效时展示）─────────────
        if enable_multi_turn and search_query != prompt:
            st.caption(f"🔍 系统自动补充上下文：{search_query}")

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

        # ── 反馈按钮 ──────────────────────────────────────────
        fb_key = f"fb_{len(st.session_state.messages)}"
        fb_result = st.feedback("thumbs", key=fb_key)
        if fb_result is not None and fb_key not in st.session_state.logged_feedback:
            log_user_feedback(
                prompt=prompt,
                target_query=target_query,
                dst_state=st.session_state.dst_state,
                answer=full_answer,
                attitude=fb_result,
            )
            st.session_state.logged_feedback.add(fb_key)
            st.toast("已记录您的反馈，我们将持续优化系统！")

    # 5. 存入历史
    st.session_state.messages.append({
        "role": "assistant",
        "content": full_answer,
        "confidence": confidence.value,
        "sources": sources_payload,
        "original_prompt": prompt,
        "resolved_query": target_query,
        "dst_state": st.session_state.dst_state.copy(),
    })
