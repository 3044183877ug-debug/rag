"""
debug_chunks.py
--------------
极简检索排查脚本 — 输入一个问题，直观查看召回的 top-K Chunk 完整内容、distance、
以及当前阈值下的置信度路由判定。

用法:
    python debug_chunks.py "合伙企业是否需要缴纳企业所得税？"
    python debug_chunks.py -k 6 "研发费用加计扣除比例"   # 指定召回数量
    python debug_chunks.py                               # 不传参则使用默认示例问题

依赖:
    - rag_agent.py 中的 rag_chain 对象（自动加载向量库 & embedding 模型）
"""

import sys

# ── 解析命令行参数 ──────────────────────────────────────────────
# 支持 -k N 指定召回数（默认 6，与生产环境 stream() 保持一致）
args = sys.argv[1:]
K = 6  # 默认与 rag_agent.stream() 一致
question_parts = []

i = 0
while i < len(args):
    if args[i] == "-k" and i + 1 < len(args):
        K = int(args[i + 1])
        i += 2
    else:
        question_parts.append(args[i])
        i += 1

if question_parts:
    question = " ".join(question_parts)
else:
    question = "我开了一家汽车修理厂，执照上写的是个体工商户，去年的净利润大概是150万左右。我看政策说小型微利企业可以按20%交企业所得税，我们修理厂能按这个交吗？" 

# ── 导入 rag_agent（会自动触发模型/向量库加载）──────────────────
print(f"正在初始化 RAG 链路（加载 embedding 模型 + 向量库）...")
print(f"问题: \"{question}\"")
print(f"召回数: k={K}")
print()

from rag_agent import rag_chain, classify_confidence, ConfidenceLevel
from rag_agent import HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD

# ══════════════════════════════════════════════════════════════════
#  检索
# ══════════════════════════════════════════════════════════════════

print("=" * 80)
print(f"  🔍 检索问题: {question}")
print("=" * 80)
print()

# 展示查询扩展效果
expanded = rag_chain._expand_query(question)
if expanded != question:
    print(f"  🔄 查询扩展: \"{question}\"\n            → \"{expanded}\"\n")

docs, distances = rag_chain._single_search(question, k=K)

if not docs:
    print("  ⚠️ 检索返回 0 条结果！请检查向量库是否为空。")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════════
#  置信度路由判定
# ══════════════════════════════════════════════════════════════════

confidence = classify_confidence(distances)
best = min(distances)

conf_label = {
    ConfidenceLevel.HIGH:  f"✅ HIGH  (≤{HIGH_CONFIDENCE_THRESHOLD}) — 走完整 RAG + 严谨回答",
    ConfidenceLevel.LOW:   f"🟡 LOW   ({HIGH_CONFIDENCE_THRESHOLD}<d≤{LOW_CONFIDENCE_THRESHOLD}) — RAG + 风险警告",
    ConfidenceLevel.NOHIT: f"🚫 NOHIT (>{LOW_CONFIDENCE_THRESHOLD}) — 阻断 LLM，固定话术",
}[confidence]

print(f"  置信度路由: {conf_label}")
print()

# ══════════════════════════════════════════════════════════════════
#  逐 Chunk 展示
# ══════════════════════════════════════════════════════════════════

print(f"召回 Top-{K} Chunk:\n")

for i, (doc, dist) in enumerate(zip(docs, distances), 1):
    meta = doc.metadata
    source_name = meta.get("source_name", meta.get("source", "?"))
    source_number = meta.get("source_number", "")
    article = meta.get("article", "")
    chapter = meta.get("chapter", "")
    doc_type = meta.get("doc_type", "")
    date_effective = meta.get("date_effective", meta.get("date", ""))
    topic = meta.get("topic", "")

    # ── 来源标签 ──────────────────────────────────────────────
    label = f"《{source_name}》"
    if source_number:
        label += f" ({source_number})"
    if doc_type:
        label += f" [{doc_type}]"
    if chapter:
        label += f" {chapter}"
    if article:
        label += f" {article}"
    if date_effective:
        label += f"  生效: {date_effective}"
    if topic:
        label += f"\n  主题: {topic}"

    # ── 命中等级标记 ──────────────────────────────────────────
    if dist <= HIGH_CONFIDENCE_THRESHOLD:
        flag = "✅ HIGH"
    elif dist <= LOW_CONFIDENCE_THRESHOLD:
        flag = "🟡 LOW"
    else:
        flag = "❌ NOHIT"

    print(f"{'─' * 80}")
    print(f"  📄 Chunk #{i}  |  distance = {dist:.4f}  |  {flag}")
    print(f"  📌 来源: {label}")
    print(f"  📏 长度: {len(doc.page_content)} 字符")
    print(f"{'─' * 80}")
    print(doc.page_content)
    print()

# ══════════════════════════════════════════════════════════════════
#  汇总
# ══════════════════════════════════════════════════════════════════

high_n = sum(1 for d in distances if d <= HIGH_CONFIDENCE_THRESHOLD)
low_n  = sum(1 for d in distances if HIGH_CONFIDENCE_THRESHOLD < d <= LOW_CONFIDENCE_THRESHOLD)
nohit_n = sum(1 for d in distances if d > LOW_CONFIDENCE_THRESHOLD)

print("=" * 80)
print(f"  Distance 汇总: {[f'{d:.4f}' for d in distances]}")
print(f"  Best: {best:.4f}  |  Worst: {max(distances):.4f}")
print(f"  HIGH(≤{HIGH_CONFIDENCE_THRESHOLD}): {high_n}  "
      f"LOW({HIGH_CONFIDENCE_THRESHOLD}<d≤{LOW_CONFIDENCE_THRESHOLD}): {low_n}  "
      f"NOHIT(>{LOW_CONFIDENCE_THRESHOLD}): {nohit_n}")
print(f"  路由判定: {conf_label}")
print("=" * 80)
