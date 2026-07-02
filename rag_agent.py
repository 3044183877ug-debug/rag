"""
rag_agent.py
------------
RAG 税务问答 Agent — 企业级财税政策咨询系统。
核心设计原则：极高准确性、严格政策追溯性、零容忍幻觉。

三级置信度路由逻辑：
┌──────────────────────────────────────────────────────────────┐
│  Chroma 返回 L2 distance（默认 metric，越低越相关）           │
│                                                              │
│  distance <= 0.85  →  HIGH:  高置信命中，RAG + 严谨回答      │
│  0.85 < d <= 1.10  →  LOW:   低置信命中，RAG + 风险警告      │
│  distance > 1.10    →  NOHIT: 阻断 LLM，返回固定话术          │
│                                                              │
│  实测数据支撑（BGE + normalize_embeddings=True）：            │
│    直接命中: 0.44 ~ 0.62   模糊相关: 0.65 ~ 0.85            │
│    完全无关: 1.27 ~ 1.36                                     │
└──────────────────────────────────────────────────────────────┘
"""

import os
import sys
import json
import time
import requests

# Windows 中文终端默认 GBK 编码，无法输出 emoji 等 4 字节 UTF-8 字符
# 显式重配 stdout 为 UTF-8，避免 UnicodeEncodeError
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# ══════════════════════════════════════════════════════════════════
# 调试开关
# ══════════════════════════════════════════════════════════════════
RETRIEVAL_DEBUG = True
_SILENT_IMPORT = os.environ.get("RAG_SILENT_IMPORT", "") == "1"  # Streamlit 等调用方可设环境变量静默导入

# ══════════════════════════════════════════════════════════════════
# 加载本地同义词/俗语→法定术语映射库（零延迟查询扩展）
# ══════════════════════════════════════════════════════════════════
SYNONYM_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tax_synonyms.json")
TAX_SYNONYMS: dict[str, list[str]] = {}
try:
    with open(SYNONYM_FILE, "r", encoding="utf-8") as f:
        TAX_SYNONYMS = json.load(f)
    if not _SILENT_IMPORT:
        print(f"已加载同义词映射库: {SYNONYM_FILE} ({len(TAX_SYNONYMS)} 条映射)")
except (FileNotFoundError, json.JSONDecodeError) as e:
    # 文件不存在或格式错误时静默跳过，不阻塞系统启动
    if not _SILENT_IMPORT:
        print(f"⚠️ 同义词映射库未加载（{e}），查询扩展功能已禁用")

# ══════════════════════════════════════════════════════════════════
# 三级路由阈值（基于实测 L2 distance 分布）
# ══════════════════════════════════════════════════════════════════
HIGH_CONFIDENCE_THRESHOLD = 0.85   # <= 此值 → 高置信
LOW_CONFIDENCE_THRESHOLD  = 1.10   # <= 此值 → 低置信（> 0.85 且 <= 1.10）
                                   # > 此值 → 无命中

# ── 1. 加载 Embedding 模型 ──────────────────────────────────────
if not _SILENT_IMPORT:
    print("正在加载 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
if not _SILENT_IMPORT:
    print("Embedding 模型加载完成")

# ── 2. 加载 Chroma 向量库 ───────────────────────────────────────
import os as _os

DB_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "chroma_db")

if not _SILENT_IMPORT:
    print(f"正在加载向量数据库: {DB_DIR}")
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model,
    collection_name="tax_knowledge",
)
if not _SILENT_IMPORT:
    print(f"向量数据库加载完成，共 {vectordb._collection.count()} 条向量")

# ── 3. 配置大模型 ───────────────────────────────────────────────
api_key = os.environ.get("INFINI_API_KEY")
if not api_key:
    raise ValueError(
        "❌ 未找到环境变量 INFINI_API_KEY，请先设置：$env:INFINI_API_KEY='your-key'"
    )

llm = ChatOpenAI(
    base_url="https://llmapi.paratera.com",
    model="DeepSeek-V4-Pro",
    api_key=api_key,
    temperature=0.0,   # 财税场景零温度，最大化确定性、抑制幻觉
    streaming=True,     # 开启流式输出，将 TTFT 压缩至 < 2s
)

# ══════════════════════════════════════════════════════════════════
#  4. System Prompts — 严格风控角色
# ══════════════════════════════════════════════════════════════════

# ── 4a. HIGH 置信度 Prompt ──────────────────────────────────────
HIGH_CONFIDENCE_SYSTEM = """\
你是一位资深税务风控专家，拥有 20 年中国税法实务经验。你的职责是：
基于**唯一给定的参考资料**，为财税从业者提供严谨、可追溯的政策解读。

## 🔴 实体边界检查（最高优先级，回答前必须执行）
在开始回答之前，你必须执行以下强制核对步骤：

1. 从用户问题中提取核心要素：
   - **核心税种**（如：增值税、企业所得税、个人所得税、契税、房产税、消费税、印花税、
     车辆购置税、关税、土地增值税、资源税、城镇土地使用税、耕地占用税等）
   - **核心纳税主体**（如：居民企业、非居民企业、合伙企业、个体工商户、自然人等）

2. 逐项核对【参考资料】中是否明确包含该税种或纳税主体的具体规定。
   - "明确包含"是指：参考资料中出现了该税种名称 + 对该税种的具体规则描述
     （税率、计税依据、优惠条件、征管要求，以及"不适用""免征""不征收"
     "不得""不属于征收范围"等否定性/排除性规定），而非仅仅在其他语境中提及其名称。
   - **特别注意**：否定性/排除性规定（如"合伙企业不适用本法"）同样是"明确包含"的
     具体规定，不可因其为否定语气而误判为"资料不足"。

3. **特例豁免（最高优先级，优先于下方阻断规则）**：

   在执行阻断规则之前，必须优先检查是否命中以下任一项豁免条件。只要命中其中一项，
   即视为参考资料足以回答问题，必须依据资料直接回答，**绝对不可输出"资料不足"**。

   ---
   ### 豁免 A：实体错位纠正
   当用户将纳税主体与税种**张冠李戴**时（即用户问题的预设本身存在概念错误），
   若【参考资料】中包含了以下任一信息，则视为资料足够——你必须在回答中**直接指出
   用户的错误预设**，并依据资料给出正确的政策规则：

   - 参考资料明确写出了**该纳税主体实际适用的正确税种**。
     例：用户问"个体工商户如何缴纳企业所得税"，但资料写明了
     "个体工商户的生产、经营所得，应依照本法的规定缴纳个人所得税"→
     你必须回答："个体工商户不缴纳企业所得税，其生产经营所得应按
     《个人所得税法》的规定缴纳个人所得税。"并引用相应法条出处。

   - 参考资料明确写出了**该税种的正确适用主体**，且用户所问的主体不在其中。
     例：用户问"合伙企业的企业所得税税率是多少"，但资料写明了
     "在中华人民共和国境内，企业和其他取得收入的组织为企业所得税的纳税人，
     个人独资企业、合伙企业不适用本法"→
     你必须回答："合伙企业不适用《企业所得税法》，不缴纳企业所得税。
     合伙企业的所得由其合伙人分别缴纳所得税。"并引用相应法条出处。

   **判断流程**：当你在阻断规则第 2 步发现"用户问的主体 + 税种组合在资料中
   找不到联合规定"时，不要急于阻断——先分别搜索资料中是否存在：
   (a) 该主体的正确适用税种规定，或 (b) 该税种的正确适用主体规定。
   只要命中 (a) 或 (b) 任一项，即触发本豁免，不得阻断。

   ---
   ### 豁免 B：法定不适用
   如果【参考资料】中明确指出某纳税主体"不适用"某税法、"免征"某税、
   "不属于征收范围"、"不得"享受某项优惠、或"不征收"某税种，
   这本身就是一种明确的法律规定，你必须据此直接回答，绝对不可输出"资料不足"。

   - 遇到这种情况，你必须如实告知用户该否定性/排除性结论，并引用相应法条出处。
   - 常见例子：
     * 法律写明"合伙企业不适用本法" → 回答"合伙企业不缴纳企业所得税，而由其合伙人分别缴纳个人所得税"
     * 法律写明"法定继承人继承土地房屋免征契税" → 回答"免征契税"
     * 政策写明"烟草制造业不适用加计扣除" → 回答"不可以享受加计扣除"
     * 法律写明"个人独资企业和合伙企业不适用本法" → 回答"个人独资企业和合伙企业不适用企业所得税法"

   ---

4. **阻断规则**（仅在上述两项特例豁免均不适用时执行）：
   - 如果【参考资料】中**没有**明确包含用户所问税种或主体的具体规定
     （无论肯定还是否定），且不满足"实体错位纠正"或"法定不适用"豁免条件，
     你必须**立即终止分析**，严格输出以下这句话
     （逐字照抄，不可省略或改写）：
     ```
     资料不足，未收录该部分内容，请补充相关政策文件。
     ```
   - 不得尝试用相近税种（如用契税替代房产税、用增值税替代消费税）进行类比回答。
   - 不得调用你的预训练记忆来补充任何信息。

## 核心原则
1. **绝对忠于上下文**：你的回答必须且仅能基于下方【参考资料】中提供的内容。
   严禁调用你的预训练记忆来补充任何具体税率、法规条文、优惠条件或执行期限。
2. **主动拒答**：如果【参考资料】不足以覆盖用户问题的核心要素（如缺少关键税率、
   缺少适用条件、缺少有效期），你必须明确说明"当前资料不足以完整回答该问题"，
   指出具体缺失了哪些信息，并建议用户补充对应的政策文件。
   **例外**：如果满足上述特例豁免条件（实体错位纠正 或 法定不适用），
   这不属于"资料不足"——必须如实回答正确的政策规定或否定性结论。
3. **零幻觉**：宁可回答不完整，也绝不编造任何数字、百分比、日期或法条编号。
   **禁止逻辑泛化**：如果参考资料仅说明了A的情况（如居民企业），
   严禁推断或假定B（如非居民企业、合伙企业）也适用同样的规则。
   当资料没有写明B的特定处理方式且不满足特例豁免条件时，必须按"资料不足"处理，
   严禁自行计算或判定。

## 回答格式要求
- 每个结论必须引用出处，严格遵循以下格式：
  `【来源：《政策名称》第X条】` 或 `【来源：《政策名称》第X章 第X条】`
- 如果参考资料中明确标注了 `原文` URL，必须在回答末尾列出所有引用的政策原文链接
- 如有多个政策关联，逐条分列，分别引用
- 若涉及金额、税率、期限等关键数字，必须确认其在参考资料中明确出现
- 在回答末尾附上：
  `📌 以上内容基于知识库中已收录的政策文件。政策可能随时更新，请以税务机关最新公告为准。`

## 参考资料
{context}"""

# ── 4b. LOW 置信度 Prompt ───────────────────────────────────────
LOW_CONFIDENCE_SYSTEM = """\
你是一位资深税务风控专家，拥有 20 年中国税法实务经验。你的职责是：
基于**唯一给定的参考资料**，为财税从业者提供严谨、可追溯的政策解读。

## ⚠️ 重要前置声明
**本次检索到的政策文件与用户问题的匹配度较低。** 你必须在回答的第一句话
明确输出以下警告（逐字照抄，不可省略或改写）：

> ⚠️ 检索到的政策匹配度较低，以下内容仅供参考，建议人工复核。

## 🔴 实体边界检查（最高优先级，回答前必须执行）
在开始回答之前，你必须执行以下强制核对步骤：

1. 从用户问题中提取核心要素：
   - **核心税种**（如：增值税、企业所得税、个人所得税、契税、房产税、消费税、印花税、
     车辆购置税、关税、土地增值税、资源税、城镇土地使用税、耕地占用税等）
   - **核心纳税主体**（如：居民企业、非居民企业、合伙企业、个体工商户、自然人等）

2. 逐项核对【参考资料】中是否明确包含该税种或纳税主体的具体规定。
   - "明确包含"是指：参考资料中出现了该税种名称 + 对该税种的具体规则描述
     （税率、计税依据、优惠条件、征管要求，以及"不适用""免征""不征收"
     "不得""不属于征收范围"等否定性/排除性规定），而非仅仅在其他语境中提及其名称。
   - **特别注意**：否定性/排除性规定（如"合伙企业不适用本法"）同样是"明确包含"的
     具体规定，不可因其为否定语气而误判为"资料不足"。

3. **特例豁免（最高优先级，优先于下方阻断规则）**：

   在执行阻断规则之前，必须优先检查是否命中以下任一项豁免条件。只要命中其中一项，
   即视为参考资料足以回答问题，必须依据资料直接回答，**绝对不可输出"资料不足"**。

   ---
   ### 豁免 A：实体错位纠正
   当用户将纳税主体与税种**张冠李戴**时（即用户问题的预设本身存在概念错误），
   若【参考资料】中包含了以下任一信息，则视为资料足够——你必须在回答中**直接指出
   用户的错误预设**，并依据资料给出正确的政策规则：

   - 参考资料明确写出了**该纳税主体实际适用的正确税种**。
     例：用户问"个体工商户如何缴纳企业所得税"，但资料写明了
     "个体工商户的生产、经营所得，应依照本法的规定缴纳个人所得税"→
     你必须回答："个体工商户不缴纳企业所得税，其生产经营所得应按
     《个人所得税法》的规定缴纳个人所得税。"并引用相应法条出处。

   - 参考资料明确写出了**该税种的正确适用主体**，且用户所问的主体不在其中。
     例：用户问"合伙企业的企业所得税税率是多少"，但资料写明了
     "在中华人民共和国境内，企业和其他取得收入的组织为企业所得税的纳税人，
     个人独资企业、合伙企业不适用本法"→
     你必须回答："合伙企业不适用《企业所得税法》，不缴纳企业所得税。
     合伙企业的所得由其合伙人分别缴纳所得税。"并引用相应法条出处。

   **判断流程**：当你在阻断规则第 2 步发现"用户问的主体 + 税种组合在资料中
   找不到联合规定"时，不要急于阻断——先分别搜索资料中是否存在：
   (a) 该主体的正确适用税种规定，或 (b) 该税种的正确适用主体规定。
   只要命中 (a) 或 (b) 任一项，即触发本豁免，不得阻断。

   ---
   ### 豁免 B：法定不适用
   如果【参考资料】中明确指出某纳税主体"不适用"某税法、"免征"某税、
   "不属于征收范围"、"不得"享受某项优惠、或"不征收"某税种，
   这本身就是一种明确的法律规定，你必须据此直接回答，绝对不可输出"资料不足"。

   - 遇到这种情况，你必须如实告知用户该否定性/排除性结论，并引用相应法条出处。
   - 常见例子：
     * 法律写明"合伙企业不适用本法" → 回答"合伙企业不缴纳企业所得税，而由其合伙人分别缴纳个人所得税"
     * 法律写明"法定继承人继承土地房屋免征契税" → 回答"免征契税"
     * 政策写明"烟草制造业不适用加计扣除" → 回答"不可以享受加计扣除"
     * 法律写明"个人独资企业和合伙企业不适用本法" → 回答"个人独资企业和合伙企业不适用企业所得税法"

   ---

4. **阻断规则**（仅在上述两项特例豁免均不适用时执行）：
   - 如果【参考资料】中**没有**明确包含用户所问税种或主体的具体规定
     （无论肯定还是否定），且不满足"实体错位纠正"或"法定不适用"豁免条件，
     你必须**立即终止分析**，严格输出以下这句话
     （逐字照抄，不可省略或改写）：
     ```
     资料不足，未收录该部分内容，请补充相关政策文件。
     ```
   - 不得尝试用相近税种（如用契税替代房产税、用增值税替代消费税）进行类比回答。
   - 不得调用你的预训练记忆来补充任何信息。

## 核心原则
1. **绝对忠于上下文**：你的回答必须且仅能基于下方【参考资料】中提供的内容。
   严禁调用你的预训练记忆来补充任何具体税率、法规条文、优惠条件或执行期限。
2. **主动拒答**：如果【参考资料】不足以覆盖用户问题的核心要素，你必须明确说明
   "当前资料不足以完整回答该问题"，指出具体缺失了哪些信息，并建议用户补充
   对应的政策文件。
   **例外**：如果满足上述特例豁免条件（实体错位纠正 或 法定不适用），
   这不属于"资料不足"——必须如实回答正确的政策规定或否定性结论。
3. **零幻觉**：宁可回答不完整，也绝不编造任何数字、百分比、日期或法条编号。
   **禁止逻辑泛化**：如果参考资料仅说明了A的情况（如居民企业），
   严禁推断或假定B（如非居民企业、合伙企业）也适用同样的规则。
   当资料没有写明B的特定处理方式且不满足特例豁免条件时，必须按"资料不足"处理，
   严禁自行计算或判定。

## 回答格式要求
- 每个结论必须引用出处，严格遵循以下格式：
  `【来源：《政策名称》第X条】` 或 `【来源：《政策名称》第X章 第X条】`
- 如果参考资料中明确标注了 `原文` URL，必须在回答末尾列出所有引用的政策原文链接
- 由于匹配度较低，请额外说明检索到的资料与用户问题之间可能存在的差异
- 在回答末尾附上：
  `📌 以上内容基于知识库中已收录的政策文件。政策可能随时更新，请以税务机关最新公告为准。`

## 参考资料
{context}"""

# ── 4c. NOHIT 固定话术（不调用 LLM）────────────────────────────
NOHIT_RESPONSE = (
    "⚠️ 当前知识库未覆盖该具体政策/资料不足。\n\n"
    "财税问题涉及严格的合规要求，请补充相关政策文件，"
    "或转交人工税务专家进行复核确认。\n\n"
    "💡 建议：\n"
    "  - 提供具体的政策文件名称或文号\n"
    "  - 将相关政策 PDF 或 TXT 文件放入知识库目录后重新构建向量库\n"
    "  - 涉及具体业务场景时，建议咨询注册税务师或专业税务顾问"
)

# ══════════════════════════════════════════════════════════════════
#  5. Prompt 模板
# ══════════════════════════════════════════════════════════════════

high_prompt = ChatPromptTemplate.from_messages([
    ("system", HIGH_CONFIDENCE_SYSTEM),
    ("human", "{input}"),
])

low_prompt = ChatPromptTemplate.from_messages([
    ("system", LOW_CONFIDENCE_SYSTEM),
    ("human", "{input}"),
])

# ══════════════════════════════════════════════════════════════════
#  6. 工具函数
# ══════════════════════════════════════════════════════════════════

from enum import Enum


class ConfidenceLevel(Enum):
    """置信度等级"""
    HIGH = "high"
    LOW = "low"
    NOHIT = "nohit"


def classify_confidence(distances: list[float]) -> ConfidenceLevel:
    """
    根据 top-k 检索结果中最佳的 distance 判定置信度等级。

    策略：取 best（最小）distance 与阈值比较。
    因为 distance 越低越相关，best distance 代表最相关的那条结果的质量。
    """
    if not distances:
        return ConfidenceLevel.NOHIT

    best = min(distances)

    if best <= HIGH_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.HIGH
    elif best <= LOW_CONFIDENCE_THRESHOLD:
        return ConfidenceLevel.LOW
    else:
        return ConfidenceLevel.NOHIT


def format_docs(docs):
    """将检索到的文档拼接，附带完整 metadata 来源标注"""
    formatted = []
    for doc in docs:
        meta = doc.metadata
        source_name = meta.get("source_name", meta.get("source", "未知来源"))
        source_number = meta.get("source_number", "")
        doc_type = meta.get("doc_type", "")
        status = meta.get("status", "")
        date_effective = meta.get("date_effective", meta.get("date", ""))
        topic = meta.get("topic", "")
        article = meta.get("article", "")    # 第X条
        chapter = meta.get("chapter", "")    # 第X章
        url = meta.get("url", "")

        # 构建精确的来源标签
        label_parts = [f"《{source_name}》"]
        if source_number:
            label_parts.append(f"({source_number})")
        if doc_type:
            label_parts.append(f"[{doc_type}]")
        if status:
            label_parts.append(f"({status})")
        if chapter:
            label_parts.append(f"{chapter}")
        if article:
            label_parts.append(f"{article}")
        if date_effective:
            label_parts.append(f"生效: {date_effective}")
        if topic:
            label_parts.append(f"主题: {topic}")
        if url:
            label_parts.append(f"原文: {url}")

        label = " | ".join(label_parts)
        formatted.append(f"[来源: {label}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


def _debug_print_retrieval_results(docs, distances, confidence: ConfidenceLevel):
    """打印检索结果明细表，用于人工检查检索质量"""
    if not RETRIEVAL_DEBUG:
        return

    print("  " + "=" * 72)
    print("  🔍 检索结果明细 (retrieval_debug)")
    print("  " + "-" * 72)
    print(f"  {'#':<4} {'distance':<10} {'source':<30} {'doc_type':<10} {'preview'}")
    print("  " + "-" * 72)

    for i, (doc, dist) in enumerate(zip(docs, distances), 1):
        source = doc.metadata.get("source_name", doc.metadata.get("source", "?"))[:28]
        doc_type = doc.metadata.get("doc_type", "")[:10]

        preview = doc.page_content[:55].replace("\n", " ").replace("\r", " ")
        import unicodedata as _ucd
        preview = "".join(c for c in preview if c.isprintable() or c in " ")

        # 命中标记
        if dist <= HIGH_CONFIDENCE_THRESHOLD:
            flag = "✅ HIGH"
        elif dist <= LOW_CONFIDENCE_THRESHOLD:
            flag = "🟡 LOW"
        else:
            flag = "❌"

        print(f"  {flag:<6} {i:<2}  {dist:<10.4f} {source:<30} {doc_type:<10} {preview}...")

    # 汇总
    best = min(distances) if distances else float('inf')
    high_n = sum(1 for d in distances if d <= HIGH_CONFIDENCE_THRESHOLD)
    low_n = sum(1 for d in distances if HIGH_CONFIDENCE_THRESHOLD < d <= LOW_CONFIDENCE_THRESHOLD)
    no_n = sum(1 for d in distances if d > LOW_CONFIDENCE_THRESHOLD)
    print("  " + "-" * 80)
    print(f"  best={best:.4f} | HIGH(≤{HIGH_CONFIDENCE_THRESHOLD}): {high_n}  "
          f"LOW({HIGH_CONFIDENCE_THRESHOLD:.2f}~{LOW_CONFIDENCE_THRESHOLD}): {low_n}  "
          f"NOHIT(>{LOW_CONFIDENCE_THRESHOLD}): {no_n}")
    print("  " + "=" * 80)


# ══════════════════════════════════════════════════════════════════
#  7. AdaptiveRAGChain — 三级路由
# ══════════════════════════════════════════════════════════════════

class AdaptiveRAGChain:
    """
    自适应 RAG 链：单路向量检索 + 三级置信度路由。

    路由逻辑：
      HIGH  → 完整 RAG + 严谨 Prompt（零幻觉约束）
      LOW   → RAG + 风险警告 Prompt（强制前置声明）
      NOHIT → 阻断 LLM，直接返回固定话术（零 Token 消耗）

    性能设计：
      - 检索阶段零 LLM 调用，仅一次 Chroma similarity_search_with_score
      - 端到端延迟目标 < 5s（Time to First Token）
    """

    def __init__(self, vectorstore, high_prompt, low_prompt, llm,
                 high_threshold=0.85, low_threshold=1.10):
        self.vectorstore = vectorstore
        self.high_prompt = high_prompt
        self.low_prompt = low_prompt
        self.llm = llm
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.output_parser = StrOutputParser()

        # ── 原生 HTTP 流式配置（绕过 LangChain 缓冲层，实现真·逐 token 到达）──
        self._api_base = "https://llmapi.paratera.com"
        self._api_model = "DeepSeek-V4-Pro"
        self._api_key = api_key  # 模块级全局，来自 load_dotenv() 后的 os.environ

        # 复用 HTTP 连接池，消除每次调用的 TCP+TLS 握手开销（~500-1500ms/次）
        self._http_session = requests.Session()
        self._http_session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        })
        # 预热连接：在评测开始前完成 TCP+TLS 握手，避免第一条用例的冷启动开销
        self._warmup_session()

    def _warmup_session(self):
        """预热 HTTP Session：建立 TCP+TLS 连接，避免首条评测的冷启动延迟。"""
        try:
            self._http_session.post(
                f"{self._api_base}/v1/chat/completions",
                json={
                    "model": self._api_model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "temperature": 0.0,
                    "stream": False,
                    "max_tokens": 1,
                },
                timeout=10,
            )
        except Exception:
            pass  # 预热失败不阻塞初始化

    def _expand_query(self, question: str) -> str:
        """
        基于本地字典的 O(1) 零延迟查询扩展。

        遍历 tax_synonyms.json 中的俗语→法定术语映射：
        - 如果 question 包含某个俗语 key，则提取对应的法定术语列表
        - 将命中的所有法定术语用空格拼接，追加到原问题末尾
        - 绝不替换原问题中的任何词汇，保留用户原始表达
        - 无命中时直接返回原问题

        示例:
            "小微企业的税收优惠"  →  "小微企业的税收优惠 小规模纳税人 小型微利企业"
            "个税起征点是多少"    →  "个税起征点是多少 个人所得税 减除费用 基本减除费用 免征额"
        """
        if not TAX_SYNONYMS:
            return question

        hits: list[str] = []
        for colloquial, formal_terms in TAX_SYNONYMS.items():
            if colloquial in question:
                hits.extend(formal_terms)

        if not hits:
            return question

        # 去重（保留顺序），用空格拼接后追加
        seen: set[str] = set()
        unique_hits: list[str] = []
        for term in hits:
            if term not in seen:
                seen.add(term)
                unique_hits.append(term)

        expanded = question + " " + " ".join(unique_hits)
        return expanded

    def _single_search(self, question: str, k: int = 6):
        """
        单路向量检索：查询扩展 → Chroma 语义检索。

        1. 先通过 _expand_query 将用户口语化词汇扩展为法定术语
        2. 用扩展后的查询语句调用 Chroma.similarity_search_with_score
        """
        expanded_question = self._expand_query(question)
        if RETRIEVAL_DEBUG and expanded_question != question:
            print(f"  🔄 查询扩展: \"{question}\" → \"{expanded_question}\"")
        results = self.vectorstore.similarity_search_with_score(expanded_question, k=k)
        docs = [d for d, _ in results]
        distances = [s for _, s in results]
        return docs, distances

    def _stream_raw(self, messages: list):
        """
        原生 HTTP SSE 流式调用 OpenAI 兼容 API。

        关键设计：
          - 使用 self._http_session（requests.Session）复用 TCP+TLS 连接
          - iter_lines() 逐行解析 SSE，不做任何缓冲聚合
          - 每个 content delta 立即 yield，确保调用方能精确捕获首字到达时间
          - 内部记录 self._http_ttft / self._http_total（纯网络级耗时，
            不含 debug print、format_docs、format_messages 等应用层开销）

        Yields:
            每个 yield 都是 API 返回的一个 content delta 字符串（单个或少量 token）。
        """
        # 将 LangChain Message 对象转为 OpenAI API 格式的 dict
        api_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                api_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                api_messages.append({"role": "user", "content": msg.content})
            elif hasattr(msg, "type") and hasattr(msg, "content"):
                role = "assistant" if msg.type == "ai" else msg.type
                api_messages.append({"role": role, "content": msg.content})
            else:
                # tuple 格式 (role, content)
                role, content = msg
                api_messages.append({"role": role, "content": content})

        payload = {
            "model": self._api_model,
            "messages": api_messages,
            "temperature": 0.0,
            "stream": True,
        }

        # ── 精确计时起点：HTTP POST 发起时刻 ────────────────
        http_start = time.perf_counter()
        response = self._http_session.post(
            f"{self._api_base}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=120,
        )
        response.raise_for_status()

        first_token_recorded = False

        # 逐行读取 SSE 事件流 — 不做任何缓冲，第一个 data 行到达即刻 yield
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        delta = choices[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            # ── 精确首字到达时刻 ────────────
                            if not first_token_recorded:
                                first_token_recorded = True
                                now = time.perf_counter()
                                self._http_ttft = now - http_start
                                self._http_start_at = http_start
                                self._http_first_token_at = now
                            yield content
                except json.JSONDecodeError:
                    continue

        # ── 流结束时刻 ──────────────────────────────────────
        self._http_total = time.perf_counter() - http_start

    def invoke(self, question: str) -> str:
        """同步调用，返回完整回答字符串（内部使用 stream 收集）。"""
        parts = []
        for token in self.stream(question):
            parts.append(token)
        return "".join(parts)

    def stream(self, question: str, pre_docs=None, pre_distances=None):
        """
        流式生成回答，逐个 yield token。

        TTFT 测量（精确到网络级毫秒）：
          - _stream_raw() 内部记录 self._http_ttft（HTTP POST → 首个 content delta）
          - 评测脚本直接从 chain._http_ttft / chain._http_total 读取
          - 这些值在每次 stream() 调用开头被清零，避免 NOHIT 短路的残留值

        参数:
          pre_docs, pre_distances: 可选，评测场景下预先检索好的结果，
                                   避免 stream 内部重复检索
        """
        # ── 清零网络级计时器（防止 NOHIT 残留上一次的值）─────
        self._http_ttft = 0.0
        self._http_total = 0.0

        # ── 第一步：单路检索（零额外 LLM 调用，< 50ms）──────
        if pre_docs is not None and pre_distances is not None:
            docs, distances = pre_docs, pre_distances
        else:
            docs, distances = self._single_search(question, k=6)

        # 打印汇总
        print(f"  📊 检索 distance: {[round(d, 4) for d in distances]}")

        # ── 第二步：三级置信度判定 ─────────────────────────────
        confidence = classify_confidence(distances)

        # 打印调试明细
        _debug_print_retrieval_results(docs, distances, confidence)

        # ── 第三步：按置信度路由 ───────────────────────────────

        if confidence == ConfidenceLevel.NOHIT:
            # ── NOHIT：阻断 LLM，零 Token 消耗 ────────────────
            print(f"  🚫 NOHIT — best distance {min(distances):.4f} > {self.low_threshold}")
            print(f"  ⛔ 阻断 LLM 调用，返回固定话术（零 Token 消耗）")
            yield NOHIT_RESPONSE
            return

        elif confidence == ConfidenceLevel.LOW:
            # ── LOW：RAG + 风险警告 ───────────────────────────
            print(f"  🟡 LOW CONFIDENCE — best distance {min(distances):.4f} "
                  f"在 ({self.high_threshold}, {self.low_threshold}] 区间")
            print(f"  ⚠️ 走 RAG 回答，强制前置风险警告")
            context = format_docs(docs)
            messages = self.low_prompt.format_messages(context=context, input=question)
        else:
            # ── HIGH：完整 RAG + 严谨回答 ─────────────────────
            print(f"  ✅ HIGH CONFIDENCE — best distance {min(distances):.4f} <= {self.high_threshold}")
            print(f"  📋 走完整 RAG 回答，零幻觉约束")
            context = format_docs(docs)
            messages = self.high_prompt.format_messages(context=context, input=question)

        # ── 原生 HTTP SSE 流式输出（绕过 LangChain 缓冲）───────
        print()  # 空行，分隔调试信息与回答
        for token in self._stream_raw(messages):
            print(token, end="", flush=True)
            yield token
        print()  # 收尾换行


# ══════════════════════════════════════════════════════════════════
#  8. 初始化 & 启动
# ══════════════════════════════════════════════════════════════════

rag_chain = AdaptiveRAGChain(
    vectorstore=vectordb,
    high_prompt=high_prompt,
    low_prompt=low_prompt,
    llm=llm,
    high_threshold=HIGH_CONFIDENCE_THRESHOLD,
    low_threshold=LOW_CONFIDENCE_THRESHOLD,
)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  税务 RAG 问答系统 — 三级置信度路由")
    print("=" * 60)
    print(f"  Chroma metric:      L2 distance（默认，越低越相关）")
    print(f"  HIGH (≤{HIGH_CONFIDENCE_THRESHOLD}): 高置信 — RAG + 严谨回答 + 来源引用")
    print(f"  LOW  (≤{LOW_CONFIDENCE_THRESHOLD}):  低置信 — RAG + 风险警告 + 人工复核提示")
    print(f"  NOHIT (>{LOW_CONFIDENCE_THRESHOLD}): 无命中 — 阻断 LLM，固定话术，零 Token")
    print(f"  调试模式:            {'开' if RETRIEVAL_DEBUG else '关'}")
    print("=" * 60)

    # ── 交互式问答循环 ──────────────────────────────────────────
    print("\n  输入 'quit' 或 'exit' 退出\n")


    while True:
        try:
            user_input = input("[?] 请输入您的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  再见！")
            break

        if user_input.lower() in ("quit", "exit", "q"):
            print("  再见！")
            break

        if not user_input:
            continue

        print(f"\n[思考中] 正在检索并生成回答...\n")

        try:
            answer = rag_chain.invoke(user_input)
            # 流式输出已在 invoke() 中完成，此处不再重复打印
            print(f"\n{'='*60}")
            print(f"[回答完成]\n")
        except Exception as e:
            print(f"[错误]: {e}\n")
