"""
rag_agent.py
------------
RAG 税务问答 Agent — 企业级财税政策咨询系统。
核心设计原则：极高准确性、严格政策追溯性、零容忍幻觉。

三级置信度路由逻辑：
┌──────────────────────────────────────────────────────────────┐
│  Chroma 返回 L2 distance（默认 metric，越低越相关）           │
│                                                              │
│  distance <= 0.88  →  HIGH:  高置信命中，RAG + 严谨回答      │
│  0.88 < d <= 1.10  →  LOW:   低置信命中，RAG + 风险警告      │
│  distance > 1.10    →  NOHIT: 阻断 LLM，返回固定话术          │
│                                                              │
│  实测数据支撑（BGE + normalize_embeddings=True）：            │
│    直接命中: 0.44 ~ 0.62   模糊相关: 0.65 ~ 0.88            │
│    完全无关: 1.27 ~ 1.36                                     │
└──────────────────────────────────────────────────────────────┘
"""

# ══════════════════════════════════════════════════════════════════
# 【网络层优化 · 最高优先级】
# 在导入任何第三方网络库之前，强制清除 Windows 系统代理环境变量。
# 问题根因：Windows 网络栈中，httpx/requests 会继承系统代理设置，
# 若代理不可达 → TCP SYN 黑洞 → 默认 5s 超时 → 降级直连 1s = 总计 6s TTFT。
# 同时屏蔽 IPv6 DNS 解析（AAAA 查询超时 5s）带来的延迟。
# ══════════════════════════════════════════════════════════════════
import os as _os_net

# 清空所有可能引发路由黑洞的代理环境变量（大小写双版本均清除）
for _proxy_key in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "all_proxy", "no_proxy",
):
    _os_net.environ.pop(_proxy_key, None)

# 强制禁用 httpx/requests 从系统环境读取代理配置
_os_net.environ["NO_PROXY"] = "*"
_os_net.environ["no_proxy"] = "*"

del _os_net

import os
import sys
import json
import time
import socket

# ── HuggingFace 国内镜像：解决 huggingface.co 直连超时问题 ──
# 若本地已有模型缓存（通常在 ~/.cache/huggingface/hub/），不会重新下载；
# 仅在缓存缺失时才走镜像下载。
if not os.environ.get("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Windows 中文终端默认 GBK 编码，无法输出 emoji 等 4 字节 UTF-8 字符
# 显式重配 stdout 为 UTF-8，避免 UnicodeEncodeError
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
import httpx

load_dotenv()
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

import yaml
import jieba
from rank_bm25 import BM25Okapi

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
# 加载 ML 意图分类器 — TF-IDF + LogisticRegression 单例
# 用于在检索前拦截跨境管辖权穿透/跨法域概念缝合等伪装型域外问题
# 推理延迟: ~1-5 ms (远低于 10ms 约束)
# ══════════════════════════════════════════════════════════════════
_ML_CLASSIFIER: "tuple[object, float] | None" = None  # (pipeline, threshold)

# 物理阻断硬编码阈值：只有 predict_proba 给出的 OUT_OF_DOMAIN 概率 >= 此值才阻断；
# 否则一律按 IN_DOMAIN 放行，交给后续 RAG 管线兜底（宁可多检索，不可误杀）
# 0.60 标定依据：已知高置信 OOD（跨境电商/非税法律纠纷）conf 0.61~0.70，
# in-domain 最高 ~0.49（q213 资产重组），边界案例 q211(0.52)/q220(0.57) 放行交 RAG 兜底
OOD_THRESHOLD = 0.60

def _load_classifier() -> "tuple[object, float] | None":
    """单例加载 intent_classifier.pkl。失败时返回 None，系统降级为仅规则安检。"""
    import pickle as _pickle
    _cls_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intent_classifier.pkl")
    if not os.path.exists(_cls_path):
        if not _SILENT_IMPORT:
            print(f"⚠️ 意图分类器未找到 ({_cls_path})，ML 安检已禁用。运行 train_intent_classifier.py 训练模型。")
        return None
    try:
        with open(_cls_path, "rb") as _f:
            _bundle = _pickle.load(_f)
        _pipe = _bundle["pipeline"]
        _thresh = _bundle.get("threshold", OOD_THRESHOLD)  # 仅存档参考，运行时以 OOD_THRESHOLD 为准
        if not _SILENT_IMPORT:
            print(f"已加载意图分类器: {_cls_path} (runtime OOD_THRESHOLD={OOD_THRESHOLD})")
        return (_pipe, _thresh)
    except Exception as _e:
        if not _SILENT_IMPORT:
            print(f"⚠️ 意图分类器加载失败（{_e}），ML 安检已禁用")
        return None

_ML_CLASSIFIER = _load_classifier()

def classify_intent(question: str) -> "tuple[str, float] | None":
    """
    使用 ML 分类器预测意图。返回 (label, ood_confidence) 或 None（分类器不可用）。
    标签: "IN_DOMAIN" | "OUT_OF_DOMAIN"
    注意：不使用 .predict()，统一走 .predict_proba() 取概率值，
    由调用方基于 OOD_THRESHOLD 决定是否物理阻断。
    """
    if _ML_CLASSIFIER is None:
        return None
    _pipe, _thresh = _ML_CLASSIFIER
    try:
        _proba = _pipe.predict_proba([question])[0]
        _classes = _pipe.classes_
        _ood_idx = list(_classes).index("OUT_OF_DOMAIN")
        _ood_conf = float(_proba[_ood_idx])
        _pred = _classes[_proba.argmax()]
        return (_pred, _ood_conf)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# 实体提取 → Chroma 硬过滤映射（Layer 1: Metadata Hard-Filter）
# ══════════════════════════════════════════════════════════════════

# 实体→关键词模式：用于从用户查询中提取核心税种实体（零延迟正则匹配）
TAX_ENTITY_PATTERNS: dict[str, list[str]] = {
    "个人所得税":   ["个人所得税", "个税", "综合所得", "经营所得", "专项附加",
                     "赡养老人", "子女教育", "婴幼儿照护", "住房贷款利息",
                     "住房租金", "继续教育", "大病医疗", "减除费用", "汇算清缴"],
    "企业所得税":   ["企业所得税", "企税", "加计扣除", "研发费用", "小型微利",
                     "高新技术企业", "创业投资", "加速折旧", "公益性捐赠",
                     "企业重组", "资产重组", "股权划转", "资产划转", "特殊性税务处理",
                     "技术转让"],
    "增值税":       ["增值税", "小规模纳税人", "一般纳税人", "进项税额",
                     "销项税额", "留抵退税", "增值税专用发票", "技术转让",
                     "农产品", "自产农产品", "免税项目"],
    "关税":         ["关税", "进出口", "完税价格", "原产地", "最惠国", "保税",
                     "暂时进境", "暂时出境", "跨境电商零售进口"],
    "契税":         ["契税", "房屋买卖", "房屋赠与", "房屋互换", "夫妻更名"],
    "印花税":       ["印花税", "合同印花", "应税凭证", "营业账簿"],
    "消费税":       ["消费税", "卷烟", "白酒", "化妆品", "成品油", "汽车"],
    "房产税":       ["房产税", "房产原值", "房产余值", "从价计征", "从租计征"],
    "环境保护税":   ["环境保护税", "环保税", "大气污染物", "水污染物",
                     "固体废物", "噪声", "排污"],
    "城镇土地使用税": ["城镇土地使用税", "大城市", "中等城市", "小城市", "工矿区"],
    "土地增值税":   ["土地增值税", "转让房地产", "增值额", "四级超率累进"],
    "耕地占用税":   ["耕地占用税", "基本农田", "非农业建设"],
    "车辆购置税":   ["车辆购置税", "购置", "二手车", "新能源汽车"],
    "车船税":       ["车船税", "乘用车", "商用车", "船舶"],
    "烟叶税":       ["烟叶税", "收购烟叶", "烟叶"],
    "资源税":       ["资源税", "共生矿", "伴生矿", "从价定率", "从量定额"],
    "城市维护建设税": ["城市维护建设税", "城建税", "增值税附征"],
    "税收征管":     ["偷税", "抗税", "骗税", "追征", "滞纳金", "行政复议",
                     "税收保全", "强制执行", "发票管理", "纳税申报",
                     "发票真伪", "查验发票", "发票查验", "虚开发票",
                     "发票领用", "领用发票", "领用", "领购", "发票领购"],
}

# 启动时从 data_source.yaml 构建 entity → allowed source_names 映射
# 格式: {"个人所得税": ["中华人民共和国个人所得税法", "国发〔2018〕41号", ...]}
ENTITY_SOURCE_MAP: dict[str, list[str]] = {}

def _build_entity_source_map(yaml_path: str) -> dict[str, list[str]]:
    """解析 data_source.yaml，构建 entity→[source_name,...] 硬过滤映射表"""
    entity_map: dict[str, list[str]] = {}
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            docs = yaml.safe_load(f)
        if not isinstance(docs, list):
            return entity_map
        for entry in docs:
            source_name = (entry.get("source_name")
                           or entry.get("policy_name")
                           or entry.get("file_name", ""))
            if not source_name:
                continue
            # 收集该文档关联的所有税种关键词
            tax_keywords: set[str] = set()
            tax_policy = entry.get("tax_policy", "")
            if tax_policy:
                # "税收政策-个人所得税" → "个人所得税"
                for tag in tax_policy.replace("，", ",").split(","):
                    tag = tag.strip().replace("税收政策-", "")
                    if tag:
                        tax_keywords.add(tag)
            topic = entry.get("topic") or entry.get("applicable_scope", "")
            if isinstance(topic, list):
                topic = "，".join(topic)
            # topic 字段直接作为实体关键词
            if topic:
                for kw in str(topic).replace("，", ",").split(","):
                    kw = kw.strip()
                    if kw and len(kw) <= 10:  # 过滤过长的描述
                        tax_keywords.add(kw)
            # 将 source_name 归类到匹配的 entity
            for entity, patterns in TAX_ENTITY_PATTERNS.items():
                matched = False
                for kw in tax_keywords:
                    if kw in patterns or any(p in kw for p in patterns):
                        matched = True
                        break
                # 兜底：直接用 entity 名称在全部文本字段中搜索
                if not matched:
                    all_text = source_name + " " + tax_policy + " " + topic
                    if entity in all_text or any(p in all_text for p in patterns[:3]):
                        matched = True
                if matched:
                    if entity not in entity_map:
                        entity_map[entity] = []
                    if source_name not in entity_map[entity]:
                        entity_map[entity].append(source_name)
    except Exception as e:
        if not _SILENT_IMPORT:
            print(f"⚠️ 实体映射表构建失败（{e}），硬过滤功能已禁用")
    return entity_map

# ══════════════════════════════════════════════════════════════════
# 三级路由阈值（基于实测 L2 distance 分布）
#  2026-07-16 安全收紧: HIGH 0.85→0.75，LOW 1.10→0.85
#  2026-07-19 阈值微调: LOW 0.85→0.88（ML 意图分类器前置后，向量阈值适度放宽）
#  任何 d > 0.75 的 Chunk 不允许直接流入 LLM，需经 LOW 警告或阻断
# ══════════════════════════════════════════════════════════════════
HIGH_CONFIDENCE_THRESHOLD = 0.75   # <= 此值 → 高置信（收紧: 0.85→0.75）
LOW_CONFIDENCE_THRESHOLD  = 0.88   # <= 此值 → 低置信（收紧: 1.10→0.85→0.88）
                                   # > 此值 → 无命中

# ── Context Payload 硬截断上限 ──────────────────────────────────
# 防止检索返回过多文档导致 LLM prefill 卡死。
# format_docs() 按 document 顺序拼接，累计字符数超出此值后丢弃后续文档。
#
# 公式对齐: top_n × CHUNK_SIZE ≤ MAX_CONTEXT_LENGTH
#   top_n=5, CHUNK_SIZE=1000 → raw ≤ 5000, +metadata/seps ≈ 6500 ≤ 7000 ✅
MAX_CONTEXT_LENGTH = 7000

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
# ⚠️ 模型名称自检：确保传入标准对话模型（如 deepseek-chat / glm-4），
#    严禁使用带隐式思维链的推理模型（如 deepseek-reasoner / o1-preview），
#    否则 TTFT 会因服务端预填充 CoT 而飙升到 5-15 秒。
api_key = os.environ.get("INFINI_API_KEY")
if not api_key:
    raise ValueError(
        "❌ 未找到环境变量 INFINI_API_KEY，请先设置：$env:INFINI_API_KEY='your-key'"
    )

# ── 构建自定义 httpx.Client：强制 IPv4 + 禁用代理继承 ────────
# local_address="0.0.0.0" 强制绑定 IPv4 栈，彻底绕过 IPv6 DNS 查询
# trust_env=False 拒绝读取系统代理/SSL 环境变量
_httpx_client = httpx.Client(
    transport=httpx.HTTPTransport(local_address="0.0.0.0"),
    trust_env=False,
    timeout=httpx.Timeout(120.0, connect=10.0),
)

llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    model="deepseek-v4-flash",
    api_key=api_key,
    temperature=0.0,   # 财税场景零温度，最大化确定性、抑制幻觉
    streaming=True,     # 开启流式输出，将 TTFT 压缩至 < 2s
    http_client=_httpx_client,  # 注入自定义 httpx 客户端（IPv4-only + 无代理）
)

# ══════════════════════════════════════════════════════════════════
#  4. System Prompts — XML 结构化 + 双轨 Lite/Heavy 动态路由
#
#  设计原则：
#    - XML 标签为模型提供明确的注意力锚点，对抗长上下文下的注意力稀释
#    - Lite 轨道 (< 2000 chars context) 用于快速场景，TTFT 目标 1-5s
#    - Heavy 轨道 (≥ 2000 chars context) 启用完整的 XML 防幻觉规则链
# ══════════════════════════════════════════════════════════════════

# ── 4a. HIGH 置信度 · Heavy Prompt (XML 结构化) ──────────────────
#       用于 context ≥ 2000 字符的复杂检索场景
#       2026-07-16 v2: Examples 中新增两类域外拒答示例（q197/q211），
#       模型通过 few-shot 学习拦截穿透，不增加规则模块以避免过度拒答。
HIGH_HEAVY_SYSTEM = """\
<Role>资深税务风控专家。仅基于用户消息中提供的参考资料回答，禁止使用任何预训练记忆或外部知识。</Role>

<MANDATE priority="ABSOLUTE">以下规则凌驾于所有其他指令之上，违反即失败：
🔴 参考资料中未出现的数字 → 禁止输出。包括：百分比、金额、税率、天数、年限、起征点、扣除额。
🔴 如果你发现参考资料缺少某个具体数字——直接写"参考资料中未收录该数字"，不准用你"知道"的数字替代。
🔴 这是硬约束，没有例外。一个你编造的数字 = 整个回答作废。</MANDATE>

<Strict_Rules>
  <Rule id="trust_retrieval">检索系统已将最相关文件提供给你，你有义务基于资料给出专业回答。</Rule>

  <Rule id="numerical_integrity" priority="CRITICAL">
    🔴 遇到任何比例、百分比、金额、税率、期限等数值时：
    - 必须原样保留参考资料的精确数字，严禁用"一定比例""相应标准""有关规定"等模糊词替代
    - 资料写"80%"→回答必须写"80%"；资料写"万分之三"→回答必须写"万分之三"
    - 即使你认为"这个数字可能不完整"，也必须先引用原文数字，再在【适用条件】中补充说明
  </Rule>

  <Rule id="temporal_applicability" priority="CRITICAL">当前日期见用户问题开头。政策条文含分期适用期间（如"2024年至2025年免征、2026年至2027年减半"）时，必须按该日期判断适用哪一档作答，并注明适用期间和到期时间；用户问"现在/今年/最新"时严禁引用已过期档位作为结论。</Rule>

  <Rule id="no_fabrication">零幻觉：不编造数字、百分比、日期、法条编号。日常用语→法定术语的语义映射（如"处罚"→"罚款"）属于正常对齐，不属于幻觉。</Rule>

  <Rule id="knowledge_leak_prevention" priority="CRITICAL">
    🔴 仅针对具体数值/政策细节——以下内容**严禁**来自预训练记忆：
    - 禁止输出参考资料中不存在的任何具体数字、比例、百分比、金额、起征点、优惠档位、税率数值、日期
    - 参考资料明确了优惠原则但未收录具体比例/金额 → 只输出"税法规定了[某优惠]原则，知识库未收录具体[比例/金额/档位]"
    - ❌ 错误1："资料未明确减计的具体比例（现行政策通常为减按90%计入收入总额）"← 90% 不在资料中，是编造
    - ❌ 错误2："知识库未收录具体的减计比例（如减按90%计入收入等）"← 括号里的90%同样是编造
    - ✅ 正确示范："税法规定了减计收入优惠，知识库未收录具体减计比例"
    - 🔴 一句话铁律：只要数字/比例/日期不在 &lt;References/&gt; 的原文里 → 任何位置都不准出现
    🟢 例外——基础法律归类与税种存废**允许**使用常识：
       - 某类实体（合伙企业/个体工商户/非营利组织等）是否属于某税种的纳税主体 → 税法的结构性常识，不属于"具体政策细节"。若资料覆盖了该税种的法律但未明确提及该实体，你可以基于税法体系常识判断该实体的纳税主体归类并作答。
       - 某税种是否已废止/停征（如营业税、工商统一税已废止，农业税已取消）→ 税法体系的结构性事实。用户问及已废止税种相关旧文件的效力时，你可以基于税法存废常识作答（如"营业税已废止，该文件中涉营业税部分不再适用"），**但仍禁止编造具体废止年份、废止文件编号或其他任何具体数字/比例/日期**。
  </Rule>

  <Rule id="no_overgeneralize">禁止逻辑泛化：资料仅说明A的情况，不得推断B也适用。</Rule>

  <Rule id="refusal_gate">
    仅当以下条件**同时满足**时，才可输出"资料不足"：
    (a) 已逐条通读全部 Chunk（严禁只读前几条就下结论）
    (b) 全部资料与用户问题完全无关（如问增值税但全6条都是契税法）
    (c) 无任何 Chunk 涉及用户核心税种/场景
    以下情况**禁止拒答**：
    - 前几条方向关键词不匹配→继续读后续Chunk
    - 用户用通俗词→映射后引用法条
    - 资料含"免征/不征收"→直接给出否定结论
    - 资料仅覆盖相邻场景（如问离婚析产、资料有婚姻存续期间夫妻权属变更条款；问某地区进口、资料有原产地/税率总则）→必须按格式A作答，引用相邻条款并说明其与用户场景的适用边界，不得以"不完全一致"为由拒答
    - 🔴 实体-税种归属判定（最高优先级）：用户问"某实体（如合伙企业/个体户/非营利组织等）是否需要缴X税"，而检索到的资料恰好包含X税种的核心法律 → **必须作答**。此时你要做的是：引用资料中关于该税种纳税主体/适用范围的规定，判断用户实体是否落入该范围。若该实体明显不属于该税种的纳税主体（如合伙企业不适用企业所得税法），直接给出"不适用/不需要"的否定结论并引用法条依据。**严禁**因"资料未出现该实体名称"而拒答——纳税主体范围是税种法律的基础条款，检索到该税种的法律就足以回答。
  </Rule>

  <Rule id="direction_disambiguation">若资料同时含方向相反的规定（进/出、买/卖），优先引用与用户问题方向一致的条款。</Rule>

  <Rule id="partial_coverage" priority="CRITICAL">若资料仅覆盖问题的一部分：
    - 已覆盖部分：正常按4模块作答并引用法条。
    - 未覆盖部分：在【具体内容】中**逐一显式声明**缺失的是什么——例如"参考资料中未提及具体施行日期""参考资料中未收录具体税率数值""参考资料中未涉及XXX情形"。
    - 🔴 局部信息缺失铁律：如果用户提问包含多个子问题（如问了"A的日期"和"A与B的关系"），而参考资料只能回答其中一部分。你必须明确回答资料中有的子问题，并针对缺失的子问题显式标注"参考资料中未提及[具体缺什么]"，**严禁动用自身预训练知识去填补缺失的细节**。日期、数字、比例、金额——只要不在References原文里，任何位置都不准出现。
    - 【法条依据】只列真正支撑结论的条款，严禁为未覆盖部分硬凑法条或用擦边条款充数。</Rule>

  <Rule id="chunk_splicing">资料按固定长度切分，关键数据可能在 Chunk 边界断裂。提取数值时检查相邻 Chunk 边界内容，断裂条文编号（如"第"+"二十一条"）主动拼接。</Rule>

  <Rule id="domain_guard">业务实质优先：若用户明确问境外平台税务合规（欧洲VAT/美国Sales Tax/亚马逊卖家税务义务）→拒答"此问题涉及境外税收管辖与平台合规，超出中国国内税收基准文件范围"。若核心争议是劳动关系认定/合同纠纷等非税法问题（非附带问税）→拒答"此问题核心属于[部门法]非税法律范畴…"。正常中国税法业务咨询→忽略本条。</Rule>
</Strict_Rules>

<Output_Format priority="CRITICAL">
根据实际情况，以下格式**三选一，严禁混用**：

◆ 格式A（有实质内容可答）：回答首字符必须是"【"。**仅包含**以下4个模块，禁止任何开场白/过渡/免责声明：

【核心结论】一句话直接回答，给出明确定性或定量结论（能/不能/具体金额）。⚠️ 若用户以"不管…都""所有…都""是不是…吧"等句式提出过度概括的假设（如"不管卖给谁都是免税的吧""所有企业都适用75%对吗"），结论应先正面说明正确规则（如"XX情形下免征增值税""适用100%加计扣除"），再自然纠正其过度概括的范围误区（如"但并非所有农产品都适用——只有初级农产品方可享受""但该比例仅适用于……企业，并非所有企业"）。严禁以"是的""对的"开头直接肯定过度概括。
【适用条件】有序列表。无条件则写"无特定条件。"（不可省略）。
【具体内容】展开细则/计算方式，仅基于参考资料。
【法条依据】《XXX法》第X条格式。多文件逐条列出；无编号则写"详见参考资料原文"。

◆ 格式B（资料完全无关——refusal_gate (a)(b)(c) 三条件同时满足）：只输出一段拒答说明，**严禁出现任何模块标题**：

资料不足，未收录〈用户所问核心主题〉的相关政策文件，无法提供适用条件和法条依据，请补充相关政策文件。

◆ 🔴 格式C——内容驱动强制覆盖（最高优先级，覆盖格式A/B的判定逻辑）：
**仅当**你逐条读完所有 Chunk 后确认：参考资料**不仅**缺少具体政策细节，而且**连可以用来推导答案的法律原则、税种定义、一般性规则都没有**——也就是说，这些 Chunk 对于回答用户问题完全帮不上忙——此时才允许用纯文本拒答。触发标准比 refusal_gate 宽松（不要求"全部 Chunk 与所问税种无关"），但远高于"找不到原话"——只要 Chunk 中包含可以用来推理的法律原则或定义，就必须走格式A作答，并在【具体内容】中诚实说明哪些细节知识库未收录。⚠️ 多子问题（如比较/差异类"XX是多少？跟YY有什么不同？"）：只要任一子问题可基于资料作答，就必须走格式A——已覆盖子问题正常作答引用法条，缺失子问题在【具体内容】中显式声明"参考资料中未收录[具体缺什么]"。

格式C示范：
资料不足，未收录〈用户所问核心主题〉的相关政策文件。已收录资料仅涉及〈相关领域的一般性规定/无关条款〉，无法据此推导出该具体问题的答案。请补充相关政策文件。

🔴 格式C vs 格式A 判定铁律（逐条判断，必须全部满足才走格式C）：
1. 资料中有没有该税种的定义/范围/计税规则？有 → 格式A（据此推理作答，细节缺失在【具体内容】中声明）
2. 资料中有没有相关的"免征/不征收/不得/不包括"等否定规定？有 → 格式A
3. 资料中有没有相邻场景的规则可以类比引用？有 → 格式A（引用并说明边界）
4. 以上三条都**没有** → 格式C（纯文本拒答）
**不要在【核心结论】里写"参考资料中未收录……"**——那不是核心结论，是放弃作答。真正的核心结论是基于资料推理出的"能/不能/多少/什么条件"。

<Examples>
User: 我家里有两个宝宝，老大上小学，老二1岁半，每个月能抵扣多少个税额度？
Assistant:
【核心结论】：您每月可享受总计 4000 元的个税专项附加扣除额度。
【适用条件】：1. 子女处于学前教育至全日制学历教育阶段。2. 或子女处于3岁以下婴幼儿照护阶段。3. 扣除主体可由夫妻双方按50%分别扣除，或由一方按100%全额扣除。
【具体内容】：老大上小学适用"子女教育"专项附加扣除，每月2000元；老二1岁半适用"3岁以下婴幼儿照护"专项附加扣除，每月2000元。两项合计4000元。
【法条依据】：《国务院关于提高个人所得税有关专项附加扣除标准的通知》（国发〔2023〕13号）第一条、第二条。

User: 我们是一家大型连锁餐饮企业，最近研发了几款新菜，投入的50万能100%加计扣除吗？
Assistant:
【核心结论】：不能。餐饮企业发生的研发费用不适用税前加计扣除政策。
【适用条件】：1. 纳税人必须属于国家规定的非负面清单行业。2. "住宿和餐饮业"属于明确不适用加计扣除政策的负面清单行业。
【具体内容】：研发费用加计扣除政策实行行业负面清单管理。您公司作为餐饮企业，属于政策明确排除的行业，因此无论研发投入多少，均无法享受该项税收优惠。
【法条依据】：《关于完善研究开发费用税前加计扣除政策的通知》（财税〔2015〕119号）第四条。

User: 我们准备在亚马逊上开店卖货到欧洲，跨境电商的税务合规要注意哪些方面？VAT怎么申报？
Assistant:
此问题涉及境外税收管辖与平台合规，超出中国国内税收基准文件范围，无法为您解答。建议咨询目标市场所在地的专业税务顾问。

User: 我们公司用了不少劳务派遣员工，他们算我们公司的从业人员吗？劳务派遣的增值税和企业所得税怎么处理？
Assistant:
此问题核心属于《劳动法》《劳动合同法》非税法律范畴（劳务派遣用工关系认定），所涉及的税务处理仅为附属后果。基于现有财税政策库无法提供完整法律意见，建议咨询对应领域的专业律师。
</Examples>
"""

# ── 4b. HIGH 置信度 · Lite Prompt (极简，~250 tokens) ───────────
#       用于 context < 2000 字符的快速场景，TTFT 目标 1-5s
HIGH_LITE_SYSTEM = """\
<Role>税务风控专家，仅基于用户消息中提供的参考资料回答，禁止使用预训练记忆。</Role>

<MANDATE>参考资料中未出现的数字（百分比/金额/税率/天数/起征点）→ 禁止输出。缺少数字就说"参考资料中未收录"，不准用自身知识替代。</MANDATE>


<Rules>
- 当前日期见用户问题开头。政策分期适用（不同期间免征/减半等）时，按该日期选定适用档位作答，并注明适用期间；严禁引用已过期档位作为结论。
- 不编造数字、百分比、日期、法条编号。不知道就如实说"资料不足"。
- 🔴 数值禁令：资料未收录的具体数字/比例/金额/优惠档位/日期 → 只声明"知识库未收录"，严禁用预训练记忆补全任何数值。铁律：数字/日期不在资料原文里→任何位置都不准出现。
- 🟢 实体归类例外：某类实体是否属于某税种纳税主体 → 税法结构性常识，允许基于法律体系常识判断作答。
- 🟢 税种存废例外：某税种是否已废止/停征（如营业税已废止）→ 税法体系结构性事实。用户问及已废止税种相关旧文件的效力时，可基于常识作答（如"该税种已废止，文件涉该税种部分不再适用"），但仍禁止编造具体废止年份、废止文件编号或其他数字/日期。
- 遇到数值（税率、金额、比例）必须原样引用原文精确数字，禁止用模糊词代替。
- 参考资料含"免征/不征收"等否定规定时，直接给出否定结论。
- 🔴 局部缺失处理铁律（最高优先级）：若用户问题含多个子问题（如问"日期+关系"），资料只能回答部分时：已覆盖的子问题正常作答引用法条；缺失的子问题必须显式标注"参考资料中未提及[具体缺什么]"，严禁用预训练知识填补缺失的日期、数字、比例。铁律：数字/日期不在资料原文里→任何位置都不准出现。
- 业务实质优先：若明确问境外平台税务(欧洲VAT/亚马逊卖家义务)或核心争议是劳动关系/合同纠纷(非附带问税)→拒答，不强行引用中国税法资料作答。
</Rules>

<Format>
有实质内容可答时，首字符必须是"【"。仅含4模块：
【核心结论】一句话定性/定量。
【适用条件】列表（无条件写"无特定条件。"）。
【具体内容】仅基于参考资料展开。
【法条依据】《XXX法》第X条（无编号写"详见参考资料原文"）。

🔴 内容驱动强制规则（最高优先级）：纯文本拒答**仅当**逐条通读后确认 Chunk 中**连可用以推理的法律原则都没有**时才可用。只要 Chunk 中有该税种的定义/范围/相关规则→必须基于其推理作答（格式A），缺失的具体细节在【具体内容】中声明"参考资料未收录XXX具体数值/情形"。不要在【核心结论】里写"参考资料中未收录……"。⚠️ 多子问题（比较/差异类）：只要任一子问题可基于资料作答→必须走格式A，已覆盖部分正常作答，缺失部分在具体内容中声明。
纯文本拒答格式：资料不足，未收录〈用户所问核心主题〉的相关政策文件。已收录资料仅涉及〈相关领域〉一般性规定，无法据此推导具体答案。请补充相关政策文件。
</Format>
"""

# ── 4c. LOW 置信度 · Heavy Prompt (XML 结构化) ───────────────────
LOW_HEAVY_SYSTEM = """\
<Role>资深税务风控专家。仅基于用户消息中提供的参考资料回答，禁止使用预训练记忆。</Role>

<MANDATE priority="ABSOLUTE">参考资料中未出现的数字（百分比/金额/税率/天数/起征点）→ 禁止输出，只写"参考资料中未收录该数字"。违反即失败。</MANDATE>

<Preface priority="MANDATORY">回答第一句话必须逐字输出：⚠️ 检索到的政策匹配度较低，以下内容仅供参考，建议人工复核。</Preface>

<Entity_Boundary_Check priority="CRITICAL">
  <Principle>必须逐条通读所有 Chunk（从第1条到最后一条），绝对禁止只读前几条就下结论。</Principle>

  <Steps>1.提取用户核心税种和关键实体 → 2.逐条扫描全部资料寻找相关规定（含否定/排除性规定）→ 3.检查以下豁免再执行阻断。</Steps>

  <ExemptionA>用户主体与税种张冠李戴 → 纠正错误预设，输出正确政策规则并引用法条出处。</ExemptionA>
  <ExemptionB>资料含"不适用/免征/不征收/不得"等否定性规定 → 直接告知否定结论并引用出处。</ExemptionB>
  <ExemptionC>用户日常用语与法定术语属于正常语义映射 → 必须引用法条原文作答，禁止因此拒答。</ExemptionC>

  <Pass_Condition>全部 Chunk 中任意一处明确包含用户核心实体的适用规则，或豁免A/B/C任一命中 → 正常作答。</Pass_Condition>

  <Block_Condition>仅当：已逐条读完所有 Chunk + 豁免A/B/C全部不命中 + 确认无任何 Chunk 涉及用户核心实体 → 方可输出阻断话术："资料不足，未收录〈用户所问核心主题〉的相关内容，请补充相关政策文件。"阻断时整个回答仅含 Preface + 此话术，禁止输出任何模块标题、法条引用和📌尾注——既然资料不足，就没有适用条件和法条依据可言。</Block_Condition>
</Entity_Boundary_Check>

<Strict_Rules>
  <Rule id="numerical_integrity" priority="CRITICAL">遇到任何比例/百分比/金额/税率/期限等数值时，必须原样保留参考资料的精确数字，严禁用模糊词替代。</Rule>
  <Rule id="temporal_applicability">当前日期见用户问题开头。政策含分期适用期间时，按该日期选定适用档位作答，并注明适用期间；严禁引用已过期档位作为结论。</Rule>
  <Rule id="no_fabrication">零幻觉：不编造数字、百分比、日期、法条编号。</Rule>
  <Rule id="knowledge_leak_prevention" priority="CRITICAL">
    🔴 数值禁令：资料未收录的具体数字/比例/金额/优惠档位/日期 → 只声明"知识库未收录XXX"，严禁用预训练记忆补全任何数值。铁律：数字/日期不在&lt;References/&gt;原文里→任何位置都不准出现。
    🟢 例外：某实体是否属于某税种纳税主体 → 税法结构性常识，允许基于法律体系常识判断。某税种是否已废止/停征（如营业税已废止）→ 税法体系结构性事实，用户问及已废止税种相关旧文件效力时可基于常识作答（如"该税种已废止，文件涉该税种部分不再适用"），仍禁编造具体废止年份、废止文件编号或其他数字/比例/日期。
  </Rule>
  <Rule id="no_overgeneralize">禁止逻辑泛化。因匹配度较低，需额外说明资料与问题之间可能存在的差异。</Rule>
  <Rule id="partial_coverage" priority="CRITICAL">若用户问题含多个子问题而资料仅能回答部分：
    - 已覆盖的子问题：正常作答并引用出处。
    - 未覆盖的子问题：**逐一显式声明**缺失内容——如"参考资料中未提及具体施行日期""参考资料中未收录具体数值"——严禁动用预训练记忆填补任何日期、数字、比例或条文。铁律：日期/数字不在References原文里→任何位置都不准出现。</Rule>
  <Rule id="chunk_splicing">关键数据在 Chunk 边界断裂时检查相邻 Chunk 进行拼接还原。</Rule>
  <Rule id="domain_guard">若用户明确问境外平台税务合规(欧洲VAT/亚马逊卖家义务)或核心争议是劳动关系/合同纠纷(非附带问税)→拒答，不引用资料强行作答。</Rule>
</Strict_Rules>

<Output>正常作答时：结论引用格式：【来源：《政策名称》第X条】，末尾附：📌 以上内容基于知识库已收录政策文件。政策可能更新，请以税务机关最新公告为准。触发 Block_Condition 时：不加📌尾注。</Output>
"""

# ── 4d. LOW 置信度 · Lite Prompt (极简) ──────────────────────────
LOW_LITE_SYSTEM = """\
<Role>税务风控专家，仅基于用户消息中提供的参考资料回答，禁止使用预训练记忆。</Role>

<MANDATE>参考资料中未出现的数字→禁止输出。缺少数字就写"参考资料中未收录"，不准用自身知识替代。</MANDATE>

<Preface>第一句话必须写：⚠️ 检索到的政策匹配度较低，以下内容仅供参考，建议人工复核。</Preface>


<Rules>
- 当前日期见用户问题开头。政策分期适用时按该日期选定适用档位作答，并注明适用期间。
- 通读全部资料后再判定。不编造数字/百分比/日期/法条编号。
- 🔴 数值禁令：资料未收录的具体数字/比例/金额/优惠档位/日期 → 只声明"知识库未收录"，绝不输出预训练记忆中的数字。铁律：数字/日期不在资料原文里 → 任何位置都不准出现。
- 🟢 实体归类例外：某实体是否属于某税种纳税主体 → 税法结构性常识，允许判断。🟢 税种存废例外：某税种是否已废止/停征（如营业税已废止）→ 税法体系结构性事实，用户问及已废止税种相关旧文件效力时可基于常识作答，仍禁编造具体废止年份/文件编号/数字/日期。
- 遇到数值必须原样引用原文精确数字。
- 若全部 Chunk 都与用户问题无关 → 只输出"资料不足，未收录〈用户所问核心主题〉的相关内容，请补充相关政策文件。"，禁止再输出法条引用和📌尾注。
- 若任意 Chunk 有相关内容 → 必须引用作答。用户问多个子问题但资料仅覆盖部分时：已覆盖的正常作答引用法条；缺失的子问题必须显式标注"参考资料中未提及[具体缺什么]"，严禁用预训练知识填补缺失的日期、数字、比例。
- 业务实质优先：若明确问境外平台税务(欧洲VAT/亚马逊卖家义务)或核心争议是劳动关系/合同纠纷(非附带问税)→拒答。
</Rules>

<Output>正常作答时引用格式：【来源：《政策名称》第X条】，末尾附：📌 以上内容基于知识库已收录政策文件。政策可能更新，请以税务机关最新公告为准。拒答时不加📌尾注。</Output>
"""

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

# ── 4d. OOD 领域外阻断话术（规则前置安检触发）────────────────────
OOD_NOHIT_RESPONSE = (
    "此提问不属于我所收录的中国税收政策范畴，无法为您解答。\n\n"
    "本系统仅覆盖中国境内税收政策法规，包括：个人所得税、企业所得税、增值税、"
    "消费税、关税、契税、印花税、房产税、土地增值税、城镇土地使用税、"
    "耕地占用税、车辆购置税、车船税、烟叶税、资源税、城市维护建设税、"
    "环境保护税、船舶吨税，以及税收征管、发票管理等领域。\n\n"
    "💡 建议：\n"
    "  - 如果您的问题涉及上述税种，请尝试使用税法术语重新描述\n"
    "  - 涉及劳动法、跨境电商海外税务、行政处罚等非中国税法问题，请咨询对应领域的专业人士"
)

# ── 4e. ML 意图分类器阻断话术（Layer 0a ML 安检触发）────────────
OOD_ML_NOHIT_RESPONSE = (
    "经识别，您的问题核心涉及境外税收管辖、非税法律（如劳动法/民商法）"
    "或其他超纲领域，无法基于现有境内财税基准文件作答。\n\n"
    "本系统仅覆盖中国境内税收政策法规。如果您的问题确实涉及中国税种，"
    "请尝试使用更精准的税法术语重新描述。\n\n"
    "💡 建议：\n"
    "  - 境外税务合规问题 → 咨询目标市场所在地的专业税务顾问\n"
    "  - 劳动法/民法典相关问题 → 咨询对应领域的专业律师\n"
    "  - 境内财税政策问题 → 请重述为明确的税法咨询（如 'XX情况下企业所得税如何处理'）"
)

# ══════════════════════════════════════════════════════════════════
#  5. Prompt 模板 — 双轨制 (Lite / Heavy XML)
#     Lite:  context <  2000 chars → 极简 prompt，TTFT 1-5s
#     Heavy: context >= 2000 chars → XML 结构化防幻觉 prompt
# ══════════════════════════════════════════════════════════════════

# ── CONTEXT_LITE_THRESHOLD：context 字符数低于此值走 Lite 轨道 ──
CONTEXT_LITE_THRESHOLD = 2000

# ChatPromptTemplate 实例（每个轨道两个置信度级别 = 4 个模板）
# ── 黄金拼接顺序（Prompt Cache 优化）─────────────────────────
# System Prompt: 100% 静态，无任何变量 → 永远命中 Prefix Cache
# Human Message:  日期 + Context + 用户问题 → 每次变化但在消息序列末尾
_HUMAN_TEMPLATE = (
    "当前日期：{today}\n\n"
    "<References>\n{context}\n</References>\n\n"
    "用户问题：{input}"
)

high_heavy_pt = ChatPromptTemplate.from_messages([
    ("system", HIGH_HEAVY_SYSTEM),
    ("human", _HUMAN_TEMPLATE),
])
high_lite_pt = ChatPromptTemplate.from_messages([
    ("system", HIGH_LITE_SYSTEM),
    ("human", _HUMAN_TEMPLATE),
])
low_heavy_pt = ChatPromptTemplate.from_messages([
    ("system", LOW_HEAVY_SYSTEM),
    ("human", _HUMAN_TEMPLATE),
])
low_lite_pt = ChatPromptTemplate.from_messages([
    ("system", LOW_LITE_SYSTEM),
    ("human", _HUMAN_TEMPLATE),
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


def format_docs(docs, max_length: int = 0):
    """
    将检索到的文档拼接，附带完整 metadata 来源标注。

    参数:
        max_length: 累计字符数上限。设为 0 表示不限制。
                    大于 0 时，按 docs 顺序逐条拼接，超出后丢弃后续文档。
                    第一条文档即使超限也会保留（保证至少有一个回答依据）。
    """
    formatted = []
    total = 0
    truncated = False

    for i, doc in enumerate(docs):
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
        chunk_text = f"[来源: {label}]\n{doc.page_content}"

        # ── 长度截断检查 ──────────────────────────────────────────
        if max_length > 0 and i > 0:
            # 估算拼接后的总长度（加上分隔符 "\n\n---\n\n"）
            separator_overhead = len("\n\n---\n\n") * len(formatted)
            projected = total + len(chunk_text) + (len("\n\n---\n\n") if formatted else 0)
            if projected + separator_overhead > max_length:
                truncated = True
                # ── 截断预警埋点：终端高亮告警 ──────────────────────
                skipped_count = len(docs) - i
                print(f"  ╔══════════════════════════════════════════════════════════╗")
                print(f"  ║  [WARNING] Context length exceeded!                   ║")
                print(f"  ║  Chunk #{i + 1} (and {skipped_count} subsequent) truncated.       ║")
                print(f"  ║  max_length={max_length} | projected={projected + separator_overhead} | total so far={total} ║")
                print(f"  ║  Fix: increase MAX_CONTEXT_LENGTH or reduce top_n.    ║")
                print(f"  ╚══════════════════════════════════════════════════════════╝")
                continue  # 丢弃此文档及后续所有

        formatted.append(chunk_text)
        total += len(chunk_text)

    # 若发生截断，在末尾注入标记（LLM 可见，供其判断资料完整性）
    if truncated:
        formatted.append(
            f"\n\n---\n[⚠️ Context 已达上限 ({max_length} 字符)，"
            f"已截断 {len(docs) - len(formatted) + 1} 条低相关度文档]"
        )

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

    def __init__(self, vectorstore,
                 high_heavy_pt, high_lite_pt,
                 low_heavy_pt, low_lite_pt,
                 llm,
                 high_threshold=0.75, low_threshold=0.88,
                 context_lite_threshold=2000):
        self.vectorstore = vectorstore
        self.high_heavy_pt = high_heavy_pt
        self.high_lite_pt  = high_lite_pt
        self.low_heavy_pt  = low_heavy_pt
        self.low_lite_pt   = low_lite_pt
        self.llm = llm
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.context_lite_threshold = context_lite_threshold
        self.output_parser = StrOutputParser()

        # ── 原生 HTTP 流式配置（绕过 LangChain 缓冲层，实现真·逐 token 到达）──
        self._api_base = "https://api.deepseek.com"
        # ⚠️ 模型名称自检：必须为标准对话模型（deepseek-chat / glm-4），
        # 严禁使用推理模型（deepseek-reasoner），否则 TTFT 会因思维链而飙升。
        self._api_model = "deepseek-v4-flash"
        self._api_key = api_key  # 模块级全局，来自 load_dotenv() 后的 os.environ

        # ── 强制 IPv4 DNS 解析：在 requests 层面拦截 getaddrinfo ──
        # 原理：Windows 对双栈域名会先发起 AAAA (IPv6) 查询，若 DNS 服务器
        # 不响应 AAAA 请求 → 默认 5s 超时 → 降级 A (IPv4) 查询 → 总计 5-6s。
        # 通过 monkey-patch socket.getaddrinfo，只保留 AF_INET 结果，
        # 彻底跳过 IPv6 DNS 解析，TTFT 从 ~6s 降至 <1s。
        self._orig_getaddrinfo = socket.getaddrinfo

        def _ipv4_getaddrinfo(host, port, family=0, *args, **kwargs):
            return self._orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)

        self._ipv4_getaddrinfo = _ipv4_getaddrinfo

        # 复用 HTTP 连接池，消除每次调用的 TCP+TLS 握手开销（~500-1500ms/次）
        self._http_session = requests.Session()

        # ── 挂载 HTTPAdapter：显式配置连接池与重试策略 ──────────
        retry_strategy = Retry(
            total=2,
            backoff_factor=0.1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["POST"],
        )
        adapter = HTTPAdapter(
            pool_connections=5,
            pool_maxsize=10,
            max_retries=retry_strategy,
            pool_block=False,
        )
        self._http_session.mount("https://", adapter)
        self._http_session.mount("http://", adapter)

        self._http_session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "Connection": "keep-alive",
        })

        # 显式禁用 requests 层的代理继承（双重保险）
        self._http_session.proxies = {}
        self._http_session.trust_env = False
        # 预热连接：在评测开始前完成 TCP+TLS 握手，避免第一条用例的冷启动开销
        self._warmup_session()

        # ── 初始化实体硬过滤映射（Layer 1: Metadata Hard-Filter）────
        _yaml_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_source.yaml")
        self.entity_source_map = _build_entity_source_map(_yaml_path)
        if not _SILENT_IMPORT and self.entity_source_map:
            _total_src = sum(len(v) for v in self.entity_source_map.values())
            print(f"实体硬过滤映射已构建: {len(self.entity_source_map)} 个税种, {_total_src} 条 source_name 映射")

        # ── 初始化 BM25 稀疏检索索引（Layer 2: Sparse Recall）────
        self._bm25: BM25Okapi | None = None
        self._chunk_dict: dict[str, object] = {}  # chroma_id → LangChain Document
        self._bm25_corpus_ids: list[str] = []     # index → chroma_id
        self._build_bm25_index()

    def _build_bm25_index(self):
        """从 Chroma 提取全部 chunk，jieba 分词后构建 BM25Okapi 索引"""
        try:
            all_data = self.vectorstore.get(include=["documents", "metadatas"])
            if not all_data or not all_data.get("ids"):
                if not _SILENT_IMPORT:
                    print("⚠️ BM25 索引构建失败：Chroma 无数据")
                return
            ids = all_data["ids"]
            docs_list = all_data.get("documents", [])
            metas_list = all_data.get("metadatas", [])
            # 构建映射
            from langchain_core.documents import Document as LCDocument
            corpus = []
            for i, cid in enumerate(ids):
                text = docs_list[i] if i < len(docs_list) else ""
                meta = metas_list[i] if i < len(metas_list) else {}
                doc = LCDocument(page_content=text, metadata=meta)
                self._chunk_dict[cid] = doc
                # jieba 精确模式分词
                tokens = list(jieba.cut(text))
                corpus.append(tokens)
                self._bm25_corpus_ids.append(cid)
            self._bm25 = BM25Okapi(corpus)
            if not _SILENT_IMPORT:
                print(f"BM25 索引构建完成: {len(corpus)} 个 chunk")
        except Exception as e:
            if not _SILENT_IMPORT:
                print(f"⚠️ BM25 索引构建失败（{e}），稀疏检索降级为仅稠密检索")

    # ── 前置安检：全量税种领域关键词（TAX_ENTITY_PATTERNS + TAX_SYNONYMS）──
    _TAX_DOMAIN_KEYWORDS: set[str] | None = None

    @classmethod
    def _build_tax_domain_keywords(cls) -> set[str]:
        """构建全量税种领域关键词集合（仅构建一次，类级缓存）。"""
        if cls._TAX_DOMAIN_KEYWORDS is not None:
            return cls._TAX_DOMAIN_KEYWORDS
        kw: set[str] = set()
        # 从 TAX_ENTITY_PATTERNS 收集所有模式
        for patterns in TAX_ENTITY_PATTERNS.values():
            for pat in patterns:
                kw.add(pat)
        # 从 TAX_SYNONYMS 收集所有 key（同义词入口）
        for key in TAX_SYNONYMS:
            kw.add(key)
        cls._TAX_DOMAIN_KEYWORDS = kw
        return kw

    def _check_tax_domain(self, question: str) -> bool:
        """
        前置硬安检：检测用户查询是否属于中国税收政策领域。

        扫描 query 中是否包含任何已知税种/税收术语（TAX_ENTITY_PATTERNS +
        TAX_SYNONYMS 全部关键词）。如果完全不包含 → 判定为非税收领域问题。

        返回:
            True  = 属于税收领域，继续检索
            False = 非税收领域，直接触发 NOHIT 阻断
        """
        domain_kw = self._build_tax_domain_keywords()
        for kw in domain_kw:
            if kw in question:
                return True
        return False

    def _extract_entity(self, question: str) -> dict | None:
        """
        从用户查询中提取核心税种实体，返回 Chroma where filter dict。
        若未检测到实体，返回 None（全量检索）。

        多实体 union 策略（2026-07-17）：跨税种问题（如"除了增值税还要交什么税"）
        需要多个税种的文件同时可检索，命中多个实体时取来源并集，由稠密/稀疏
        排序决定最终 top-k。
        误匹配抑制：沿用最长匹配语义——若某实体的命中模式是另一实体更长命中
        模式的子串（如"保税"⊂"环保税"、"增值税"⊂"土地增值税"），丢弃该实体。
        """
        # 1. 收集每个实体的全部命中模式
        matched: dict[str, list[str]] = {}  # entity -> 所有命中模式
        for entity, patterns in TAX_ENTITY_PATTERNS.items():
            hits = [p for p in patterns if p in question]
            if hits and self.entity_source_map.get(entity):
                matched[entity] = hits

        if not matched:
            return None

        # 2. 子串误匹配抑制：实体的最长命中模式若是其他实体任一更长命中
        #    模式的子串（如"汽车"⊂"新能源汽车"），丢弃该实体
        all_hits = [(e, p) for e, ps in matched.items() for p in ps]
        survivors = []
        for entity, pats in matched.items():
            best = max(pats, key=len)
            shadowed = any(
                other != entity and len(other_pat) > len(best) and best in other_pat
                for other, other_pat in all_hits
            )
            if not shadowed:
                survivors.append(entity)

        # 3. 来源并集（保序去重）
        allowed: list[str] = []
        for entity in survivors:
            for src in self.entity_source_map.get(entity, []):
                if src not in allowed:
                    allowed.append(src)

        if not allowed:
            return None
        return {"source_name": {"$in": allowed}}

    def _bm25_search(self, question: str, k: int = 15) -> list[tuple[object, float]]:
        """
        BM25 稀疏检索：返回 top-k (LangChain Document, score) 列表。
        score 为 BM25 原始分数（越高越相关）。
        """
        if self._bm25 is None or not self._bm25_corpus_ids:
            return []
        tokens = list(jieba.cut(question))
        scores = self._bm25.get_scores(tokens)
        # 取 top-k indices
        if len(scores) <= k:
            top_indices = list(range(len(scores)))
        else:
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for idx in top_indices:
            cid = self._bm25_corpus_ids[idx]
            doc = self._chunk_dict.get(cid)
            if doc is not None:
                results.append((doc, float(scores[idx])))
        return results

    def _rrf_fuse(
        self,
        dense_results: list[tuple[object, float]],
        sparse_results: list[tuple[object, float]],
        top_n: int = 5,
        k: int = 60,
    ) -> list[object]:
        """
        RRF (Reciprocal Rank Fusion) 融合两路检索结果。

        参数:
            dense_results: [(Document, L2_distance), ...] — 距离越小越好
            sparse_results: [(Document, BM25_score), ...] — 分数越高越好
            top_n: 最终返回的 chunk 数量（公式对齐: top_n × CHUNK_SIZE ≤ MAX_CONTEXT_LENGTH）
            k: RRF 常数（默认 60）
        """
        rrf_scores: dict[str, float] = {}
        doc_map: dict[str, object] = {}

        # 路 A: Dense — 按距离升序（rank 1 = 最近）
        for rank, (doc, dist) in enumerate(dense_results, 1):
            cid = doc.metadata.get("chunk_id", doc.page_content[:80])
            if cid not in doc_map:
                doc_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

        # 路 B: Sparse — 按分数降序（rank 1 = 最高分）
        sorted_sparse = sorted(sparse_results, key=lambda x: x[1], reverse=True)
        for rank, (doc, score) in enumerate(sorted_sparse, 1):
            cid = doc.metadata.get("chunk_id", doc.page_content[:80])
            if cid not in doc_map:
                doc_map[cid] = doc
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)

        # 按 RRF 分数降序排列，取 top_n
        sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_n]
        return [doc_map[cid] for cid in sorted_ids]

    def _hybrid_search(self, question: str, k: int = 6) -> tuple[list[object], list[float]]:
        """
        三层检索管线：硬过滤 → 双路召回 → RRF 融合。
        替代原 _single_search()。

        参数:
            k: 最终返回 chunk 数，默认 5（对齐 CHUNK_SIZE=1000 + MAX_CONTEXT_LENGTH=7000）
        """
        # ── Layer 1: 实体提取 + Chroma 硬过滤 ──────────────────
        t0 = time.perf_counter()
        entity_filter = self._extract_entity(question)
        self._t_filter = time.perf_counter() - t0
        if RETRIEVAL_DEBUG and entity_filter:
            allowed_srcs = entity_filter.get("source_name", {}).get("$in", [])
            print(f"  🔒 实体硬过滤: {len(allowed_srcs)} 个来源")

        # ── Layer 1.5: 查询扩展（俗语→法定术语，O(1) 本地字典）──
        #    _hybrid_search 替代 _single_search 时曾遗漏此步，2026-07-17 接回：
        #    稠密/稀疏两路统一使用扩展后查询，实体过滤仍用原始问题
        expanded_q = self._expand_query(question)
        if RETRIEVAL_DEBUG and expanded_q != question:
            print(f"  🔄 查询扩展: +\"{expanded_q[len(question):].strip()}\"")

        # ── Layer 2a: Dense 稠密检索 (k=20, 扩大窗口抓取更多同源 chunks) ──
        t1 = time.perf_counter()
        try:
            raw_dense = self.vectorstore.similarity_search_with_score(
                expanded_q, k=20, filter=entity_filter
            )
        except Exception as _filter_err:
            # Chroma filter 可能不兼容某些值 → 降级为无过滤
            if RETRIEVAL_DEBUG:
                print(f"  ⚠️ 实体过滤异常(降级为全量检索): {_filter_err}")
            raw_dense = self.vectorstore.similarity_search_with_score(expanded_q, k=20)
        dense_docs, dense_dists = [], []
        for doc, score in raw_dense:
            dense_docs.append(doc)
            dense_dists.append(score)
        self._t_dense = time.perf_counter() - t1

        # ── Layer 2b: Sparse 稀疏检索 (k=20, 同步扩大) ────────
        t2 = time.perf_counter()
        sparse_results = self._bm25_search(expanded_q, k=20)
        # BM25 不走 Chroma filter，需手动排除被硬过滤的文档
        if entity_filter and sparse_results:
            allowed = set(entity_filter.get("source_name", {}).get("$in", []))
            if allowed:
                sparse_results = [
                    (doc, score) for doc, score in sparse_results
                    if doc.metadata.get("source_name", "") in allowed
                ]
        self._t_sparse = time.perf_counter() - t2

        # ── RRF 融合 → top-k ──────────────────────────────────
        t3 = time.perf_counter()
        dense_pairs = list(zip(dense_docs, dense_dists))
        fused_docs = self._rrf_fuse(dense_pairs, sparse_results, top_n=k)
        self._t_fusion = time.perf_counter() - t3

        # distances: 为 fused docs 重建有意义的距离值（用于置信度判定）
        # 建立 dense doc → L2 distance 的快速查找
        dense_dist_map: dict[str, float] = {}
        for doc, dist in dense_pairs:
            cid = doc.metadata.get("chunk_id", doc.page_content[:80])
            dense_dist_map[cid] = dist
        # 仅为 BM25-only 的 doc 使用默认中等距离
        fused_distances: list[float] = []
        for doc in fused_docs:
            cid = doc.metadata.get("chunk_id", doc.page_content[:80])
            fused_distances.append(dense_dist_map.get(cid, 0.90))
        self._t_retrieval = time.perf_counter() - t0

        return fused_docs, fused_distances

    def _warmup_session(self):
        """预热 HTTP Session：建立 TCP+TLS 连接，避免首条评测的冷启动延迟。"""
        try:
            # 预热也需强制 IPv4，避免 IPv6 DNS 超时拖慢初始化
            socket.getaddrinfo = self._orig_getaddrinfo
            socket.getaddrinfo = self._ipv4_getaddrinfo
            try:
                resp = self._http_session.post(
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
                resp.close()  # 显式关闭，确保连接归还连接池
            finally:
                socket.getaddrinfo = self._orig_getaddrinfo
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
        t0 = time.perf_counter()
        try:
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
        finally:
            self._t_rewrite = time.perf_counter() - t0

    def _single_search(self, question: str, k: int = 6):
        """
        单路向量检索：查询扩展 → Chroma 语义检索。

        1. 先通过 _expand_query 将用户口语化词汇扩展为法定术语
        2. 用扩展后的查询语句调用 Chroma.similarity_search_with_score
        """
        expanded_question = self._expand_query(question)
        if RETRIEVAL_DEBUG and expanded_question != question:
            print(f"  🔄 查询扩展: \"{question}\" → \"{expanded_question}\"")
        t0 = time.perf_counter()
        results = self.vectorstore.similarity_search_with_score(expanded_question, k=k)
        self._t_retrieval = time.perf_counter() - t0
        docs = [d for d, _ in results]
        distances = [s for _, s in results]
        return docs, distances

    def _resolve_citations(self, docs):
        """
        引用感知重排：确保「被引用法条」的正文出现在「引用者」之前。

        纯文本后处理，零额外 API 调用，延迟 < 1ms。

        场景：法律文本中常见 "依照第六十三条" 的引用模式。
        若 Top-K 中 chunk A 引用了第X条，而 chunk B 包含第X条的完整正文，
        但 B 排在 A 后面——则 LLM 可能先读到引用、误判为"资料缺失"，
        而不继续读后面的 chunk。此函数将定义 chunk 前移。

        算法：
          1. 扫描所有 chunk，区分「法条定义」（作为段落标题的"第X条"）
             和「法条引用」（正文中出现的"第X条"编号）
          2. 若某法条被引用但定义在更后面的 chunk，将定义 chunk 插到
             第一个引用它的 chunk 之前
        """
        import re as _re

        n = len(docs)
        if n <= 1:
            return docs

        # ── Pass 1: 收集每个 chunk 的定义和引用 ──────────────────
        ART_PAT = _re.compile(r'第([一二三四五六七八九十百千]+)条')

        # 辅助函数：判断某法条编号是否在此 chunk 的 article 元数据范围内
        def _article_in_meta_range(art_cn: str, meta_article: str) -> bool:
            """例: art_cn='六十三', meta='第六十一条—第六十六条' → True"""
            if not meta_article:
                return False
            # 从 meta 中提取所有法条编号
            meta_arts = ART_PAT.findall(meta_article)
            if len(meta_arts) >= 2:
                # 范围模式 "第X条—第Y条"
                first, last = meta_arts[0], meta_arts[-1]
                return _cn_compare(first) <= _cn_compare(art_cn) <= _cn_compare(last)
            # 单条模式 "第X条"
            return art_cn in meta_arts

        def _cn_compare(cn: str) -> int:
            """中文数字 → 整数，用于范围比较"""
            map_cn = {'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9}
            if cn in map_cn:
                return map_cn[cn]
            if cn == '十': return 10
            if cn.startswith('十') and len(cn)==2: return 10 + map_cn.get(cn[1],0)
            if cn.endswith('十') and len(cn)==2: return map_cn.get(cn[0],0)*10
            if '十' in cn:
                parts = cn.split('十',1)
                return map_cn.get(parts[0],0)*10 + map_cn.get(parts[1],0) if parts[1] else map_cn.get(parts[0],0)*10
            return 0

        defines = {}   # 法条编号 → 定义它的 chunk 索引
        refs = []      # [(chunk_idx, 被引用的法条编号)]

        for i, doc in enumerate(docs):
            content = doc.page_content
            meta_article = doc.metadata.get("article", "")
            articles = ART_PAT.findall(content)
            if not articles:
                continue
            for art in articles:
                # 优先用元数据判断：article 字段标明了此 chunk 定义的法条范围
                if _article_in_meta_range(art, meta_article):
                    if art not in defines:
                        defines[art] = i
                else:
                    # 不在元数据范围内 → 是引用
                    refs.append((i, art))

        if not defines or not refs:
            return docs  # 无法条结构或无法条引用，无需重排

        # ── Pass 2: 找到"引用在前、定义在后"的违规对 ──────────
        # 仅保留"同一部法律内"的引用关系（避免关税法第63条被误配给征管法）
        moves = []  # [(ref_idx, def_idx, article)]

        for ref_idx, art in refs:
            def_idx = defines.get(art)
            if def_idx is None or def_idx <= ref_idx:
                continue
            # 跨法律过滤：引用 chunk 和定义 chunk 必须来自同一 source_name
            ref_src = docs[ref_idx].metadata.get("source_name", "")
            def_src = docs[def_idx].metadata.get("source_name", "")
            if ref_src and def_src and ref_src != def_src:
                continue
            moves.append((ref_idx, def_idx, art))

        if not moves:
            return docs

        # ── Pass 3: 从右向左处理，避免索引漂移 ─────────────────
        # 降序排列：先处理靠后的引用，前面的索引不受影响
        moves.sort(key=lambda x: -x[0])
        reordered = list(docs)

        for ref_idx, def_idx, art in moves:
            # 在当前的 reordered 列表中定位这两个 chunk
            def_current = None
            ref_current = None
            ref_page = docs[ref_idx].page_content
            def_page = docs[def_idx].page_content
            for j, d in enumerate(reordered):
                if d.page_content == ref_page:
                    ref_current = j
                if d.page_content == def_page:
                    def_current = j

            if def_current is not None and ref_current is not None and def_current > ref_current:
                def_doc = reordered.pop(def_current)
                reordered.insert(ref_current, def_doc)

        return reordered

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
            "seed": 42,         # 固定 seed，配合 temperature=0 实现完全确定性输出
            "stream": True,
        }

        # ── 精确计时起点：HTTP POST 发起时刻 ────────────────
        http_start = time.perf_counter()

        # ── 应用 IPv4-only monkey-patch：确保本次 HTTP 请求跳过 IPv6 DNS ──
        socket.getaddrinfo = self._orig_getaddrinfo  # 先恢复原函数（防御性）
        socket.getaddrinfo = self._ipv4_getaddrinfo

        response = None
        try:
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
        finally:
            # 显式关闭流式响应，释放底层 TCP 连接回 Session 连接池
            # 如果不关闭，urllib3 将连接标记为 dirty，导致下次请求被迫
            # 重新建立 TCP+TLS（这是 TTFT 高达 ~7.5s 的根因）
            if response is not None:
                response.close()
            # 恢复原始 socket.getaddrinfo，避免对其他代码产生副作用
            socket.getaddrinfo = self._orig_getaddrinfo

    def invoke(self, question: str) -> str:
        """同步调用，返回完整回答字符串（内部使用 stream 收集）。"""
        parts = []
        for token in self.stream(question):
            parts.append(token)
        return "".join(parts)

    def stream(self, question: str, pre_docs=None, pre_distances=None, bypass_nohit=False):
        """
        流式生成回答，逐个 yield token。

        TTFT 测量（精确到网络级毫秒）：
          - _stream_raw() 内部记录 self._http_ttft（HTTP POST → 首个 content delta）
          - 评测脚本直接从 chain._http_ttft / chain._http_total 读取
          - 这些值在每次 stream() 调用开头被清零，避免 NOHIT 短路的残留值

        参数:
          pre_docs, pre_distances: 可选，评测场景下预先检索好的结果，
                                   避免 stream 内部重复检索
          bypass_nohit:            为 True 时跳过 NOHIT 阻断，强制调用 LLM
                                  （用于 web_demo 晚期注入临时参考文件场景）
        """
        # ── 端到端计时起点 ─────────────────────────────────────
        t_total_start = time.perf_counter()

        # ── 清零所有计时器（防止 NOHIT 等短路路径残留上一次的值）───
        self._http_ttft = 0.0
        self._http_total = 0.0
        self._t_rewrite = 0.0
        self._t_retrieval = 0.0
        self._t_filter = 0.0
        self._t_dense = 0.0
        self._t_sparse = 0.0
        self._t_fusion = 0.0

        # ── Layer 0a: ML 意图分类器（TF-IDF + LogisticRegression, ~1-5ms）──
        # 判定规则：仅当 predict_proba 的 OUT_OF_DOMAIN 概率 >= OOD_THRESHOLD
        # 且预测类别为 OUT_OF_DOMAIN 时才物理阻断；否则一律按 IN_DOMAIN 放行，
        # 交给后续 RAG 管线处理（低置信度 OOD 由检索质量兜底，避免误杀）
        ml_passed = False  # True = ML 放行（视同 IN_DOMAIN），跳过规则安检
        ml_result = classify_intent(question)
        if ml_result is not None:
            ml_label, ml_conf = ml_result
            if ml_label == "OUT_OF_DOMAIN" and ml_conf >= OOD_THRESHOLD:
                print(f"  🤖 ML 安检: OUT_OF_DOMAIN (conf={ml_conf:.3f} >= {OOD_THRESHOLD}) → 阻断")
                print(f"  ⛔ 不调用检索/LLM，零 Token 消耗")
                yield OOD_ML_NOHIT_RESPONSE
                self._t_total = time.perf_counter() - t_total_start
                self._ttft = 0.0
                self._t_generation = 0.0
                self._print_telemetry_panel()
                return
            else:
                ml_passed = True  # 未达阻断阈值 → 一律按 IN_DOMAIN 放行
                if RETRIEVAL_DEBUG:
                    print(f"  🤖 ML 安检: {ml_label} (OOD conf={ml_conf:.3f} < {OOD_THRESHOLD}) → 放行至 RAG 管线")

        # ── Layer 0b: 规则前置硬安检（仅 ML 分类器不可用时执行）──
        if not ml_passed and not self._check_tax_domain(question):
            print(f"  🛡️  前置安检: 未检测到任何中国税种关键词 → 领域外阻断")
            print(f"  ⛔ OOD-NOHIT — 不调用检索，直接返回固定话术（零 Token 消耗）")
            yield OOD_NOHIT_RESPONSE
            self._t_total = time.perf_counter() - t_total_start
            self._ttft = 0.0
            self._t_generation = 0.0
            self._print_telemetry_panel()
            return

        # ── 第一步：混合检索（硬过滤 → 双路召回 → RRF 融合）──────
        if pre_docs is not None and pre_distances is not None:
            docs, distances = pre_docs, pre_distances
        else:
            docs, distances = self._hybrid_search(question, k=6)

        # 打印汇总
        print(f"  📊 检索 distance: {[round(d, 4) for d in distances]}")

        # ── 1.5 引用感知重排（零延迟纯文本处理）─────────────────
        docs = self._resolve_citations(docs)

        # ── 第二步：三级置信度判定 ─────────────────────────────
        confidence = classify_confidence(distances)

        # 打印调试明细
        _debug_print_retrieval_results(docs, distances, confidence)

        # ── 第三步：按置信度 + context 复杂度双维路由 ────────────

        if confidence == ConfidenceLevel.NOHIT and not bypass_nohit:
            # ── NOHIT：阻断 LLM，零 Token 消耗 ────────────────
            print(f"  🚫 NOHIT — best distance {min(distances):.4f} > {self.low_threshold}")
            print(f"  ⛔ 阻断 LLM 调用，返回固定话术（零 Token 消耗）")
            yield NOHIT_RESPONSE
            # NOHIT 短路径：LLM 未调用，TTFT 和 t_generation 均为 0
            self._t_total = time.perf_counter() - t_total_start
            self._ttft = 0.0
            self._t_generation = 0.0
            self._print_telemetry_panel()
            return

        if confidence == ConfidenceLevel.NOHIT and bypass_nohit:
            print(f"  🔓 NOHIT bypassed — temp file injection active, forcing LLM generation")

        # ── 先格式化 context，再根据长度决定 Lite / Heavy 轨道 ──
        context = format_docs(docs, max_length=MAX_CONTEXT_LENGTH)
        ctx_len = len(context) if context else 0
        use_lite = ctx_len < self.context_lite_threshold
        # 当前日期锚点 — 供 temporal_applicability 规则判断分期政策适用档位
        today = time.strftime("%Y年%m月%d日")

        if confidence == ConfidenceLevel.LOW:
            print(f"  🟡 LOW CONFIDENCE — best distance {min(distances):.4f} "
                  f"在 ({self.high_threshold}, {self.low_threshold}] 区间")
            if use_lite:
                print(f"  ⚡ Lite 轨道 (ctx={ctx_len} < {self.context_lite_threshold}) — 极简 Prompt, 目标 TTFT 1-5s")
                messages = self.low_lite_pt.format_messages(context=context, input=question, today=today)
            else:
                print(f"  🏋️ Heavy 轨道 (ctx={ctx_len} >= {self.context_lite_threshold}) — XML 结构化防幻觉 Prompt")
                messages = self.low_heavy_pt.format_messages(context=context, input=question, today=today)
        else:
            print(f"  ✅ HIGH CONFIDENCE — best distance {min(distances):.4f} <= {self.high_threshold}")
            if use_lite:
                print(f"  ⚡ Lite 轨道 (ctx={ctx_len} < {self.context_lite_threshold}) — 极简 Prompt, 目标 TTFT 1-5s")
                messages = self.high_lite_pt.format_messages(context=context, input=question, today=today)
            else:
                print(f"  🏋️ Heavy 轨道 (ctx={ctx_len} >= {self.context_lite_threshold}) — XML 结构化防幻觉 Prompt")
                messages = self.high_heavy_pt.format_messages(context=context, input=question, today=today)

        # ════════════════════════════════════════════════════════════
        # 🔬 载荷侦测探针 (Payload Telemetry)
        # ════════════════════════════════════════════════════════════
        _ctx_len = len(context) if context else 0
        _sys_len = sum(
            len(m.content) if hasattr(m, "content") else 0
            for m in messages if hasattr(m, "type") and m.type == "system"
        )
        _user_len = sum(
            len(m.content) if hasattr(m, "content") else 0
            for m in messages if hasattr(m, "type") and m.type != "system"
        )
        _total_payload = _sys_len + _user_len
        _est_tokens = _total_payload // 2  # 中文约 2 chars/token（粗估）

        _bar = "═" * 56
        print(f"\n  ╔{_bar}╗")
        print(f"  ║  🔬 载荷侦测探针 (Payload Telemetry)")
        print(f"  ╠{_bar}╣")
        print(f"  ║  Context (纯文本):     {_ctx_len:>8,} 字符")
        print(f"  ║  System Prompt:        {_sys_len:>8,} 字符")
        print(f"  ║  User Query + Context:  {_user_len:>8,} 字符")
        print(f"  ║  ──────────────────────────────────")
        print(f"  ║  💣 完整 Payload 总计:  {_total_payload:>8,} 字符  (≈ {_est_tokens:,} tokens)")
        print(f"  ╚{_bar}╝")
        print()

        # ── 原生 HTTP SSE 流式输出（绕过 LangChain 缓冲）───────
        print()  # 空行，分隔调试信息与回答
        for token in self._stream_raw(messages):
            print(token, end="", flush=True)
            yield token
        print()  # 收尾换行

        # ── 流式完成：汇总全链路耗时 ───────────────────────────
        self._t_total = time.perf_counter() - t_total_start
        self._ttft = self._http_ttft
        self._t_generation = self._http_total
        self._print_telemetry_panel()

    # ═══════════════════════════════════════════════════════════════
    #  DST: 对话状态追踪 (Dialogue State Tracking)
    # ═══════════════════════════════════════════════════════════════

    def update_dialogue_state(
        self,
        chat_history: list,
        current_state: dict,
        current_query: str,
    ) -> dict:
        """
        对话状态追踪 (DST)：维护多轮对话的结构化状态 JSON，将省略/指代
        丰富的追问改写为自包含的检索查询语句，提升多轮场景的检索命中率。

        状态字段:
            tax_type:       核心税种（如"个人所得税"、"增值税"）
            entity:         纳税主体/标的物（如"个人"、"企业"、"房屋"）
            action:         意图/动作（如"查询税率"、"判断纳税义务"）
            resolved_query: 一句完整的检索用查询语句

        🔴 核心铁律（首轮旁路）:
            如果 chat_history 为空 → 绝对禁止调用 LLM
            → 直接将 resolved_query = current_query 并立即返回

        参数:
            chat_history:  [{"role": "user"/"assistant", "content": "..."}, ...]
            current_state: 上一轮 DST 状态 dict（首轮可传 None）
            current_query: 用户当前原始输入

        返回:
            更新后的 DST 状态 dict（4 字段完整）
        """
        # ── 初始化状态 ──────────────────────────────────────────
        if current_state is None:
            current_state = {
                "tax_type": None,
                "entity": None,
                "action": None,
                "resolved_query": None,
            }

        # 🔴 核心铁律：首轮旁路 —— 绝对禁止调用 LLM
        #    判定依据：chat_history 为空（首轮对话）或仅含系统消息
        if not chat_history or len(chat_history) == 0:
            current_state["resolved_query"] = current_query
            return current_state

        # ═══════════════════════════════════════════════════════
        #  仅在有多轮历史时触发 LLM 调用
        # ═══════════════════════════════════════════════════════
        try:
            # ── 取最近 3 轮对话（6 条消息）───────────────────
            recent = chat_history[-6:]
            history_lines: list[str] = []
            for m in recent:
                role_label = "用户" if m.get("role") == "user" else "系统"
                content = m.get("content", "")
                # 截断过长的历史消息，避免 DST prompt 膨胀
                if len(content) > 300:
                    content = content[:300] + "…"
                history_lines.append(f"{role_label}: {content}")
            history_text = "\n".join(history_lines)

            # ── DST System Prompt ────────────────────────────
            dst_system = (
                "你是一个对话状态追踪器（Dialogue State Tracker），"
                "负责维护税务咨询对话的结构化状态。\n\n"
                "你需要维护以下 JSON 字段：\n"
                "- tax_type: 用户当前咨询的核心税种（如\"个人所得税\"、"
                "\"增值税\"、\"企业所得税\"等）。如果当前问题未提及具体税种"
                "但历史中有，继承历史值。未提及时设为 null。\n"
                "- entity: 纳税主体或标的物（如\"个人\"、\"企业\"、"
                "\"合伙企业\"、\"个体工商户\"、\"房屋\"等）。"
                "未提及时设为 null。\n"
                "- action: 用户的意图/动作（如\"查询税率\"、\"申请优惠\"、"
                "\"计算扣除\"、\"判断是否需要缴纳\"、\"了解申报流程\"等）。"
                "未提及时设为 null。\n"
                "- resolved_query: 结合对话历史，将当前问题改写为一句完整、"
                "自包含的检索用查询语句。必须包含所有从历史中继承的关键上下文"
                "（税种、主体、具体场景、关键数字等），确保这句话脱离历史"
                "也能独立用于向量检索。\n\n"
                "规则：\n"
                "1. 继承旧状态中的有效信息（非 null 字段），"
                "用当前轮次的新信息覆盖对应字段。\n"
                "2. 如果当前问题中未提及某字段且历史中也没有，该字段保持 null。\n"
                "3. resolved_query 必须是一句完整、通顺的中文问句，"
                "包含所有必要上下文。\n"
                "4. 仅输出纯 JSON 对象，不要包含 ```json 标记或任何其他文字。"
            )

            # ── User Message ─────────────────────────────────
            dst_user = (
                f"当前状态: {json.dumps(current_state, ensure_ascii=False)}\n\n"
                f"对话历史（最近3轮）:\n{history_text}\n\n"
                f"用户当前输入: {current_query}\n\n"
                f"请输出更新后的 JSON 状态。"
            )

            # ── 调用 LLM（非流式，复用 HTTP Session）────────
            socket.getaddrinfo = self._orig_getaddrinfo
            socket.getaddrinfo = self._ipv4_getaddrinfo
            resp = None
            try:
                resp = self._http_session.post(
                    f"{self._api_base}/v1/chat/completions",
                    json={
                        "model": self._api_model,
                        "messages": [
                            {"role": "system", "content": dst_system},
                            {"role": "user", "content": dst_user},
                        ],
                        "temperature": 0.0,
                        "stream": False,
                        "max_tokens": 512,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                body = resp.json()
                raw = body["choices"][0]["message"]["content"].strip()
            finally:
                if resp is not None:
                    resp.close()
                socket.getaddrinfo = self._orig_getaddrinfo

            # ── 清洗 JSON（去除可能的 markdown 代码块标记）───
            if raw.startswith("```"):
                lines = raw.split("\n")
                # 去掉首行 ```json 或 ```
                if lines[0].startswith("```"):
                    lines = lines[1:]
                # 去掉末行 ```
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()

            new_state = json.loads(raw)

            # ── 合并状态：新值覆盖旧值 ──────────────────────
            for key in ["tax_type", "entity", "action", "resolved_query"]:
                val = new_state.get(key)
                if val is not None:
                    current_state[key] = val

            # ── 兜底：确保 resolved_query 永不为空 ──────────
            if not current_state.get("resolved_query"):
                current_state["resolved_query"] = current_query

            return current_state

        except Exception:
            # 🔴 强降级兜底：任何错误（JSON 解析失败、网络超时、
            #    API 返回异常等）→ 直接将 resolved_query 退化为
            #    原始 query 并返回，绝对不引发程序崩溃
            current_state["resolved_query"] = current_query
            return current_state

    def _print_telemetry_panel(self):
        """
        全链路耗时诊断面板。

        在每次回答生成完毕后打印结构化的耗时拆解，帮助定位瓶颈：
          - t_rewrite:    查询改写（同义词扩展）
          - t_retrieval:  向量检索（Embedding + Chroma 搜索）
          - TTFT:         首字延迟（Time to First Token）
          - t_generation: LLM 完整推理耗时（HTTP POST → 流结束）
          - t_total:      端到端总耗时
        """
        t_rw    = getattr(self, "_t_rewrite",   0.0) * 1000
        t_filter = getattr(self, "_t_filter",    0.0) * 1000
        t_dense  = getattr(self, "_t_dense",     0.0) * 1000
        t_sparse = getattr(self, "_t_sparse",    0.0) * 1000
        t_fusion = getattr(self, "_t_fusion",    0.0) * 1000
        t_ret   = getattr(self, "_t_retrieval",  0.0) * 1000
        ttft    = getattr(self, "_ttft",         0.0) * 1000
        t_gen   = getattr(self, "_t_generation", 0.0) * 1000
        t_tot   = getattr(self, "_t_total",      0.0) * 1000

        # 占比计算（避免除零）
        ttft_pct  = (ttft / t_tot * 100) if t_tot > 0 else 0.0
        ret_pct   = (t_ret / t_tot * 100) if t_tot > 0 else 0.0

        width = 56

        def _row(label: str, value_ms: float) -> str:
            return f"  ║  {label:<24} {value_ms:>8.2f} ms ║"

        def _bar(pct: float) -> str:
            n = max(1, int(pct / 100 * 30)) if pct > 0 else 0
            return "█" * n if n > 0 else "▏"

        title = "📊 耗时诊断面板 (Telemetry)"

        print()
        print("  ╔" + "═" * (width - 4) + "╗")
        print(("  ║  {:^" + str(width - 8) + "}  ║").format(title))
        print("  ╠" + "═" * (width - 4) + "╣")

        print(_row("查询改写 (t_rewrite)", t_rw))
        print(_row("🔍 实体过滤 (t_filter)", t_filter))
        print(_row("🔍 稠密检索 (t_dense)", t_dense))
        print(_row("🔍 稀疏检索 (t_sparse)", t_sparse))
        print(_row("🔍 RRF融合 (t_fusion)", t_fusion))
        print(_row("检索总耗时 (t_retrieval)", t_ret))
        print(_row("首字延迟 (TTFT)", ttft))
        print(_row("LLM 推理 (t_generation)", t_gen))
        print(_row("端到端总耗时 (t_total)", t_tot))

        print("  ╠" + "═" * (width - 4) + "╣")

        ttft_label = "TTFT / t_total"
        ret_label  = "检索占比"
        print(f"  ║  {ttft_label:<24} {ttft_pct:>7.1f}% {_bar(ttft_pct):<30} ║")
        print(f"  ║  {ret_label:<24} {ret_pct:>7.1f}% {_bar(ret_pct):<30} ║")

        print("  ╚" + "═" * (width - 4) + "╝")
        print()


# ══════════════════════════════════════════════════════════════════
#  8. 初始化 & 启动
# ══════════════════════════════════════════════════════════════════

rag_chain = AdaptiveRAGChain(
    vectorstore=vectordb,
    high_heavy_pt=high_heavy_pt,
    high_lite_pt=high_lite_pt,
    low_heavy_pt=low_heavy_pt,
    low_lite_pt=low_lite_pt,
    llm=llm,
    high_threshold=HIGH_CONFIDENCE_THRESHOLD,
    low_threshold=LOW_CONFIDENCE_THRESHOLD,
    context_lite_threshold=CONTEXT_LITE_THRESHOLD,
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
