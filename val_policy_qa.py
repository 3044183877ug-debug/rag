"""
val_policy_qa.py
----------------
政策问答系统离线评测脚本。

评测维度：
  1. 相关性（Relevance）        — 回答是否切题、准确回应用户问题
  2. 引用准确率（Citation）      — 政策名称、条款编号是否与知识库一致
  3. 幻觉控制（Hallucination）   — 是否包含检索资料中不存在的内容
  4. NOHIT 阻断率（Blocking）    — out-of-domain 问题是否正确拒答
  5. 响应时间（Latency）         — 端到端延迟

评测方法：
  - in-domain / trap：LLM-as-Judge 三维打分（1-3）+ 期望关键词命中
  - out-of-domain：      响应内容模式匹配，检测 NOHIT 阻断

输入：tax law documents/test_questions.jsonl（250 条三级分类用例，支持 --sample 抽样）
输出：终端表格 + val_results.json（详细逐题结果）+ val_report.md（Markdown 报告）
"""

import json
import time
import sys
from pathlib import Path
from collections import defaultdict

# Windows 中文终端 GBK 编码 → UTF-8
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════════
# 0. 动态导入 rag_agent（会触发模块初始化：加载模型、向量库、LLM）
# ══════════════════════════════════════════════════════════════════
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("正在初始化 RAG 系统（加载 Embedding 模型 + 向量库 + LLM）...")
import rag_agent  # noqa: E402

chain = rag_agent.rag_chain

# ── 配置 ──────────────────────────────────────────────────────────
TEST_FILE = ROOT / "tax law documents" / "test_questions.jsonl"
OUTPUT_FILE = ROOT / "val_results.json"
JUDGE_MODEL_TEMP = 0.0   # 评委模型温度

# ── 评测阈值 ──────────────────────────────────────────────────────
KEYWORD_MATCH_MIN = 1               # 至少命中 1 个期望关键词
JUDGE_PASS_MIN_RELEVANCE = 2        # 评委相关性至少 2 分才算通过（1=答非所问/是非错误）
# NOHIT 特征：必须是系统级阻断（固定话术），不是 LLM 自然语言拒答
NOHIT_SIGNATURES = [
    "当前知识库未覆盖该具体政策",   # NOHIT_RESPONSE 独有开头
    "转交人工税务专家进行复核确认",  # NOHIT_RESPONSE 独有结尾
    "此提问不属于我所收录的中国税收政策范畴",  # OOD_NOHIT_RESPONSE（规则安检阻断）
    "无法基于现有境内财税基准文件作答",        # OOD_ML_NOHIT_RESPONSE（ML 安检阻断）
]
# out-of-domain 扩展：LLM 正确拒答特征
REFUSAL_PATTERNS = [
    "当前资料不足以完整回答",
    "资料不足以",
    "未包含.*相关信息",
    "未收录",
    "知识库中未包含",
    "资料不足，未收录该部分内容",   # 新实体边界防火墙触发语
    "超出中国国内税收基准文件范围",  # domain_guard 境外管辖拒答
    "非税法律范畴",                 # domain_guard 跨法域拒答
]


# ══════════════════════════════════════════════════════════════════
# 1. 加载测试用例
# ══════════════════════════════════════════════════════════════════
def load_test_cases(path: Path) -> list[dict]:
    """从 JSONL 或 JSON 数组加载测试用例"""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().strip()
    if raw.startswith("["):
        # JSON 数组格式：[{...}, {...}]
        cases = json.loads(raw)
    else:
        # JSONL 格式：每行一个独立 JSON 对象
        cases = []
        for line in raw.split("\n"):
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


# ══════════════════════════════════════════════════════════════════
# 2. 关键词命中检测（快速基准）
# ══════════════════════════════════════════════════════════════════
def keyword_match(answer: str, expected_keywords: list[str]) -> dict:
    """检测回答中命中了多少期望关键词"""
    hits = []
    misses = []
    for kw in expected_keywords:
        if kw.lower() in answer.lower():
            hits.append(kw)
        else:
            misses.append(kw)
    return {
        "hits": hits,
        "misses": misses,
        "hit_count": len(hits),
        "total": len(expected_keywords),
        "pass": len(hits) >= KEYWORD_MATCH_MIN if expected_keywords else True,
    }


# ══════════════════════════════════════════════════════════════════
# 3. NOHIT 阻断与拒答检测
# ══════════════════════════════════════════════════════════════════
import re as _re

# LOW 置信度路由的强制前言（拒答判定前先剥离）
_LOW_CONF_PREFACE = "⚠️ 检索到的政策匹配度较低，以下内容仅供参考，建议人工复核。"

def detect_nohit(answer: str) -> bool:
    """检测回答是否为系统级 NOHIT 阻断（固定话术，非 LLM 自然拒答）"""
    return any(pattern in answer for pattern in NOHIT_SIGNATURES)


def detect_refusal(answer: str) -> bool:
    """检测回答是否（正确地）拒答/声明知识库不足。

    口径（2026-07-17 收紧）：拒答话术必然出现在回答开头（阻断固定话术或
    降级拒答格式）。以"【"开头的4模块回答属于实质作答——其【具体内容】中
    "知识库未收录XXX"的部分覆盖声明不计入拒答，避免拒答率虚高。
    """
    if detect_nohit(answer):
        return True
    body = answer.strip()
    # 剥离 LOW 置信度强制前言后再判定开头
    if body.startswith(_LOW_CONF_PREFACE):
        body = body[len(_LOW_CONF_PREFACE):].strip()
    if body.startswith("【"):
        return False  # 4模块实质作答（部分覆盖式声明不算拒答）
    return any(_re.search(p, body) for p in REFUSAL_PATTERNS)


# ══════════════════════════════════════════════════════════════════
# 4. LLM-as-Judge 评分 Prompt
# ══════════════════════════════════════════════════════════════════
JUDGE_SYSTEM_PROMPT = """\
你是一位资深税务问答质量评估专家。请根据以下信息，严格评估一个 RAG 系统的回答质量。

## 0. 是非判断题特别规则（优先级最高，先于其他三个维度执行）

很多税务咨询问题是"需不需要/能不能/是不是/要不要/有没有/免不免/用不用交/要不要交/可以……吗"等形式的是非判断题。
对于这类问题，你必须在评估其他维度之前，首先判断系统回答的是非结论是否正确：

1. 识别用户问题的提问方向（在问"需要"还是"不需要"、"能"还是"不能"、"可以"还是"不可以"等）
2. 从系统回答的【核心结论】中提取其明确的是非立场
3. 对照【参考资料】的规定以及【期望关键词】中的是非指示词，判断该立场是否正确
4. **若是非结论明显错误**（如法律明确规定"免征"但系统答"需要缴纳"，或应当答"不可以"却答"可以"），则**相关性(relevance)直接评为1分**，并在notes中标注"❌是非判断错误：应[正确立场]，答[错误立场]"

注意：开放式/解释类/列举类问题（如"税率是多少""怎么计算""有哪些条件""如何申报"等）不受此规则影响。

## 评估维度（1-3 分制）

### 1. 相关性（relevance）
- 3分：回答完全切题，准确回应用户问题的所有核心要素
- 2分：回答部分切题，但遗漏了关键信息或答非所问的部分
- 1分：回答基本不相关，或完全未回应用户问题（是非判断题答错方向自动归为此档）

### 2. 引用准确性（citation）
- 3分：所有政策引用（名称、文号、条款）均与【参考资料】一致，无编造
- 2分：大部分引用正确，但有 1 处引用缺失或不精确
- 1分：引用错误较多，或引用了参考资料中不存在的政策名称/条款号

### 3. 幻觉控制（hallucination）
- 3分：回答中的所有事实（数字、比例、日期、条文）均在【参考资料】中能找到依据
- 2分：大部分内容有据可查，但存在 1-2 处参考资料未覆盖的推论性表述
- 1分：包含明显编造的数字、条文或政策——这些内容在【参考资料】中完全不存在

**🔴 阅读理解失败 ≠ 幻觉**（重要区分）：
- 若系统回答声称"资料不足/未提供/未包含"某信息，但实际上【参考资料】中**明确包含**该信息 → 这是**阅读理解失败**（LLM 没读到），归入**相关性(relevance)**维度扣分，**不得**因此降低 hallucination 分数
- 仅当系统回答**主动编造**了参考资料中不存在的具体数字、百分比、日期、法条编号或政策名称时，才降低 hallucination 分数
- 示例："资料未提供具体比例"（但资料里有80%）→ relevance 扣分，hallucination 不扣分
- 示例："减按90%计入收入总额"（但资料只写了"减计收入"没写比例）→ hallucination 扣分

## 输出格式
请**只**返回一行 JSON，不要加任何解释、markdown 代码块标记或额外文字：
{"relevance": <1|2|3>, "citation": <1|2|3>, "hallucination": <1|2|3>, "notes": "<一句话说明>"}
"""


def build_judge_messages(question: str, answer: str, context: str,
                         expected_kw: list[str]) -> list:
    """构造评委消息"""
    ek = "、".join(expected_kw) if expected_kw else "（无）"

    user_msg = f"""【用户问题】
{question}

【检索到的参考资料】
{context}

【系统回答】
{answer}

【期望关键词（仅供参考，不作为唯一评判标准）】
{ek}

请对以上回答进行三维评估，返回 JSON。"""

    return [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", user_msg),
    ]


def judge_answer(question: str, answer: str, context: str,
                 expected_keywords: list[str]) -> dict:
    """使用 LLM 评委评估单个回答的质量"""

    from langchain_core.messages import SystemMessage, HumanMessage

    ek = "、".join(expected_keywords) if expected_keywords else "（无）"
    _today = time.strftime("%Y年%m月%d日")
    user_text = f"""【当前日期】
{_today}（重要：政策条文含分期适用期间时——如"2024至2025年免征、2026至2027年减半"——必须以当前日期判断哪一档适用。系统回答按当前日期选择正确档位的，是非判断应视为正确，不得以已过期档位为准判错。）

【用户问题】
{question}

【检索到的参考资料】
{context}

【系统回答】
{answer}

【期望关键词（仅供参考，不作为唯一评判标准）】
{ek}

请对以上回答进行三维评估，返回 JSON。"""

    messages = [
        SystemMessage(content=JUDGE_SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]

    try:
        from langchain_core.output_parsers import StrOutputParser
        response = chain.llm.invoke(messages)
        raw = StrOutputParser().invoke(response).strip()

        # 清理 markdown 代码块标记
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        if raw.startswith("```json"):
            raw = raw[7:].strip()
            if raw.endswith("```"):
                raw = raw[:-3].strip()

        result = json.loads(raw)
        return {
            "relevance": int(result.get("relevance", 1)),
            "citation": int(result.get("citation", 1)),
            "hallucination": int(result.get("hallucination", 1)),
            "notes": result.get("notes", ""),
        }
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"    ⚠️ 评委 JSON 解析失败: {e}")
        print(f"    原始返回: {raw[:120] if 'raw' in dir() else 'N/A'}")
        return {"relevance": 1, "citation": 1, "hallucination": 1, "notes": f"PARSE_ERROR: {e}"}
    except Exception as e:
        print(f"    ⚠️ 评委调用异常: {e}")
        return {"relevance": 1, "citation": 1, "hallucination": 1, "notes": f"ERROR: {e}"}


# ══════════════════════════════════════════════════════════════════
# 5. 单题评测流水线
# ══════════════════════════════════════════════════════════════════
def evaluate_one(case: dict, index: int, total: int) -> dict:
    """评测单条用例，返回完整的评测记录"""
    qid = case["id"]
    qtype = case["type"]
    category = case["category"]
    question = case["question"]
    expected_kw = case.get("expected_keywords", [])

    print(f"[{index}/{total}] {qid} [{qtype}] {question[:50]}...")

    # ── 1. 检索（一次检索，同时供评委 context 和 stream 路由使用）─
    docs, distances = chain._hybrid_search(question, k=5)
    context = rag_agent.format_docs(docs)
    best_distance = min(distances) if distances else float("inf")

    # ── 2. 流式生成 + 计时（TTFT + 总耗时）────────────────────
    #      TTFT 由 _stream_raw 内部精确记录（HTTP POST → 首个 content delta），
    #      不含 debug print / format_docs / format_messages 等应用层开销
    answer_parts = []
    try:
        for token in chain.stream(question, pre_docs=docs, pre_distances=distances):
            answer_parts.append(token)
        answer = "".join(answer_parts)
    except Exception as e:
        answer = f"[ERROR] {e}"
    # 读取 _stream_raw 内部测量的纯网络级指标
    ttft = getattr(chain, "_http_ttft", 0.0)
    total_time = getattr(chain, "_http_total", 0.0)

    # ── 3. NOHIT / 拒答检测 ──────────────────────────────────
    is_nohit = detect_nohit(answer)           # 系统级阻断
    is_refusal = detect_refusal(answer)       # 系统阻断或 LLM 正确拒答

    # ── 4. 关键词命中 ──────────────────────────────────────────
    kw_result = keyword_match(answer, expected_kw)

    # ── 5. LLM 评委评分（out-of-domain 仅用拒答检测，不请评委）──
    judge_scores = None
    if qtype != "out-of-domain":
        print(f"    🤖 评委评分中...")
        judge_scores = judge_answer(question, answer, context, expected_kw)
    else:
        # out-of-domain：正确拒答则满分，错误作答则零分
        if is_refusal:
            judge_scores = {
                "relevance": 3,
                "citation": 3,
                "hallucination": 3,
                "notes": "NOHIT阻断" if is_nohit else "LLM正确拒答",
            }
        else:
            judge_scores = {
                "relevance": 1,
                "citation": 1,
                "hallucination": 1,
                "notes": "应拒答但未拒答",
            }

    # ── 6. 拼装结果 ────────────────────────────────────────────
    # 综合判定：out-of-domain 看拒答，其他看关键词+评委双重校验
    if qtype == "out-of-domain":
        judge_pass = is_refusal      # OOD：正确拒答=通过
        overall_pass = is_refusal
    else:
        judge_pass = (judge_scores["relevance"] >= JUDGE_PASS_MIN_RELEVANCE) if judge_scores else True
        overall_pass = kw_result["pass"] and judge_pass

    record = {
        "id": qid,
        "type": qtype,
        "category": category,
        "question": question,
        "answer": answer[:500],
        "expected_keywords": expected_kw,
        "best_distance": round(best_distance, 4),
        "is_nohit": is_nohit,
        "is_refusal": is_refusal,
        "ttft": round(ttft, 3),           # 首字延迟（Time To First Token）
        "total_time": round(total_time, 3),  # 端到端总耗时
        "keyword_hits": kw_result["hits"],
        "keyword_misses": kw_result["misses"],
        "keyword_pass": kw_result["pass"],
        "judge_pass": judge_pass,          # 评委判定（相关性>=2）
        "overall_pass": overall_pass,      # 综合判定（关键词+评委）
        "judge": judge_scores,
    }

    # 打印单题摘要
    if qtype == "out-of-domain":
        prefix = "🛡️" if overall_pass else "🚨"
    elif overall_pass:
        prefix = "✅"
    else:
        prefix = "❌"
    kw_str = "、".join(kw_result["hits"][:3])
    if judge_scores:
        print(f"    {prefix} TTFT={ttft:.2f}s total={total_time:.2f}s dist={best_distance:.4f} "
              f"命中=[{kw_str}] 评委=相关性{judge_scores['relevance']}/"
              f"引用{judge_scores['citation']}/幻觉{judge_scores['hallucination']}")
    else:
        print(f"    {prefix} TTFT={ttft:.2f}s total={total_time:.2f}s dist={best_distance:.4f} "
              f"命中=[{kw_str}] 评委=N/A")

    return record


# ══════════════════════════════════════════════════════════════════
# 6. 汇总统计
# ══════════════════════════════════════════════════════════════════
def compute_summary(results: list[dict]) -> dict:
    """从逐题结果计算汇总指标"""
    groups = defaultdict(list)
    for r in results:
        groups[r["type"]].append(r)

    summary = {}
    for gtype, group in groups.items():
        n = len(group)
        times = [r["total_time"] for r in group]
        times_sorted = sorted(times)
        p50 = times_sorted[int(n * 0.5)] if n > 0 else 0
        p95 = times_sorted[min(int(n * 0.95), n - 1)] if n > 0 else 0

        # TTFT 指标
        ttfts = [r["ttft"] for r in group]
        ttfts_sorted = sorted(ttfts)

        item = {
            "count": n,
            "avg_ttft": round(sum(ttfts) / n, 3) if n else 0,
            "p95_ttft": round(ttfts_sorted[min(int(n * 0.95), n - 1)], 3) if n else 0,
            "avg_total_time": round(sum(times) / n, 2) if n else 0,
            "p50_total_time": round(p50, 2),
            "p95_total_time": round(p95, 2),
        }

        if gtype == "out-of-domain":
            # 阻断率（含 NOHIT 阻断 + LLM 正确拒答）
            blocked = sum(1 for r in group if r["is_refusal"])
            item["nohit_block_rate"] = round(blocked / n * 100, 1) if n else 0
            item["avg_relevance"] = None
            item["avg_citation"] = None
            item["avg_hallucination"] = None
            item["hallucination_rate"] = None
        else:
            # LLM 评委指标
            judged = [r for r in group if r.get("judge")]
            jn = len(judged)
            if jn > 0:
                item["avg_relevance"] = round(sum(j["judge"]["relevance"] for j in judged) / jn, 2)
                item["avg_citation"] = round(sum(j["judge"]["citation"] for j in judged) / jn, 2)
                item["avg_hallucination"] = round(sum(j["judge"]["hallucination"] for j in judged) / jn, 2)
                # 幻觉率：hallucination < 3 即存在瑕疵
                hallucinated = sum(1 for j in judged if j["judge"]["hallucination"] < 3)
                item["hallucination_rate"] = round(hallucinated / jn * 100, 1)
            else:
                item["avg_relevance"] = None
                item["avg_citation"] = None
                item["avg_hallucination"] = None
                item["hallucination_rate"] = None
            item["nohit_block_rate"] = None

        # 关键词基准
        kw_pass = sum(1 for r in group if r.get("keyword_pass"))
        item["keyword_pass_rate"] = round(kw_pass / n * 100, 1) if n else 0

        # 综合通过率（关键词 + 评委双重校验）
        overall_pass_count = sum(1 for r in group if r.get("overall_pass"))
        item["overall_pass_rate"] = round(overall_pass_count / n * 100, 1) if n else 0

        summary[gtype] = item

    # 全量汇总
    all_times = [r["total_time"] for r in results]
    all_times_sorted = sorted(all_times)
    all_ttfts = [r["ttft"] for r in results]
    all_ttfts_sorted = sorted(all_ttfts)
    total_n = len(results)
    # 质量指标仅统计 in-domain + trap（排除 OOD，OOD 的 judge 分数是人工赋值的）
    all_judged = [r for r in results if r.get("judge") and r["type"] != "out-of-domain"]
    ajn = len(all_judged)
    all_ood = [r for r in results if r["type"] == "out-of-domain"]

    # 幻觉率拆分：区分真实幻觉 vs 拒答
    # - 真实幻觉：LLM 回答了但编造了参考资料中不存在的内容 (H<3 且非拒答)
    # - 拒答：LLM 声明资料不足/无法回答 (含正确拒答和错误拒答)
    true_hallucination = [r for r in all_judged
                          if r["judge"]["hallucination"] < 3 and not r.get("is_refusal")]
    refusal_cases = [r for r in all_judged if r.get("is_refusal")]
    # 错误拒答：in-domain/trap 题目中 LLM 错误地拒答了
    wrong_refusal = [r for r in refusal_cases
                     if r["judge"]["relevance"] < 2]

    summary["total"] = {
        "count": total_n,
        "avg_ttft": round(sum(all_ttfts) / total_n, 3),
        "p95_ttft": round(all_ttfts_sorted[min(int(total_n * 0.95), total_n - 1)], 3),
        "avg_total_time": round(sum(all_times) / total_n, 2),
        "p50_total_time": round(all_times_sorted[int(total_n * 0.5)], 2),
        "p95_total_time": round(all_times_sorted[min(int(total_n * 0.95), total_n - 1)], 2),
        # 质量指标（仅 in-domain + trap）
        "avg_relevance": round(sum(r["judge"]["relevance"] for r in all_judged) / ajn, 2) if ajn else None,
        "avg_citation": round(sum(r["judge"]["citation"] for r in all_judged) / ajn, 2) if ajn else None,
        "avg_hallucination": round(sum(r["judge"]["hallucination"] for r in all_judged) / ajn, 2) if ajn else None,
        # 旧幻觉率（向后兼容，含拒答）— 保留但降级展示
        "hallucination_rate": round(sum(1 for r in all_judged if r["judge"]["hallucination"] < 3) / ajn * 100, 1) if ajn else None,
        # 新：真实幻觉率（排除拒答题，仅统计 LLM 确实编造了内容的情况）
        "true_hallucination_rate": round(len(true_hallucination) / ajn * 100, 1) if ajn else None,
        # 新：拒答率（in-domain+trap 中系统拒答的比例）
        "refusal_rate": round(len(refusal_cases) / ajn * 100, 1) if ajn else None,
        # 新：错误拒答率（拒答但应该回答的比例）
        "wrong_refusal_rate": round(len(wrong_refusal) / ajn * 100, 1) if ajn else None,
        "nohit_block_rate": round(sum(1 for r in all_ood if r["is_refusal"]) / len(all_ood) * 100, 1) if all_ood else None,
        "keyword_pass_rate": round(sum(1 for r in results if r.get("keyword_pass")) / total_n * 100, 1),
        "overall_pass_rate": round(sum(1 for r in results if r.get("overall_pass")) / total_n * 100, 1),
    }

    return summary


# ══════════════════════════════════════════════════════════════════
# 7. 输出表格（Markdown 文件格式）
# ══════════════════════════════════════════════════════════════════

REPORT_FILE = ROOT / "val_report.md"

def _md_escape(text: str) -> str:
    """转义 Markdown 表格中可能破坏格式的字符"""
    return str(text).replace("|", "\\|").replace("\n", " ")


def write_report(results: list[dict], summary: dict, output_path: Path):
    """将评测结果写入 Markdown 表格文件，便于直接粘贴到文档中"""

    lines: list[str] = []

    def w(line: str = ""):
        lines.append(line)

    w("# 政策问答系统离线评测结果")
    w()
    w(f"**评测时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"**总题数**: {len(results)}")
    w()

    # ── 7a. 按题目类型汇总 ──────────────────────────────────────
    w("## 按题目类型汇总")
    w()
    w("| 类型 | 数量 | 相关性(avg) | 引用准确率(avg) | 幻觉率 | NOHIT阻断率 | 关键词命中率 | 综合通过率 | 平均TTFT | P95_TTFT | 平均总耗时 |")
    w("|------|------|-------------|-----------------|--------|-------------|--------------|------------|----------|----------|------------|")

    type_order = ["in-domain", "trap", "out-of-domain"]
    for gtype in type_order:
        if gtype not in summary:
            continue
        s = summary[gtype]
        rel = f"{s['avg_relevance']:.2f}" if s['avg_relevance'] is not None else "-"
        cit = f"{s['avg_citation']:.2f}" if s['avg_citation'] is not None else "-"
        hal = f"{s['hallucination_rate']:.1f}%" if s['hallucination_rate'] is not None else "-"
        blk = f"{s['nohit_block_rate']:.1f}%" if s['nohit_block_rate'] is not None else "-"
        kw = f"{s['keyword_pass_rate']:.1f}%"
        ov = f"{s['overall_pass_rate']:.1f}%"
        avg_ttft = f"{s['avg_ttft']:.3f}s"
        p95_ttft = f"{s['p95_ttft']:.3f}s"
        avg_t = f"{s['avg_total_time']:.2f}s"

        w(f"| {gtype} | {s['count']} | {rel} | {cit} | {hal} | {blk} | {kw} | {ov} | {avg_ttft} | {p95_ttft} | {avg_t} |")

    # 总计行（质量指标仅统计 in-domain + trap，不含 OOD）
    t = summary["total"]
    rel = f"{t['avg_relevance']:.2f}" if t['avg_relevance'] is not None else "-"
    cit = f"{t['avg_citation']:.2f}" if t['avg_citation'] is not None else "-"
    hal = f"{t['hallucination_rate']:.1f}%" if t['hallucination_rate'] is not None else "-"
    true_hal = f"{t['true_hallucination_rate']:.1f}%" if t.get('true_hallucination_rate') is not None else "-"
    blk = f"{t['nohit_block_rate']:.1f}%" if t['nohit_block_rate'] is not None else "-"
    kw = f"{t['keyword_pass_rate']:.1f}%"
    ov = f"{t['overall_pass_rate']:.1f}%"
    avg_ttft = f"{t['avg_ttft']:.3f}s"
    p95_ttft = f"{t['p95_ttft']:.3f}s"
    avg_t = f"{t['avg_total_time']:.2f}s"
    w(f"| **总计** | **{t['count']}** | **{rel}** | **{cit}** | **{hal}** | **{blk}** | **{kw}** | **{ov}** | **{avg_ttft}** | **{p95_ttft}** | **{avg_t}** |")
    w()

    # ── 7a2. 幻觉率拆分（核心诊断指标）──────────────────────────
    w("### 幻觉率拆分（in-domain + trap 仅计）")
    w()
    w("| 指标 | 值 | 说明 |")
    w("|------|-----|------|")
    w(f"| 旧幻觉率 (H<3全量) | {hal} | 含拒答、含 OOD 误归类，**已弃用** |")
    w(f"| **真实幻觉率** (排除拒答) | **{true_hal}** | **LLM 编造了参考资料中不存在的内容** ← 核心指标 |")
    ref_rate = f"{t['refusal_rate']:.1f}%" if t.get('refusal_rate') is not None else "-"
    wrong_ref_rate = f"{t['wrong_refusal_rate']:.1f}%" if t.get('wrong_refusal_rate') is not None else "-"
    w(f"| 拒答率 | {ref_rate} | in-domain+trap 中系统声明\"资料不足\"的比例 |")
    w(f"| 错误拒答率 | {wrong_ref_rate} | 拒答但应该回答的比例（检索失败导致） |")
    w()

    # ── 7b. 按类别汇总 ──────────────────────────────────────────
    w("## 按类别汇总")
    w()
    w("| 类别 | 数量 | 类型 | 相关性(avg) | 引用准确率(avg) | 幻觉率 | 关键词命中率 |")
    w("|------|------|------|-------------|-----------------|--------|--------------|")

    cats = defaultdict(list)
    for r in results:
        cats[(r["category"], r["type"])].append(r)

    for (cat, ctype), group in sorted(cats.items()):
        n = len(group)
        judged = [r for r in group if r.get("judge")]
        jn = len(judged)

        if ctype == "out-of-domain":
            rel_s = "-"
            cit_s = "-"
            hal_s = "-"
        elif jn > 0:
            rel_s = f"{sum(j['judge']['relevance'] for j in judged) / jn:.2f}"
            cit_s = f"{sum(j['judge']['citation'] for j in judged) / jn:.2f}"
            h_count = sum(1 for j in judged if j["judge"]["hallucination"] < 3)
            hal_s = f"{h_count / jn * 100:.1f}%"
        else:
            rel_s = "-"
            cit_s = "-"
            hal_s = "-"

        kw_pass = sum(1 for r in group if r.get("keyword_pass"))
        kw_s = f"{kw_pass / n * 100:.1f}%" if n else "-"

        w(f"| {_md_escape(cat)} | {n} | {ctype} | {rel_s} | {cit_s} | {hal_s} | {kw_s} |")
    w()

    # ── 7c. 逐题明细 ────────────────────────────────────────────
    w("## 逐题明细")
    w()
    w("| ID | 类型 | 类别 | TTFT(s) | 总耗时(s) | Best Dist | NOHIT | 综合判定 | R | C | H | 问题摘要 |")
    w("|----|------|------|---------|-----------|-----------|-------|----------|---|---|---|----------|")

    for r in results:
        dist = f"{r['best_distance']:.4f}" if r['best_distance'] != float("inf") else "-"
        nohit = "Y" if r['is_nohit'] else "-"
        overall = "PASS" if r.get("overall_pass") else "FAIL"
        j = r.get("judge") or {}
        rel = str(j.get("relevance", "-"))
        cit = str(j.get("citation", "-"))
        hal = str(j.get("hallucination", "-"))
        q = _md_escape(r['question'][:60])

        w(f"| {r['id']} | {r['type']} | {_md_escape(r['category'])} | "
          f"{r['ttft']:.2f} | {r['total_time']:.2f} | {dist} | {nohit} | {overall} | "
          f"{rel} | {cit} | {hal} | {q} |")
    w()

    # ── 7d. 指标说明 ────────────────────────────────────────────
    w("## 指标说明")
    w()
    w("- **相关性 (R) / 引用准确率 (C) / 幻觉控制 (H)**: LLM 评委 1-3 分制，3=最优")
    w("- **旧幻觉率**: 评委判定存在瑕疵（hallucination < 3）的比例（含拒答、OOD误归类），**已弃用**")
    w("- **真实幻觉率**: 排除拒答后，LLM 确实编造了参考资料中不存在的内容的比例（**核心指标**）")
    w("- **拒答率**: in-domain/trap 中系统声明\"资料不足\"的比例（过高说明检索或路由有问题）")
    w("- **错误拒答率**: 拒答但评委判定应该回答的比例（检索到资料但 LLM 没读到）")
    w("- **NOHIT阻断率**: out-of-domain 问题被正确拒绝的比例")
    w("- **关键词命中率**: 期望关键词至少 1 个出现在回答中的比例（快速基准）")
    w("- **综合通过率**: 关键词 + 评委相关性(relevance≥2) 双重校验后的通过率（核心指标）")
    w("- **综合判定**: PASS=关键词命中且评委相关性≥2；FAIL=任一条件不满足")
    w("- **TTFT** (Time To First Token): 首个非空 token 返回前的耗时（首字延迟）")
    w()

    # ── 写入文件 ────────────────────────────────────────────────
    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n评测报告已保存至: {output_path}")

    # 仍然在终端打印一份简要摘要
    print(f"\n{'='*60}")
    print(f"评测完成 — 简要摘要")
    print(f"{'='*60}")
    t = summary["total"]
    print(f"总题数: {t['count']}  |  综合通过率: {t['overall_pass_rate']:.1f}%  |  关键词命中率: {t['keyword_pass_rate']:.1f}%")
    if t['avg_relevance'] is not None:
        print(f"LLM评委: 相关性={t['avg_relevance']:.2f} 引用={t['avg_citation']:.2f} 幻觉={t['avg_hallucination']:.2f}")
    true_hal = t.get('true_hallucination_rate')
    ref_rate = t.get('refusal_rate')
    wrong_ref = t.get('wrong_refusal_rate')
    if true_hal is not None:
        print(f"真实幻觉率: {true_hal:.1f}%  |  拒答率: {ref_rate:.1f}%  |  错误拒答率: {wrong_ref:.1f}%")
    print(f"平均TTFT: {t['avg_ttft']:.3f}s  |  P95_TTFT: {t['p95_ttft']:.3f}s  |  平均总耗时: {t['avg_total_time']:.2f}s")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════════
# 8. 主入口
# ══════════════════════════════════════════════════════════════════
def main():
    import argparse
    import random

    parser = argparse.ArgumentParser(description="政策问答离线评测")
    parser.add_argument("--sample", type=int, default=0,
                        help="随机抽样 N 条（按题型比例），0=全量")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子（默认 42，保证可复现）")
    parser.add_argument("--ids", type=str, default="",
                        help="指定题目 ID（逗号分隔），如 'q01,q03,q143'。与 --sample 互斥")
    parser.add_argument("--resume", action="store_true",
                        help="断点续跑：跳过 val_results.json 中已完成的题目")
    args = parser.parse_args()

    # ── 加载用例 ──────────────────────────────────────────────────
    if not TEST_FILE.exists():
        print(f"[ERROR] 测试文件不存在: {TEST_FILE}")
        return

    cases = load_test_cases(TEST_FILE)
    total = len(cases)
    type_counts = defaultdict(int)
    for c in cases:
        type_counts[c["type"]] += 1

    # ── 指定 ID 过滤 ────────────────────────────────────────────
    if args.ids:
        target_ids = set(i.strip() for i in args.ids.split(",") if i.strip())
        missing = target_ids - set(c["id"] for c in cases)
        if missing:
            print(f"⚠️ 以下 ID 不存在: {', '.join(sorted(missing))}")
        cases = [c for c in cases if c["id"] in target_ids]
        if not cases:
            print("[ERROR] 没有匹配的题目")
            return
        type_counts = defaultdict(int)
        for c in cases:
            type_counts[c["type"]] += 1

    # ── 抽样 ────────────────────────────────────────────────────
    elif args.sample > 0 and args.sample < total:
        random.seed(args.seed)
        # 按题型比例分层抽样
        sampled: list[dict] = []
        for qtype in ["in-domain", "trap", "out-of-domain"]:
            pool = [c for c in cases if c["type"] == qtype]
            n = max(1, round(args.sample * len(pool) / total))  # 至少 1 条
            n = min(n, len(pool))
            chosen = random.sample(pool, n)
            sampled.extend(chosen)
        # 如果因取整导致数量偏差，从最大池补或裁
        while len(sampled) < args.sample:
            pool = [c for c in cases if c not in sampled]
            sampled.append(random.choice(pool))
        cases = sampled[:args.sample]
        random.shuffle(cases)
        # 重新统计
        type_counts = defaultdict(int)
        for c in cases:
            type_counts[c["type"]] += 1

    total = len(cases)
    print(f"\n加载 {total} 条测试用例: in-domain={type_counts.get('in-domain',0)}, "
          f"trap={type_counts.get('trap',0)}, out-of-domain={type_counts.get('out-of-domain',0)}")
    if args.sample > 0:
        print(f"(从全量 {len(load_test_cases(TEST_FILE))} 条中随机抽样, seed={args.seed})")

    # ── 断点续跑 ──────────────────────────────────────────────────
    existing_results = {}  # id → record
    if args.resume and OUTPUT_FILE.exists():
        try:
            prev = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            for r in prev.get("results", []):
                existing_results[r["id"]] = r
            print(f"📂 从 {OUTPUT_FILE.name} 恢复，已完成 {len(existing_results)} 题")
        except Exception:
            pass

    # ── 逐题评测（每题后增量保存）───────────────────────────────────
    for i, case in enumerate(cases, 1):
        if case["id"] in existing_results:
            continue
        record = evaluate_one(case, i, total)
        existing_results[case["id"]] = record
        # 即时写入 JSON，断了也不丢
        try:
            all_records = list(existing_results.values())
            summary = compute_summary(all_records)
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "results": all_records}, f,
                          ensure_ascii=False, indent=2)
        except Exception:
            pass

    # ── 最终汇总 & 报告 ─────────────────────────────────────────────
    all_records = list(existing_results.values())
    summary = compute_summary(all_records)
    write_report(all_records, summary, REPORT_FILE)
    print(f"详细结果(JSON)已保存至: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
