"""
rag_agent.py
------------
RAG 税务问答 Agent：
1. 从本地 Chroma 向量库加载 Retriever
2. 对接无极（Infini-AI）平台的大模型 API
3. 构建检索增强生成（RAG）工作流
4. 提供控制台交互式问答

自适应策略：
- 查询转换：用 LLM 将模糊问题改写为精准检索查询
- 多路检索：原始问题 + 改写查询各检索一次，合并去重
- 检索相似度 >= 0.5 → 走 RAG（基于资料库回答）
- 检索相似度 < 0.5 → 降级为 LLM 自身知识回答（标注参考资料不足）
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# 加载 .env 文件中的环境变量
load_dotenv()
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# ── 1. 加载 Embedding 模型 ──────────────────────────────────────
# 必须与 build_vector_db.py 使用相同的模型和参数
print("正在加载 Embedding 模型...")
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-zh-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("Embedding 模型加载完成")

# ── 2. 加载已有的 Chroma 向量库 ─────────────────────────────────
import os as _os

DB_DIR = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "chroma_db")

print(f"正在加载向量数据库: {DB_DIR}")
vectordb = Chroma(
    persist_directory=DB_DIR,
    embedding_function=embedding_model,
    collection_name="tax_knowledge",
)
print(f"向量数据库加载完成，共 {vectordb._collection.count()} 条向量")

# ── 3. 配置大模型（无极 Infini-AI 平台）─────────────────────────
api_key = os.environ.get("INFINI_API_KEY")
if not api_key:
    raise ValueError(
        "❌ 未找到环境变量 INFINI_API_KEY，请先设置：$env:INFINI_API_KEY='your-key'"
    )

llm = ChatOpenAI(
    base_url="https://cloud.infini-ai.com/maas/v1",
    model="deepseek-v4-pro",
    api_key=api_key,
    temperature=0.3,  # 税务场景需要较低温度以保证准确性
)

# ── 4. 构建自适应 RAG 工作流 ────────────────────────────────────

# 相似度阈值：低于此值视为不相关，降级为 LLM 自身回答
SCORE_THRESHOLD = 0.5

# ----- 提示词 A：RAG 模式（资料充分时） -----
rag_system_prompt = (
    "你是一位经验丰富的税务研究员，精通中国税法体系。\n\n"
    "你的回答需要遵循以下原则：\n"
    '1. 基于提供的【参考资料】给出准确回答，引用时注明出处（如"根据《增值税法》第十条"）；\n'
    "2. 每条参考资料末尾标注了来源法条，请务必在回答中引用；\n"
    "3. 如果参考资料不足以回答，请如实说明，并给出一般性建议；\n"
    "4. 用清晰、结构化的方式呈现答案，必要时使用分点列举；\n"
    '5. 在回答末尾可附上"温馨提示"，提醒用户政策可能随时更新，建议以税务机关最新公告为准。\n\n'
    "【参考资料】\n"
    "{context}"
)

# ----- 提示词 B：直接回答模式（资料不足时） -----
direct_system_prompt = (
    "你是一位经验丰富的税务研究员，精通中国税法体系。\n\n"
    "由于当前知识库中没有找到与问题直接相关的税法条文，"
    "请基于你自身的法律知识给出回答。\n\n"
    "回答时请严格遵循以下格式：\n"
    "1. 开头必须明确说明："
    "'⚠️ 当前知识库中未找到直接相关的税法条文，以下回答基于通用知识，仅供参考'\n"
    "2. 用清晰、结构化的方式呈现答案\n"
    "3. 如果涉及具体税率或法条，建议用户查阅官方最新文件\n"
    "4. 在回答末尾附上："
    "'温馨提示：以上回答基于通用知识，政策可能随时更新，建议以税务机关最新公告为准。'"
)

rag_prompt = ChatPromptTemplate.from_messages([
    ("system", rag_system_prompt),
    ("human", "{input}"),
])

direct_prompt = ChatPromptTemplate.from_messages([
    ("system", direct_system_prompt),
    ("human", "{input}"),
])


def format_docs(docs):
    """将检索到的文档拼接，并附带来源标注"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        article = doc.metadata.get("article", "")
        chapter = doc.metadata.get("chapter", "")
        if article:
            label = f"《{source}》{article}"
        else:
            label = f"《{source}》"
        if chapter:
            label += f" | {chapter}"
        formatted.append(f"[来源: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


class AdaptiveRAGChain:
    """自适应 RAG 链：查询转换 + 多路检索 + 质量判断"""

    def __init__(self, vectorstore, rag_prompt, direct_prompt, llm, threshold=0.5):
        self.vectorstore = vectorstore
        self.rag_prompt = rag_prompt
        self.direct_prompt = direct_prompt
        self.llm = llm
        self.threshold = threshold
        self.output_parser = StrOutputParser()

    # ── 查询转换提示词 ──────────────────────────────────────────
    QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个查询改写助手。用户会用日常语言提出税务问题，"
         "你的任务是把问题改写为更适合向量检索的查询语句。\n\n"
         "规则：\n"
         "1. 将指代不明的词替换为具体法律术语（如\"第三条\"→\"个人所得税法第三条\"、\"那个税\"→\"契税\"）\n"
         "2. 如果用户提到了\"某个法\"、\"某条\"但没有说是哪部法，基于知识库中已有的法律推断\n"
         "3. 如果问题本身已经很明确，直接返回原文\n"
         "4. 只输出改写后的查询，不要加任何解释\n\n"
         "知识库中已有的法律：\n"
         "- 中华人民共和国增值税法\n"
         "- 中华人民共和国契税法\n"
         "- 中华人民共和国企业所得税法\n"
         "- 中华人民共和国个人所得税法"),
        ("human", "原始问题：{question}\n\n改写后的查询："),
    ])

    def _rewrite_query(self, question: str) -> str:
        """用 LLM 将用户问题改写为更适合检索的查询"""
        try:
            messages = self.QUERY_REWRITE_PROMPT.format_messages(question=question)
            response = self.llm.invoke(messages)
            rewritten = self.output_parser.invoke(response).strip()
            # 防止 LLM 输出空或过长
            if rewritten and len(rewritten) <= 200:
                return rewritten
            return question
        except Exception:
            return question

    def _multi_search(self, question: str, k: int = 4):
        """多路检索：原始问题 + 改写查询，合并去重取最佳"""
        # 改写查询
        rewritten = self._rewrite_query(question)

        # 两路检索
        results_original = self.vectorstore.similarity_search_with_score(
            question, k=k
        )
        if rewritten != question:
            print(f"  🔍 查询转换: \"{question}\" -> \"{rewritten}\"")
            results_rewritten = self.vectorstore.similarity_search_with_score(
                rewritten, k=k
            )
        else:
            results_rewritten = []

        # 合并去重（按 page_content 去重，保留分数更低的）
        seen = {}
        for doc, score in results_original + results_rewritten:
            key = doc.page_content[:100]  # 用前100字符作为去重标识
            if key not in seen or score < seen[key][1]:
                seen[key] = (doc, score)

        # 按分数升序（分数越低相似度越高），取前 k 条
        merged = sorted(seen.values(), key=lambda x: x[1])[:k]
        return [d for d, _ in merged], [s for _, s in merged]

    def invoke(self, question: str) -> str:
        # 第一步：多路检索（原始问题 + 改写查询）
        docs, scores = self._multi_search(question, k=4)

        # 打印检索结果供调试
        print(f"  📊 检索相似度: {[round(s, 4) for s in scores]}")

        # 判断：至少有一条结果相似度 >= 阈值
        high_score = [s for s in scores if s >= self.threshold]
        if high_score:
            print(f"  ✅ {len(high_score)} 条结果达标（阈值 {self.threshold}），走 RAG 回答")
            context = format_docs(docs)
            messages = self.rag_prompt.format_messages(context=context, input=question)
            response = self.llm.invoke(messages)
            return self.output_parser.invoke(response)
        else:
            print(f"  ⚠️ 无结果达标（阈值 {self.threshold}），降级为 LLM 自身知识回答")
            messages = self.direct_prompt.format_messages(input=question)
            response = self.llm.invoke(messages)
            return self.output_parser.invoke(response)


rag_chain = AdaptiveRAGChain(
    vectorstore=vectordb,
    rag_prompt=rag_prompt,
    direct_prompt=direct_prompt,
    llm=llm,
    threshold=SCORE_THRESHOLD,
)

print("自适应 RAG 工作流构建完成，可以开始提问！")
print(f"  相似度阈值: {SCORE_THRESHOLD}（低于此值将降级为直接回答）\n")

# ── 5. 交互式问答循环 ──────────────────────────────────────────
print("=" * 60)
print("  税务 RAG 问答系统")
print("  输入 'quit' 或 'exit' 退出")
print("=" * 60)
print()

if __name__ == "__main__":
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
            print(f"[回答]:\n{answer}\n")
        except Exception as e:
            print(f"[错误]: {e}\n")