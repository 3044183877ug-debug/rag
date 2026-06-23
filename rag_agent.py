"""
rag_agent.py
------------
RAG 税务问答 Agent：
1. 从本地 Chroma 向量库加载 Retriever
2. 对接无极（Infini-AI）平台的大模型 API
3. 构建检索增强生成（RAG）工作流
4. 提供控制台交互式问答
"""

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

# 加载 .env 文件中的环境变量
load_dotenv()
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
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

# 将向量库转换为 Retriever（检索器）
# k 表示每次检索返回的最相关文档片段数
retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# ── 3. 配置大模型（无极 Infini-AI 平台）─────────────────────────
api_key = os.environ.get("INFINI_API_KEY")
if not api_key:
    raise ValueError("❌ 未找到环境变量 INFINI_API_KEY，请先设置：$env:INFINI_API_KEY='your-key'")

llm = ChatOpenAI(
    base_url="https://cloud.infini-ai.com/maas/v1",
    model="deepseek-v4-pro",
    api_key=api_key,
    temperature=0.3,  # 税务场景需要较低温度以保证准确性
)

# ── 4. 构建 RAG 工作流 ──────────────────────────────────────────
# 系统提示词：设定为专业的税务研究员
system_prompt = (
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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# 创建 RAG 链：使用新的 LangChain v1.x 方式
def format_docs(docs):
    """将检索到的文档拼接，并附带来源标注"""
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        article = doc.metadata.get("article", "")
        chapter = doc.metadata.get("chapter", "")
        # 构建来源标签，如 "《增值税法》第十条 | 第二章 税率"
        if article:
            label = f"《{source}》{article}"
        else:
            label = f"《{source}》"
        if chapter:
            label += f" | {chapter}"
        formatted.append(f"[来源: {label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG 工作流构建完成，可以开始提问！\n")

# ── 5. 交互式问答循环 ──────────────────────────────────────────
print("=" * 60)
print("  税务 RAG 问答系统")
print("  输入 'quit' 或 'exit' 退出")
print("=" * 60)
print()

while True:
    user_input = input("👤 请输入您的问题: ").strip()

    if user_input.lower() in ("quit", "exit", "q"):
        print("👋 再见！")
        break

    if not user_input:
        continue

    print("\n🤖 正在检索并生成回答...\n")

    try:
        # 在新版本中，直接返回答案字符串
        answer = rag_chain.invoke(user_input)
        print(f"📋 回答:\n{answer}\n")

    except Exception as e:
        print(f"❌ 出错了: {e}\n")
