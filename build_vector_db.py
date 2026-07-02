"""
build_vector_db.py
------------------
YAML 驱动的税务政策向量化工具。

读取 data_source.yaml 作为"总索引"，遍历每个 policy entry，
定位 tax law documents/ 下的物理文件，加载全文后使用
RecursiveCharacterTextSplitter 切分，并将 YAML 中的全部元数据字段
注入每一个 Document 分块，最后存入 Chroma 向量数据库持久化。

支持的物理文件格式:
  - .txt   → 直接 UTF-8 读取
  - .pdf   → pymupdf (fitz) 逐页提取纯文本
  - .doc / .docx → (预留扩展点)
"""

import os
import re
import shutil
from pathlib import Path

import yaml
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ══════════════════════════════════════════════════════════════════
#  路径配置
# ══════════════════════════════════════════════════════════════════
BASE_DIR = Path(__file__).resolve().parent
YAML_PATH = BASE_DIR / "data_source.yaml"
DOCS_DIR = BASE_DIR / "tax law documents"
DB_DIR = BASE_DIR / "chroma_db"

# ══════════════════════════════════════════════════════════════════
#  切分配置
# ══════════════════════════════════════════════════════════════════
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
# ── 切分符设计原则 ──────────────────────────────────────────────
# 法律文本长句极多，按逗号/分号切分会导致"主谓分离"。
# 例如："契税的纳税义务发生时间，为签订合同的当日。" → 切在逗号处丢失主语。
#
# 优先级递减：
#   \n\n  → 段落边界（最安全）
#   。    → 句子边界（法律文本的天然语义单元）
#   \n    → 换行（PDF 重建后只剩真正的段落内换行）
#   ；    → 分句边界（仅作为最后防线，阻止 RecursiveCharacterTextSplitter
#           在无任何分隔符时回退到逐字符硬切；正常法条不会触发此级）
CHUNK_SEPARATORS = ["\n\n", "。", "\n", "；"]


# ══════════════════════════════════════════════════════════════════
#  文件加载器
# ══════════════════════════════════════════════════════════════════

def load_txt(file_path: Path) -> str:
    """加载 .txt 文件（UTF-8 编码）"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(file_path: Path) -> str:
    """加载 .pdf 文件，逐页提取纯文本并重建段落结构。

    pymupdf 的 get_text() 按视觉布局输出，每一视觉行都以 \\n 结尾，
    导致法条句子被大量假换行截断（每 40~60 字一个 \\n）。

    本函数做两件事：
    1. 移除段落内部的假换行：单个 \\n（前后均非 \\n）→ 删除，使句子重新连贯
    2. 保留真正的段落边界：连续两个以上 \\n → 保持不变
    """
    import fitz  # pymupdf

    doc = fitz.open(str(file_path))
    if doc.is_encrypted:
        # 尝试空密码
        doc.authenticate("")

    texts: list[str] = []
    for page in doc:
        text = page.get_text()
        if not text.strip():
            continue

        # ── 核心：段落重建 ──────────────────────────────────────
        # 将孤立的 \\n（前后都不是 \\n）视为 PDF 视觉换行，直接删除
        # 连续 \\n\\n 或更多才是真正的段落分界，原样保留
        text = re.sub(r'(?<!\n)\n(?!\n)', '', text)
        texts.append(text.strip())

    doc.close()
    return "\n\n".join(texts)


# 文件扩展名 → 加载器映射
LOADER_MAP = {
    ".txt": load_txt,
    ".pdf": load_pdf,
}


# ══════════════════════════════════════════════════════════════════
#  元数据注入核心逻辑
# ══════════════════════════════════════════════════════════════════

# YAML 中需要从 metadata 排除的字段（文件名只用于定位，不注入）
METADATA_EXCLUDE_FIELDS = {"file_name"}

# ══════════════════════════════════════════════════════════════════
#  字段归一化：YAML 中同时存在两套命名惯例，统一映射为规范名
#
#  惯例 A（研发条例等）：policy_name / source（文号） / applicable_scope
#  惯例 B（其他条目）：source_name / source_number / topic
#  ↓ 归一化后 ↓
#  规范名：       source_name / source_number / topic
# ══════════════════════════════════════════════════════════════════
FIELD_NORMALIZE = {
    "policy_name":      "source_name",
    "source":           "source_number",   # 注意：YAML 中 source 实际是文号
    "applicable_scope": "topic",
}

# 向下兼容别名：metadata 中生成 "source" 指向 source_name
# （rag_agent.py format_docs() 仍会 fallback 到 source）
METADATA_ALIAS_MAP = {
    "source_name": "source",
}


def _parse_article_chapter(text: str) -> dict:
    """
    从 chunk 文本中提取"章/节/条"结构信息。

    支持三种中国税务文档格式：
      - 法律类：第X章、第X节、第X条（如增值税法、契税法、个税法）
      - 行政法规类：第X章、第X条（如个税专项扣除暂行办法）
      - 通知/公告类：一、二、三、… 编号段落（如财税〔2015〕119号）

    Returns:
        {"chapter": str, "article": str}  — 缺省时为空字符串
    """
    # 中文数字 → 数值映射（用于解析"十条"、"二十条"等）
    _CN_DIGIT = {
        "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9,
    }

    def _cn_to_arabic(cn: str) -> str:
        """将中文数字转为阿拉伯数字字符串，无法转换则返回原字符串。"""
        cn = cn.strip()
        if not cn:
            return cn
        # 单位数
        if cn in _CN_DIGIT:
            return str(_CN_DIGIT[cn])
        # 十、十一...十九
        if cn == "十":
            return "10"
        if cn.startswith("十") and len(cn) == 2:
            return str(10 + _CN_DIGIT.get(cn[1], 0))
        # 二十、三十...九十
        if cn.endswith("十") and len(cn) == 2:
            return str(_CN_DIGIT.get(cn[0], 0) * 10)
        # 二十一...九十九
        if "十" in cn and len(cn) >= 3:
            parts = cn.split("十", 1)
            tens = _CN_DIGIT.get(parts[0], 0) * 10
            ones = _CN_DIGIT.get(parts[1], 0) if parts[1] else 0
            return str(tens + ones)
        return cn

    result = {"chapter": "", "article": ""}

    # ── 1. 提取"第X章" ──────────────────────────────────────────
    chapter_m = re.search(r'第([一二三四五六七八九十百千]+)章', text)
    if chapter_m:
        result["chapter"] = f"第{chapter_m.group(1)}章"

    # ── 2. 提取"第X条"（可能有多条，取首尾组成范围）────────────
    article_matches = re.findall(r'第([一二三四五六七八九十百千]+)条', text)
    if article_matches:
        first = article_matches[0]
        last = article_matches[-1] if len(article_matches) > 1 else first
        if first == last:
            result["article"] = f"第{first}条"
        else:
            result["article"] = f"第{first}条—第{last}条"

    # ── 3. 通知/公告类：没有"第X条"，提取"一、二、"等编号 ────
    if not result["article"] and not result["chapter"]:
        section_m = re.search(r'[（\(]?([一二三四五六七八九十]+)[）\)]?\s*[、．.]', text)
        if section_m:
            result["article"] = f"{section_m.group(1)}、"

    return result


def _normalize_text(text: str) -> str:
    """物理排版恢复 — 在切分前重建法律/财税文本的段落结构。

    解决的问题：
      a) PDF/OCR 跨页截断 — 一句话被硬生生切成两半，中间夹了一个换行符
      b) 文本"泥石流"   — 丢失全部段落换行，法条挤成一整块，splitter 无法切分

    处理管线（严格按顺序执行）：
      第一步  缝合跨页断句 — 非完结标点后的 \\n 一律视为异常截断，直接吃掉
      第二步  恢复法律排版 — 第X章 / 第X条 前注入段落边界
      第三步  恢复公文排版 — 大点（一、）中点（（一））小点（1.）前注入换行
      第四步  收尾压缩     — 连续 3+ 个 \\n 压缩为 \\n\\n

    设计目标：无论原始文本有多"脏"，经过本函数后 splitter 都能按语义边界
    干净切分，不再出现法条中截断、跨页残留、或段落粘连。
    """
    # ── 0. 行尾符归一化（Windows \\r\\n / 老 Mac \\r → Unix \\n）──
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.strip()

    # ════════════════════════════════════════════════════════════════
    # 第一步：缝合跨页断句
    #
    # 规则：扫描所有 \\n。如果 \\n 前面的字符不是中文完结标点
    #       （。！？；：、”、】），就认为这是一句被 PDF 跨页截断的话，
    #       直接吃掉这个 \\n，将前后文本无缝拼接。
    #
    # 例子：
    #   "纳税义务发生时间，\n为签订合同的当日。"  → 拼接为一句
    #   "本法自2026年1月1日起施行。\n第二条..."   → 句号后的 \\n 保留
    # ════════════════════════════════════════════════════════════════
    #
    # 采用捕获组方案（而非 lookbehind）：匹配"非完结标点 + \\n"，
    # 替换时只保留捕获的非完结标点字符，吃掉 \\n。
    #
    # 注意：[^...] 中不包含 \\n 本身，因此 \\n\\n 中第二个 \\n
    # 会被前一个 \\n "吃掉" → \\n\\n 降为 \\n，后续第四步统一处理。
    text = re.sub(
        r'([^。！？；：、”、】\n])\n',
        r'\1',
        text,
    )

    # ════════════════════════════════════════════════════════════════
    # 第二步：恢复法律体排版（章、条）
    # ════════════════════════════════════════════════════════════════

    # 2a. 第X章 → \\n\\n 前导 + \\n 后随
    #     匹配 "第一章"、"第十二章"、"第一百二十章" 等
    text = re.sub(
        r'(第[一二三四五六七八九十百千]+章)',
        r'\n\n\1\n',
        text,
    )

    # 2b. 第X条 → \\n\\n 前导
    #     匹配 "第一条"、"第十五条"、"第一百二十条" 等
    text = re.sub(
        r'(第[一二三四五六七八九十百千]+条)',
        r'\n\n\1',
        text,
    )

    # ════════════════════════════════════════════════════════════════
    # 第三步：恢复财税公文体排版（大点 / 中点 / 小点）
    # ════════════════════════════════════════════════════════════════

    # 3a. 大点：中文数字 + 顿号 → \\n\\n 前导
    #     匹配 "一、" "二、" ... "十、" "十一、" ... "三十一、" 等
    text = re.sub(
        r'([一二三四五六七八九十]+、)',
        r'\n\n\1',
        text,
    )

    # 3b. 中点：带括号的中文数字 → \\n 前导
    #     兼容全角括号（一）（二）和半角括号 (一)(二)
    #     中文数字 1~3 个字符（一 到 九十九）
    text = re.sub(
        r'([（(][一二三四五六七八九十]{1,3}[）)])',
        r'\n\1',
        text,
    )

    # 3c. 小点：阿拉伯数字 + 点号 → \\n 前导
    #     兼容全角点号 1．2．和半角点号 1. 2.
    #     (?<!\d) 防止匹配 "3.14" 中的 "3." 部分
    #     (?!\d)  防止匹配 "3.14" 中的 ".1" 部分（点号后紧跟数字视为小数）
    text = re.sub(
        r'(?<!\d)(\d+)[．.](?!\d)',
        r'\n\1.',
        text,
    )

    # ════════════════════════════════════════════════════════════════
    # 第四步：收尾清理
    # ════════════════════════════════════════════════════════════════

    # 4a. 连续 3 个及以上 \\n → 压缩为 \\n\\n（恰好一个空行）
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 4b. 去除首尾换行（步骤 2/3 可能在文本两端插入了前导/尾随 \\n\\n）
    return text.strip()


def _build_metadata(entry: dict) -> dict:
    """
    从 YAML entry 构建 metadata dict。

    处理流程：
    1. 字段归一化：将 YAML 中的两套命名惯例统一为规范名
       (policy_name → source_name, source → source_number, applicable_scope → topic)
    2. 排除 file_name（物理定位用，不注入语义元数据）
    3. list 类型转为逗号分隔字符串（Chroma 只接受标量）
    4. 生成 source 别名指向 source_name（兼容 rag_agent.py）
    5. 添加统一 date 字段（优先 date_effective，其次 date_published）
    6. 其他字段（如 status、doc_type、date_*、url）原样保留
    """
    metadata = {}

    for key, value in entry.items():
        if key in METADATA_EXCLUDE_FIELDS:
            continue

        # 字段归一化：将 YAML 中的异构命名映射为统一 key
        normalized_key = FIELD_NORMALIZE.get(key, key)

        # 将 list 类型转为逗号分隔字符串（Chroma 只接受标量）
        if isinstance(value, list):
            value = "，".join(str(v) for v in value)

        metadata[normalized_key] = value

    # 生成向下兼容的别名（如 source → source_name）
    for src_key, alias in METADATA_ALIAS_MAP.items():
        if src_key in metadata and alias not in metadata:
            metadata[alias] = metadata[src_key]

    # 生成统一 date 字段（优先生效日期，其次发布日期）
    if "date" not in metadata:
        metadata["date"] = metadata.get("date_effective") or metadata.get("date_published") or ""

    return metadata


def load_and_chunk(entry: dict) -> list[Document]:
    """
    加载单个 YAML entry 对应的物理文件，切分并注入元数据。

    步骤：
    1. 从 entry["file_name"] 获取文件名
    2. 在 DOCS_DIR 下查找物理文件
    3. 根据扩展名选择加载器读取全文
    4. 用 RecursiveCharacterTextSplitter 切分
    5. 将 entry 中除 file_name 外的所有字段注入每个 chunk
    6. 为每个 chunk 解析章/节/条信息并注入 metadata
    """
    file_name = entry["file_name"]
    file_path = DOCS_DIR / file_name

    if not file_path.exists():
        print(f"  [WARN]  跳过（文件不存在）: {file_path}")
        return []

    ext = file_path.suffix.lower()
    loader = LOADER_MAP.get(ext)

    if loader is None:
        print(f"  [WARN]  跳过（不支持的格式 .{ext}）: {file_path}")
        return []

    # 1. 加载全文
    print(f"  读取: {file_name}")
    raw_text = loader(file_path)
    text = _normalize_text(raw_text)
    print(f"    提取完整文本，共 {len(text)} 个字符")

    if not text:
        print(f"  [WARN]  跳过（文件内容为空）: {file_path}")
        return []

    # 2. 切分
    splitter = RecursiveCharacterTextSplitter(
        separators=CHUNK_SEPARATORS,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    chunks = splitter.split_text(text)
    print(f"    切分为 {len(chunks)} 个文本块")

    # 3. 构建元数据（基于 YAML entry）
    base_metadata = _build_metadata(entry)

    # 4. 全局上下文注入：提取文件标题，拼接为 chunk 前缀
    #    解决"语义词汇鸿沟"问题 — 包含具体数字（如"1500元"）的 chunk
    #    因缺少税种关键词而在向量检索中被挤出 Top-K。前缀让每个 chunk
    #    都携带所属文件的完整标题，embedding 计算时包含文件级语义锚点。
    doc_title = base_metadata.get("source_name") or base_metadata.get("source") or "未知文件"
    context_prefix = f"【所属文件标题：{doc_title}】\n"

    # 5. 组装 Document 对象，每个 chunk 附加独立的 article/chapter
    documents = []
    for chunk in chunks:
        chunk_meta = base_metadata.copy()
        # 解析当前 chunk 的章/条结构
        struct = _parse_article_chapter(chunk)
        if struct["chapter"]:
            chunk_meta["chapter"] = struct["chapter"]
        if struct["article"]:
            chunk_meta["article"] = struct["article"]
        # 注入上下文前缀 + 正文
        documents.append(Document(page_content=context_prefix + chunk, metadata=chunk_meta))

    return documents


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════

def main():
    # ── 1. 加载 YAML ──────────────────────────────────────────────
    if not YAML_PATH.exists():
        print(f"[ERROR] 未找到 data_source.yaml: {YAML_PATH}")
        return

    with open(YAML_PATH, "r", encoding="utf-8") as f:
        entries = yaml.safe_load(f)

    if not entries:
        print("[ERROR] data_source.yaml 为空或格式错误")
        return

    print(f"[YAML] 加载 YAML 索引: {len(entries)} 个政策条目\n")

    # ── 2. 遍历 YAML entries，加载 + 切分 + 注入元数据 ──────────
    all_documents: list[Document] = []

    for i, entry in enumerate(entries, 1):
        source_name = entry.get("source_name", entry.get("file_name", "?"))
        print(f"[{i}/{len(entries)}] {source_name}")

        try:
            docs = load_and_chunk(entry)
            all_documents.extend(docs)
            print(f"    已生成 {len(docs)} 个 Document（累计: {len(all_documents)}）\n")
        except Exception as e:
            print(f"    [ERROR] 处理失败: {e}\n")
            continue

    if not all_documents:
        print("[ERROR] 没有生成任何文档片段，请检查文件路径和格式")
        return

    # ── 3. 打印预览 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"总计 {len(all_documents)} 个文档片段\n")

    # 统计各来源分布
    source_counts: dict[str, int] = {}
    for doc in all_documents:
        src = doc.metadata.get("source_name", "未知")
        source_counts[src] = source_counts.get(src, 0) + 1

    print("来源分布:")
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {count:>4}  | {src}")

    # 打印前 5 条样本
    print(f"\n前 5 条片段预览:\n")
    for i, doc in enumerate(all_documents[:5], 1):
        meta = doc.metadata
        src = meta.get("source_name", "?")[:30]
        num = meta.get("source_number", "")
        tp = meta.get("doc_type", "")
        preview = doc.page_content[:100].replace("\n", " ").replace("\r", " ")
        # 移除不可打印字符
        import unicodedata
        preview = "".join(c for c in preview if c.isprintable() or c in " ")
        print(f"  [{i}] {src}")
        if num:
            print(f"      文号: {num}")
        print(f"      类型: {tp}  |  生效: {meta.get('date_effective', '')}")
        topics = meta.get("topic", "")
        if topics:
            print(f"      主题: {topics}")
        print(f"      内容: {preview}...")
        print()

    # ── 4. 加载嵌入模型 ──────────────────────────────────────────
    print("正在加载 Embedding 模型（BAAI/bge-small-zh-v1.5）...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("Embedding 模型加载完成\n")

    # ── 5. 存入 Chroma ───────────────────────────────────────────
    print(f"正在构建向量数据库，存储路径: {DB_DIR}")

    if DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("  已清除旧的向量数据库")

    vectordb = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=str(DB_DIR),
        collection_name="tax_knowledge",
    )

    count = vectordb._collection.count()
    print(f"\n[OK] 向量数据库构建完成！")
    print(f"   集合名称: tax_knowledge")
    print(f"   向量总数: {count}")
    print(f"   持久化路径: {DB_DIR}")


if __name__ == "__main__":
    main()
