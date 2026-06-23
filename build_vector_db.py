"""
build_vector_db.py
------------------
插件式税务文档向量化工具。
支持 .txt 和 .pdf 文件，按法条切分，保留元数据。
新增文件格式时，只需添加对应的加载器函数即可。

支持的加载器:
  - load_txt_law: 按"第X条"切分 TXT 法律文本
  - load_pdf_generic: 通用 PDF 加载器（自动检测法条/表格/段落）
  - load_pdf_webpage: 网页截图类 PDF 加载器（自动剥离导航栏等噪声）
"""

import os
import re
import shutil
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# ── 0. 路径配置 ─────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

# ══════════════════════════════════════════════════════════════════
#  文件配置列表（新增文件只需在这里加一行）
#  每个条目包含：
#    path      - 文件路径
#    source    - 来源名称（如"中华人民共和国增值税法"）
#    loader    - 加载器函数名（见下方各加载器）
#    loader_kwargs - 传给加载器的额外参数（可选）
# ══════════════════════════════════════════════════════════════════
TAX_FILES = [
    {
        "path": os.path.join(BASE_DIR, "tax_data.txt"),
        "source": "中华人民共和国增值税法",
        "loader": "load_txt_law",
        "loader_kwargs": {"has_chapters": True, "skip_marker": "第一章　总则"},
    },
    {
        "path": os.path.join(BASE_DIR, "tax_data2.txt"),
        "source": "中华人民共和国契税法",
        "loader": "load_txt_law",
        "loader_kwargs": {"has_chapters": False, "skip_marker": None},
    },
    {
        "path": os.path.join(BASE_DIR, "tax3.pdf"),
        "source": "tax3.pdf",
        "loader": "load_pdf_generic",
        "loader_kwargs": {"password": None},
    },
    {
        "path": os.path.join(BASE_DIR, "tax4.pdf"),
        "source": "中华人民共和国个人所得税法",
        "loader": "load_pdf_webpage",
        "loader_kwargs": {
            "password": None,
            "article_start_marker": "第一条",
            "noise_markers": ["首页", "总局概况", "信息公开", "新闻发布",
                             "政策法规", "纳税服务", "互动交流", "专题专栏",
                             "网站地图", "网站管理", "联系我们", "主办单位",
                             "版权所有", "地址", "EN", "请输入您要搜索的内容",
                             "搜一搜", "打印本页", "字体", "大中小"],
        },
    },
]

# ══════════════════════════════════════════════════════════════════
#  加载器注册表：名称 → 函数
# ══════════════════════════════════════════════════════════════════
LOADER_REGISTRY = {}

def register_loader(name: str):
    """装饰器：将加载器函数注册到 LOADER_REGISTRY"""
    def decorator(func):
        LOADER_REGISTRY[name] = func
        return func
    return decorator


# ══════════════════════════════════════════════════════════════════
#  加载器 1：TXT 法律文本（按"第X条"切分）
# ══════════════════════════════════════════════════════════════════

# 中文数字映射
CN_NUM_MAP = {
    "一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
}
for i in range(1, 10):
    CN_NUM_MAP[f"十{i}"] = 10 + i
for i in range(1, 10):
    CN_NUM_MAP[f"二十{i}"] = 20 + i
for i in range(1, 10):
    CN_NUM_MAP[f"三十{i}"] = 30 + i
CN_NUM_MAP["二十"] = 20
CN_NUM_MAP["三十"] = 30

CHAPTER_PATTERN = re.compile(r"第([一二三四五六七八九十百]+)章\s*(.+)")


@register_loader("load_txt_law")
def load_txt_law(file_path: str, source: str,
                 has_chapters: bool = False,
                 skip_marker: str | None = None) -> list[Document]:
    """
    加载 TXT 格式的法律文本，按"第X条"切分。
    支持自动跳过目录、追踪章节信息。
    """
    # 1. 读取并清洗
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if skip_marker is not None:
        clean_lines = []
        started = False
        for line in lines:
            if not started and skip_marker in line:
                started = True
            if started:
                clean_lines.append(line)
        lines = clean_lines

    text = "".join(lines)
    print(f"  已读取正文，共 {len(text)} 个字符")

    # 2. 按法条切分
    documents = []
    current_chapter = ""
    buffer_lines = []
    current_article = ""

    def flush_article():
        nonlocal buffer_lines, current_article
        content = "\n".join(buffer_lines).strip()
        if not content:
            buffer_lines = []
            return
        doc = Document(
            page_content=content,
            metadata={
                "source": source,
                "article": current_article,
                "chapter": current_chapter,
            },
        )
        documents.append(doc)
        buffer_lines = []

    for line in text.split("\n"):
        if has_chapters:
            ch_match = CHAPTER_PATTERN.search(line)
            if ch_match:
                flush_article()
                current_article = ""
                current_chapter = f"第{ch_match.group(1)}章 {ch_match.group(2).strip()}"
                documents.append(Document(
                    page_content=line.strip(),
                    metadata={
                        "source": source,
                        "article": "",
                        "chapter": current_chapter,
                        "type": "chapter_heading",
                    },
                ))
                continue

        article_match = re.match(r"第([一二三四五六七八九十百]+)条\s*", line)
        if article_match:
            flush_article()
            current_article = f"第{article_match.group(1)}条"
            buffer_lines.append(line)
        else:
            if buffer_lines or current_article:
                buffer_lines.append(line)

    flush_article()

    print(f"  切分为 {len(documents)} 个片段")

    # 3. 合并过短的法条
    documents = _merge_short_docs(documents, min_chars=80)
    print(f"  合并短法条后共 {len(documents)} 个片段")

    return documents


# ══════════════════════════════════════════════════════════════════
#  加载器 2：PDF 通用文档（按页 + 段落切分）
# ══════════════════════════════════════════════════════════════════

@register_loader("load_pdf_generic")
def load_pdf_generic(file_path: str, source: str,
                     password: str | None = None,
                     chunk_size: int = 500,
                     chunk_overlap: int = 50) -> list[Document]:
    """
    加载 PDF 文件。
    支持表格检测：有表格的页面自动转为 Markdown 表格格式，
    再进行法条/段落切分。

    表格过滤策略：
    - 行数 >= 2 且列数 >= 2 才视为"潜在表格"
    - 进一步检查表格内容是否包含"数据特征"（如数字、百分比、税率相关关键词）
    - 过滤掉导航栏、页脚等布局表格
    """
    import fitz  # pymupdf

    # 1. 打开 PDF
    doc_pdf = fitz.open(file_path)
    if doc_pdf.is_encrypted and password:
        doc_pdf.authenticate(password)
    elif doc_pdf.is_encrypted and not password:
        doc_pdf.authenticate("")

    # 2. 逐页提取，检测表格并转为 Markdown
    MIN_TABLE_ROWS = 2
    MIN_TABLE_COLS = 2

    full_text_parts = []
    table_count = 0

    for page in doc_pdf:
        tables_found = page.find_tables()
        real_tables = [
            t for t in tables_found.tables
            if t.row_count >= MIN_TABLE_ROWS and t.col_count >= MIN_TABLE_COLS
        ]

        if real_tables:
            kept_tables = []
            for table in real_tables:
                md = table.to_markdown()
                if not md.strip():
                    continue
                # 过滤非数据表格：检查表格内容是否包含数据特征
                if _is_data_table(md):
                    kept_tables.append(md.strip())
                    table_count += 1
                else:
                    # 非数据表格（导航栏等），退化为纯文本提取
                    text = page.get_text()
                    if text.strip():
                        full_text_parts.append(text.strip())
                    break  # 有非数据表格的页面整体用文本提取
            full_text_parts.extend(kept_tables)
        else:
            text = page.get_text()
            if text.strip():
                full_text_parts.append(text.strip())

    page_count = len(doc_pdf)
    doc_pdf.close()

    full_text = "\n\n".join(full_text_parts)
    print(f"  已提取 PDF 文本，共 {len(full_text)} 个字符，{page_count} 页")
    if table_count > 0:
        print(f"  检测到 {table_count} 个数据表格，已转为 Markdown 格式")

    # 3. 判断是否包含法律条文结构
    if _has_article_structure(full_text):
        print("  检测到法条结构，按'第X条'切分...")
        return _split_by_articles(full_text, source)
    else:
        print(f"  未检测到法条结构，按 {chunk_size} 字符切分...")
        return _split_by_chunks(full_text, source, chunk_size, chunk_overlap)


# ══════════════════════════════════════════════════════════════════
#  通用切分工具函数
# ══════════════════════════════════════════════════════════════════

def _is_data_table(markdown_table: str) -> bool:
    """
    判断 Markdown 表格是否为"数据表格"（而非导航栏、页脚等布局表格）。

    数据表格的特征：
    - 包含数字、百分比、金额等数值内容
    - 包含税务相关关键词（税率、扣除、应纳税等）
    - 列数 >= 3 更可能是数据表格
    - 不含导航栏特征词（首页、网站地图等）
    """
    # 导航栏/页脚特征词
    noise_keywords = [
        "首页", "网站地图", "网站管理", "联系我们", "主办单位",
        "版权所有", "地址", "总局概况", "信息公开", "新闻发布",
        "政策法规", "纳税服务", "互动交流", "专题专栏",
        "EN", "请输入", "搜一搜", "打印本页", "字体", "大中小",
    ]
    noise_count = sum(1 for kw in noise_keywords if kw in markdown_table)
    # 如果超过 3 个噪声词，判定为导航/页脚表格
    if noise_count >= 3:
        return False

    # 数据特征词
    data_keywords = [
        "税率", "应纳税", "扣除", "级数", "所得", "税",
        "%", "％", "元", "万", "收入", "工资", "累计",
        "不超过", "超过", "速算", "扣除数",
    ]
    data_count = sum(1 for kw in data_keywords if kw in markdown_table)

    # 数字特征：检查单元格中是否包含数字
    import re
    number_cells = len(re.findall(r"\|\s*[\d,.，。]+\s*\|", markdown_table))

    # 判定条件：
    # 1. 有数据关键词 + 有数字单元格 → 数据表格
    # 2. 有大量数字单元格（>= 4） → 数据表格
    # 3. 否则 → 非数据表格
    if data_count >= 2 and number_cells >= 1:
        return True
    if number_cells >= 4:
        return True
    return False


# ══════════════════════════════════════════════════════════════════
#  加载器 3：网页截图 PDF（自动剥离导航栏等噪声）
# ══════════════════════════════════════════════════════════════════

@register_loader("load_pdf_webpage")
def load_pdf_webpage(file_path: str, source: str,
                     password: str | None = None,
                     article_start_marker: str = "第一条",
                     noise_markers: list[str] | None = None,
                     chunk_size: int = 500,
                     chunk_overlap: int = 50) -> list[Document]:
    """
    加载网页截图类 PDF（如从国家税务总局网站截取的税法页面）。
    自动剥离导航栏、页眉、页脚等噪声，提取正文内容。

    处理策略：
    1. 逐页提取文本
    2. 按行过滤噪声标记（导航栏文字、网站标识等）
    3. 从 article_start_marker 开始截取正文（跳过网页标题区域）
    4. 再按法条结构切分
    """
    import fitz  # pymupdf

    if noise_markers is None:
        noise_markers = []

    # 1. 打开 PDF
    doc_pdf = fitz.open(file_path)
    if doc_pdf.is_encrypted and password:
        doc_pdf.authenticate(password)
    elif doc_pdf.is_encrypted and not password:
        doc_pdf.authenticate("")

    # 2. 逐页提取文本，过滤噪声
    clean_lines = []
    started = False  # 是否已找到正文起始标记

    for page_num, page in enumerate(doc_pdf):
        text = page.get_text()
        if not text.strip():
            continue

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                if started:
                    clean_lines.append("")
                continue

            # 检测正文起始标记
            if not started and article_start_marker in stripped:
                started = True

            if not started:
                continue

            # 检查是否为噪声行
            is_noise = False
            for marker in noise_markers:
                if marker in stripped:
                    # 如果噪声标记恰好也是法条中出现的词，不跳过
                    # 噪声标记通常单独成行或出现在行首
                    if stripped == marker or stripped.startswith(marker):
                        is_noise = True
                        break
                    # 如果整行只有噪声标记相关的词，跳过
                    if len(stripped) <= len(marker) + 3:
                        is_noise = True
                        break

            if not is_noise:
                clean_lines.append(stripped)

    doc_pdf.close()

    full_text = "\n".join(clean_lines)
    print(f"  已提取并清洗网页 PDF 文本，共 {len(full_text)} 个字符")

    # 统计有效行数
    non_empty_lines = [l for l in clean_lines if l.strip()]
    print(f"  有效行数: {len(non_empty_lines)}")

    # 3. 判断是否包含法律条文结构
    if _has_article_structure(full_text):
        print("  检测到法条结构，按'第X条'切分...")
        docs = _split_by_articles(full_text, source)
        print(f"  切分为 {len(docs)} 个片段")
        return docs
    else:
        print(f"  未检测到法条结构，按 {chunk_size} 字符切分...")
        return _split_by_chunks(full_text, source, chunk_size, chunk_overlap)


def _has_article_structure(text: str) -> bool:
    """判断文本是否包含"第X条"结构"""
    return bool(re.search(r"第[一二三四五六七八九十百]+条", text))


def _split_by_articles(text: str, source: str) -> list[Document]:
    """按"第X条"切分（用于 PDF 中的法律文本）"""
    # 按"第X条"分割
    pattern = re.compile(r"(第[一二三四五六七八九十百]+条)")
    parts = pattern.split(text)

    documents = []
    current_article = ""
    for part in parts:
        if pattern.fullmatch(part):
            current_article = part
        elif current_article:
            content = (current_article + part).strip()
            if len(content) > 20:
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "article": current_article,
                        "chapter": "",
                    },
                ))
            current_article = ""

    return _merge_short_docs(documents, min_chars=80)


def _split_by_chunks(text: str, source: str,
                     chunk_size: int, chunk_overlap: int) -> list[Document]:
    """按固定大小切分（用于无结构的通用文档）"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    docs = splitter.create_documents(
        [text],
        metadatas=[{"source": source, "article": "", "chapter": ""}],
    )
    return docs


def _merge_short_docs(documents: list[Document],
                      min_chars: int = 80) -> list[Document]:
    """合并过短的文档片段"""
    if not documents:
        return documents

    merged = []
    buffer_doc = None

    for doc in documents:
        if doc.metadata.get("type") == "chapter_heading":
            if buffer_doc:
                merged.append(buffer_doc)
                buffer_doc = None
            merged.append(doc)
            continue

        if buffer_doc is None:
            buffer_doc = doc
        else:
            if (len(buffer_doc.page_content) < min_chars or
                    len(doc.page_content) < min_chars):
                buffer_doc.page_content += "\n" + doc.page_content
                a1 = buffer_doc.metadata.get("article", "")
                a2 = doc.metadata.get("article", "")
                if a1 and a2:
                    buffer_doc.metadata["article"] = f"{a1}—{a2}"
            else:
                merged.append(buffer_doc)
                buffer_doc = doc

    if buffer_doc:
        merged.append(buffer_doc)

    return merged


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════

def main():
    all_documents = []

    for cfg in TAX_FILES:
        file_path = cfg["path"]
        source = cfg["source"]
        loader_name = cfg["loader"]
        loader_kwargs = cfg.get("loader_kwargs", {})

        if not os.path.exists(file_path):
            print(f"\n⚠️  跳过（文件不存在）: {file_path}")
            continue

        print(f"\n{'='*50}")
        print(f"处理文件: {source}")
        print(f"  路径: {file_path}")
        print(f"  加载器: {loader_name}")

        loader = LOADER_REGISTRY.get(loader_name)
        if loader is None:
            print(f"  ❌ 未知加载器: {loader_name}，跳过")
            continue

        docs = loader(file_path, source=source, **loader_kwargs)
        all_documents.extend(docs)

    # 打印预览
    print(f"\n{'='*50}")
    print(f"总计 {len(all_documents)} 个文档片段:\n")
    for i, doc in enumerate(all_documents):
        src = doc.metadata["source"]
        ch = doc.metadata.get("chapter", "")
        art = doc.metadata.get("article", "")
        tp = doc.metadata.get("type", "")
        label = art if art else (f"[{tp}]" if tp else "[段落]")
        chapter_info = f" | {ch}" if ch else ""
        preview = doc.page_content[:80].replace("\n", " ").replace("\r", " ")
        # 移除不可打印的控制字符，避免 GBK 编码错误
        import unicodedata as _ucd
        preview = "".join(c for c in preview if c.isprintable() or c in "\t\n\r")
        print(f"  [{i+1}] {src}{chapter_info} {label}")
        try:
            print(f"      {preview}...")
        except UnicodeEncodeError:
            # 降级处理：强制替换无法编码的字符
            print(f"      {preview.encode('gbk', errors='replace').decode('gbk')}...")
        print()

    # ── 加载嵌入模型 ────────────────────────────────────────────
    print("正在加载嵌入模型...")
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-zh-v1.5",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("嵌入模型加载完成")

    # ── 存入 Chroma ─────────────────────────────────────────────
    print(f"\n正在构建向量数据库，存储路径: {DB_DIR}")

    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        print("  已清除旧的向量数据库")

    vectordb = Chroma.from_documents(
        documents=all_documents,
        embedding=embedding_model,
        persist_directory=DB_DIR,
        collection_name="tax_knowledge",
    )

    print(f"\n向量数据库构建完成，共 {vectordb._collection.count()} 条向量")
    print(f"数据已持久化到: {DB_DIR}")


if __name__ == "__main__":
    main()