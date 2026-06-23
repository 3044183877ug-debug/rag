# 税务 RAG 问答系统

基于 RAG（检索增强生成）的本地税务问答 Agent，支持多种格式的税法文档，可随时扩展新的税法文件。

## 项目结构

```
├── build_vector_db.py   # 向量库构建脚本（插件式，支持 txt/pdf）
├── rag_agent.py         # RAG 问答 Agent（控制台交互）
├── tax_data.txt         # 增值税法
├── tax_data2.txt        # 契税法
├── tax3.pdf             # 企业所得税法
├── tax4.pdf             # 个人所得税法
├── chroma_db/           # 本地向量数据库
├── .env                 # 环境变量（API Key，不上传）
└── .gitignore
```

## 快速开始

### 1. 安装依赖

```bash
pip install langchain langchain-core langchain-text-splitters
pip install langchain-huggingface langchain-chroma
pip install langchain-openai pymupdf python-dotenv
```

### 2. 配置 API Key

创建 `.env` 文件，填入无极（Infini-AI）平台的 API Key：

```
INFINI_API_KEY=sk-your-key-here
```

### 3. 构建向量库

```bash
python build_vector_db.py
```

脚本会读取目录下的所有税法文件，切分并向量化后存入 `chroma_db/`。新增文件只需在 `TAX_FILES` 列表中添加一行配置即可。

### 4. 启动问答

```bash
python rag_agent.py
```

输入问题即可获得基于税法的回答，输入 `quit` 退出。

## 添加新税法文件

在 `build_vector_db.py` 的 `TAX_FILES` 列表中添加条目：

```python
{
    "path": os.path.join(BASE_DIR, "your_new_file.pdf"),
    "source": "法律名称",
    "loader": "load_pdf_generic",   # 或 load_txt_law / load_pdf_webpage
    "loader_kwargs": {},
},
```

支持的加载器：

| 加载器 | 适用场景 |
|--------|---------|
| `load_txt_law` | 按"第X条"切分的法律 TXT 文本 |
| `load_pdf_generic` | 通用 PDF，自动检测表格和法条结构 |
| `load_pdf_webpage` | 网页截图 PDF，自动剥离导航栏等噪声 |

## 技术栈

- **Embedding**：`BAAI/bge-small-zh-v1.5`（HuggingFace）
- **向量库**：Chroma（本地持久化）
- **大模型**：DeepSeek-v4-pro（通过 Infini-AI 兼容 API）
- **RAG 框架**：LangChain v1.x（`create_stuff_documents_chain` 风格）
- **PDF 解析**：PyMuPDF（支持表格检测与 Markdown 转换）