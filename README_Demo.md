# README — 税务 RAG 问答系统 Demo 部署指南

## 概述

基于 Streamlit 的可视化对话界面，类 ChatGPT 风格，支持流式逐字输出与检索来源透明展示。

入口文件：`web_demo.py`，复用 `rag_agent.py` 中已构建的 Chroma 向量库和 LLM 链路。

---

## 环境要求

| 依赖 | 最低版本 | 说明 |
|------|---------|------|
| Python | 3.10+ | |
| streamlit | 1.33+ | Web UI（需要 `st.write_stream`） |
| langchain + langchain-chroma + langchain-huggingface | — | RAG 核心 |
| chromadb | — | 向量数据库 |
| sentence-transformers | — | Embedding 模型 |

---

## 安装

```powershell
pip install streamlit
```

> 其余依赖（`langchain`, `chromadb`, `sentence-transformers` 等）应在项目开发阶段已安装。

---

## 前置条件

启动 Web UI 前，请确保：

1. **向量库已构建**：`chroma_db/` 目录存在且非空。
   ```powershell
   python build_vector_db.py
   ```

2. **环境变量已设置**（LLM API Key）：
   ```powershell
   $env:INFINI_API_KEY = "your-api-key-here"
   ```
   或在项目根目录创建 `.env` 文件：
   ```
   INFINI_API_KEY=your-api-key-here
   ```

---

## 启动

```powershell
streamlit run web_demo.py
```

启动后浏览器会自动打开 `http://localhost:8501`。

---

## 界面说明

| 区域 | 说明 |
|------|------|
| **顶部输入框** | 输入税务问题，回车发送 |
| **置信度标签** | 每个回答顶部显示 HIGH（绿）/ LOW（黄）/ NOHIT（红） |
| **来源面板** | 点击 "📚 查看检索来源" 展开 Top-6 检索结果及其 L2 distance |
| **侧边栏** | 显示向量库规模、阈值配置、"清空对话"按钮 |
| **流式输出** | 回答逐字打印，TTFT < 2s |

---

## 常见问题

### Q: 启动时报 `ModuleNotFoundError: No module named 'streamlit'`

```powershell
pip install streamlit
```

### Q: Streamlit 页面空白 / 一直 Loading

首屏加载需要初始化 Embedding 模型（BAAI/bge-small-zh-v1.5），约 10-30 秒。请等待 "正在加载 Embedding 模型 & 向量数据库..." 提示消失。

### Q: Windows 终端无法显示 Emoji

`rag_agent.py` 已内置 `sys.stdout.reconfigure(encoding='utf-8')` 修复。如仍有问题，可用 Windows Terminal 代替传统的 conhost。
