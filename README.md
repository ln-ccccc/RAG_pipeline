# 多模态 RAG（Qwen-VL + DashScope + FAISS）

这是一个最小可运行的多模态 RAG 终端 Demo：从 PDF 中抽取文本与图片，把图片转成可检索的「图像说明（caption）」并向量化存入 FAISS；对用户问题做 Query 改写、向量检索与重排，再把检索到的文本与图片作为上下文交给 Qwen-VL 生成回答。

## 背景

在企业知识库/技术文档场景里，关键信息经常不只在正文：大量结论藏在图表、流程图、截图、表格里。传统“纯文本 RAG”即使能把 PDF 文本抽出来，也会丢掉图片承载的信息，导致检索命中但回答缺关键细节。

这个项目的目标是用“图片 → 说明文本 → 向量检索”的方式，把 PDF 中的视觉信息也纳入检索空间，并在生成阶段让多模态模型同时看到「检索文本 + 相关图片」，从而提升回答的可用性与可解释性。

## 方案概览

- 数据侧：PDF 解析 → 抽取长文本块 → 抽取图片 → 对图片生成 caption
- 表征侧：对“文本块 + caption”统一走 `text-embedding-v1` 得到向量；归一化后用 FAISS `IndexFlatIP` 做近似“余弦相似度”检索
- 检索侧：Query 改写（更适合检索）→ 初检 TopK → 用改写 Query 的向量对候选重排
- 生成侧：把 TopN 文本拼成 context，并可附带 1 张最相关图片，交给 `qwen-vl-plus` 生成回答（约束“只基于上下文回答”）

## 项目结构

- `RAG_api_Qwen_VL.py`：主脚本（建库/加载索引、检索、重排、对话循环）
- `knowledge_base_multimodal/`：本地知识库目录（放 PDF）
- `knowledge_base_multimodal/extracted_images/`：从 PDF 抽取出来的图片（自动生成）
- `faiss_index_qwen_api_rag/`：向量库缓存（自动生成）
  - `index.faiss`：FAISS 索引
  - `index_to_doc.pkl`：向量 id → 元数据（文本/图片说明/source）
  - `embeddings.pkl`：归一化后的向量矩阵（用于重排）

## 快速开始（本地）

### 1）准备 Python 环境

建议 Python 3.10+，并使用虚拟环境

安装依赖（按脚本 import 列表给出）：

```bash
pip install pymupdf pillow numpy faiss-cpu scikit-learn dashscope python-dotenv
```

说明：`faiss-cpu` 在 macOS/Apple Silicon 上如遇到安装问题，可考虑使用 conda 方案或更换 Python 版本（这属于环境兼容性问题，不影响代码逻辑）。

### 2）配置 DashScope Key

在项目根目录创建 `.env`，写入：

```dotenv
DASHSCOPE_API_KEY=你的DashScopeKey
```

脚本会在启动时读取该变量；缺失会直接退出。

### 3）准备知识库 PDF

把你的 PDF 放到 `knowledge_base_multimodal/` 目录下，例如：

```text
knowledge_base_multimodal/
  your_doc_1.pdf
  your_doc_2.pdf
```

首次运行会自动抽取图片到 `knowledge_base_multimodal/extracted_images/` 并建立向量索引到 `faiss_index_qwen_api_rag/`。

### 4）启动

```bash
python3 RAG_api_Qwen_VL.py
```

看到提示 `=== 多模态 RAG 流式对话系统... ===` 后即可输入问题；输入 `exit` 或 `quit` 退出。

## 可配置项（脚本内常量）

在 `RAG_api_Qwen_VL.py` 顶部可以调整：

- `DATA_DIR`：知识库目录（默认 `knowledge_base_multimodal`）
- `VECTOR_STORE_PATH`：索引缓存目录（默认 `faiss_index_qwen_api_rag`）
- `TEXT_EMBED_MODEL`：文本向量模型（默认 `text-embedding-v1`）
- `QWEN_VL_MODEL`：多模态生成模型（默认 `qwen-vl-plus`）
- `CHAT_MODEL`：Query 改写用的生成模型（默认 `qwen-turbo`）

## 关键实现点

- 多模态信息进入检索空间：把图片转成 caption，再与文本一起做 embedding，避免“图表信息完全丢失”
- 向量检索的工程化落地：向量归一化 + `IndexFlatIP`，用内积实现余弦相似度检索，成本低、可控
- Query 改写 + 重排：用改写后的 query 做初检，并用同一语义空间向量对候选重打分，缓解“用户问题口语化/歧义”导致的召回不稳
- 生成阶段约束：system prompt 要求“仅基于提供上下文回答”，减少幻觉
- 计算缓存：索引与元数据落盘，二次启动直接加载，避免重复调用 embedding/caption

## 常见问题

- 运行时报 “No embeddings generated”：
  - 确认 `knowledge_base_multimodal/` 下有 PDF 文件
  - 确认 `DASHSCOPE_API_KEY` 有效（embedding/caption 都依赖 API）
- 想强制重建索引：
  - 删除 `faiss_index_qwen_api_rag/` 目录后重新运行脚本即可

