# -*- coding: utf-8 -*-
"""
多模态 RAG：加入 Query 改写与重排模块
"""

import os
import io
import uuid
import time
import pickle
from http import HTTPStatus

import fitz                        # PyMuPDF
from PIL import Image
import numpy as np
import faiss
from sklearn.preprocessing import normalize
import dashscope
from dotenv import load_dotenv    # To load API key from .env file

# --- Configuration ---
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not DASHSCOPE_API_KEY:
    print("ERROR: DASHSCOPE_API_KEY not set."); exit(1)
dashscope.api_key = DASHSCOPE_API_KEY

DATA_DIR          = "knowledge_base_multimodal"
IMAGE_SAVE_DIR    = os.path.join(DATA_DIR, "extracted_images")
VECTOR_STORE_PATH = "faiss_index_qwen_api_rag"

TEXT_EMBED_MODEL  = "text-embedding-v1"
QWEN_VL_MODEL     = "qwen-vl-plus"
CHAT_MODEL        = "qwen-turbo"  # 用于 query 改写

os.makedirs(IMAGE_SAVE_DIR, exist_ok=True)


# --- 1. API Helpers ---

def rewrite_query(original_query: str) -> str:
    """
    调用 ChatCompletion 将用户原始查询改写为更适合检索的形式。
    """
    system = {"role":"system","content":"Rewrite the user's query to a concise, search-optimized form."}
    user   = {"role":"user","content":original_query}
    try:
        resp = dashscope.Generation.call(
            model=CHAT_MODEL,
            messages=[system, user],
            result_format='message'
        )
        if resp.status_code == HTTPStatus.OK:
            return resp.output['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"[Rewrite Error] {e}")
    return original_query  # 失败则退回原始


def get_text_embeddings_api(text_list):
    """同前，将 text_list 批量送入 DashScope TextEmbedding API。"""
    if not text_list:
        return []
    try:
        resp = dashscope.TextEmbedding.call(
            model=TEXT_EMBED_MODEL,
            input=text_list
        )
        if resp.status_code == HTTPStatus.OK:
            embs = [None]*len(text_list)
            for info in resp.output['embeddings']:
                idx = info['text_index']; embs[idx] = info['embedding']
            return embs
    except Exception as e:
        print(f"[Embedding Error] {e}")
    return [None]*len(text_list)


def generate_caption_api(image_path):
    """同前，用 Qwen-VL 生成图片说明。"""
    uri = f"file://{os.path.abspath(image_path)}"
    msg = [{"role":"user","content":[{"image":uri},{"text":"Describe this image in detail."}]}]
    try:
        resp = dashscope.MultiModalConversation.call(
            api_key=DASHSCOPE_API_KEY,
            model=QWEN_VL_MODEL,
            messages=msg
        )
        if resp.status_code == HTTPStatus.OK:
            return resp.output['choices'][0]['message']['content'] or "No description."
    except Exception as e:
        print(f"[Caption Error] {e}")
    return "Description error."


def generate_qwen_vl_response_api(query, retrieved, max_images=1):
    """同前，用检索到的多模态上下文调用 Qwen-VL 生成回答。"""
    system = {"role":"system","content":[{"text":"You are a helpful assistant. Answer ONLY from provided context."}]}
    user_content = []
    # 添加入选图像
    added=0
    for item in sorted(retrieved, key=lambda x:x['score'], reverse=True):
        if item['type']=="image_caption" and added<max_images:
            p = item.get('image_path')
            if p and os.path.exists(p):
                user_content.append({"image":f"file://{os.path.abspath(p)}"})
                added+=1
    # 文本上下文
    texts=[it['content'] for it in retrieved if it['type']=="text"]
    if texts:
        user_content.append({"text":"--- Context ---\n"+ "\n\n".join(texts) + "\n--- End ---"})
    # 用户问题
    user_content.append({"text":f"Question: {query}"})
    user = {"role":"user","content":user_content}

    try:
        resp = dashscope.MultiModalConversation.call(
            model=QWEN_VL_MODEL,
            messages=[system, user]
        )
        if resp.status_code==HTTPStatus.OK:
            ans = resp.output['choices'][0]['message']['content']
            if isinstance(ans, list):
                return "".join([c.get('text','') for c in ans])
            return ans
    except Exception as e:
        return f"[Generation Error] {e}"
    return "Generation failed."


# --- 2. 索引与向量存储 ---

def extract_and_index_api(data_dir, image_save_dir, index_path):
    texts_meta, images_meta = [], []
    # 从 PDF 中提取文本 & 图像
    for fn in os.listdir(data_dir):
        if not fn.lower().endswith(".pdf"): continue
        fp = os.path.join(data_dir, fn)
        try:
            doc = fitz.open(fp)
            for i in range(len(doc)):
                pg = doc.load_page(i)
                for blk in pg.get_text("blocks"):
                    txt=blk[4].strip()
                    if len(txt)>30:
                        texts_meta.append({"type":"text","content":txt,"source":f"{fn}:page{i+1}"})
                for img in pg.get_images(full=True):
                    xref=img[0]
                    imgd=doc.extract_image(xref)
                    ext=imgd["ext"].lower()
                    if ext not in ("png","jpg","jpeg","webp"): continue
                    name=f"img_{uuid.uuid4()}.{ext}"
                    savep=os.path.join(image_save_dir,name)
                    Image.open(io.BytesIO(imgd["image"])).save(savep)
                    images_meta.append({"type":"image","path":savep,"source":f"{fn}:page{i+1}"})
            doc.close()
        except Exception as e:
            print(f"[Extract Error]{fn}:{e}")

    # 准备 embedding：文本 + 图像说明
    all_meta, docs = [], []
    for t in texts_meta:
        all_meta.append(t); docs.append(t["content"])
    for img in images_meta:
        cap = generate_caption_api(img["path"])
        all_meta.append({"type":"image_caption","content":cap,"image_path":img["path"],"source":img["source"]})
        docs.append(cap)
        time.sleep(0.2)

    # 批量获取 embeddings
    embeddings, final_meta = [], []
    B=20
    for i in range(0,len(docs),B):
        batch=docs[i:i+B]
        embs=get_text_embeddings_api(batch)
        for m,e in zip(all_meta[i:i+B],embs):
            if e is not None:
                embeddings.append(e)
                final_meta.append(m)
        time.sleep(0.5)
    if not embeddings:
        print(f"No embeddings generated. Make sure there are PDFs under: {data_dir}")
        return None,None

    # 归一化 & build FAISS
    arr=np.array(embeddings,dtype='float32')
    arr=normalize(arr,axis=1)
    dim=arr.shape[1]
    idx=faiss.IndexFlatIP(dim)
    idx.add(arr)

    # 保存 index + meta + embeddings
    os.makedirs(index_path,exist_ok=True)
    faiss.write_index(idx,os.path.join(index_path,"index.faiss"))
    with open(os.path.join(index_path,"index_to_doc.pkl"),"wb") as f:
        pickle.dump(final_meta,f)
    with open(os.path.join(index_path,"embeddings.pkl"),"wb") as f:
        pickle.dump(arr,f)

    print(f"Built FAISS with {idx.ntotal} vectors.")
    return idx, final_meta


# --- 3. 检索 & 重排 ---

def retrieve_from_index_api(query, index, mapping, k=5):
    """初次检索，返回 top k items（已按距离排序）"""
    embs = get_text_embeddings_api([query])
    if not embs or embs[0] is None:
        return []
    q = normalize(np.array([embs[0]],dtype='float32'),axis=1)
    D,I = index.search(q,k)
    res=[]
    for dist,idx in zip(D[0],I[0]):
        if 0<=idx<len(mapping):
            itm = mapping[idx].copy()
            itm["mapping_idx"] = int(idx)
            itm['score']=float(dist)
            res.append(itm)
    return res


def rerank_results(query, candidates, embeddings_array):
    """
    使用改写后query的 embedding 与每个候选的原始 embedding 做内积，
    重新计算得分并排序。
    """
    # 1. embed 改写查询
    embs = get_text_embeddings_api([query])
    if not embs or embs[0] is None:
        return candidates
    qv = normalize(np.array([embs[0]],dtype='float32'),axis=1)[0]

    # 2. 遍历 candidates，根据 mapping index 找到 embedding
    scored=[]
    for c in candidates:
        idx = c.get('mapping_idx')
        if idx is None: 
            # 没有 idx 信息则保持原 score
            scored.append((c['score'],c))
        else:
            vec = embeddings_array[idx]
            new_score = float(np.dot(qv,vec))
            c['score'] = new_score
            scored.append((new_score,c))
    # 3. 重排
    scored.sort(key=lambda x:x[0],reverse=True)
    return [c for _,c in scored]


# --- 4. 主流程 & 对话循环 ---

# 加载或建库
idx_file = os.path.join(VECTOR_STORE_PATH,"index.faiss")
meta_file = os.path.join(VECTOR_STORE_PATH,"index_to_doc.pkl")
emb_file  = os.path.join(VECTOR_STORE_PATH,"embeddings.pkl")

if os.path.exists(idx_file) and os.path.exists(meta_file) and os.path.exists(emb_file):
    print("Loading FAISS index, metadata & embeddings...")
    faiss_index      = faiss.read_index(idx_file)
    with open(meta_file,"rb") as f: mapping = pickle.load(f)
    with open(emb_file,"rb")  as f: embeddings_array = pickle.load(f)
else:
    print("Building new FAISS index...")
    faiss_index, mapping = extract_and_index_api(DATA_DIR, IMAGE_SAVE_DIR, VECTOR_STORE_PATH)
    if faiss_index is None or mapping is None:
        print(f"ERROR: index build failed. Put PDF files under `{DATA_DIR}` and rerun.")
        exit(1)
    with open(os.path.join(VECTOR_STORE_PATH,"embeddings.pkl"),"rb") as f:
        embeddings_array = pickle.load(f)

def stream_print(text,delay=0.005):
    for ch in text:
        print(ch,end="",flush=True); time.sleep(delay)
    print()

def main_chat_loop():
    print("\n=== 多模态 RAG 流式对话系统（含 Query 改写 & 重排） ===")
    print("输入问题后回车，输入 'exit' 或 'quit' 退出。\n")
    while True:
        raw = input("You: ").strip()
        if raw.lower() in ("exit","quit"):
            print("再见！"); break

        # 1. Query 改写
        rewritten = rewrite_query(raw)
        print(f"[改写后 Query] {rewritten}")

        # 2. 初次检索
        topn = retrieve_from_index_api(rewritten, faiss_index, mapping, k=10)

        # 3. 重排（取前 5）
        reranked = rerank_results(rewritten, topn, embeddings_array)[:5]
        print(f"[重排后 Top5] Scores: {[round(x['score'],4) for x in reranked]}")

        # 4. 生成回答
        print("Assistant: ",end="",flush=True)
        ans = generate_qwen_vl_response_api(raw, reranked)
        stream_print(ans)

if __name__=="__main__":
    main_chat_loop()
