import pandas as pd
import sqlite3
from pathlib import Path
import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 텍스트 임베딩 모델
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

# 뉴스 DB → Document로 변환
def convert_news_to_documents(db_path: str) -> list[Document]:
    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM news", conn)
    conn.close()

    df = df.dropna(subset=["contents"])
    df["keyword"] = df["keyword"].apply(lambda x: ", ".join(eval(x)) if isinstance(x, str) and x.startswith("[") else x)

    documents = []
    for _, row in df.iterrows():
        metadata = {
            "keyword": row["keyword"],
            "title": row["title"],
            "link": row["link"],
            "categories": row.get("categories", ""),
            "pubDate": row["pubDate"]          
        }
        content = row["contents"]
        if content:
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)
    return documents

# 가이드 HTML → Document로 변환 (이미 문서 리스트 형태로 저장됨)
def load_guide_documents(pkl_path: str) -> list[Document]:
    with open(pkl_path, "rb") as f:
        docs = pickle.load(f)
    return docs

# 공통 청크 분할 함수
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    return splitter.split_documents(documents)

# 뉴스 기반 FAISS 인덱스 저장
def build_faiss_from_sqlite(db_path: str, faiss_save_path: str):
    documents = convert_news_to_documents(db_path)
    chunks = split_documents(documents)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(faiss_save_path)

# 가이드 기반 FAISS 인덱스 저장
def build_faiss_from_guide(pkl_path: str, faiss_save_path: str):
    documents = load_guide_documents(pkl_path)
    chunks = split_documents(documents)
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(faiss_save_path)