# 초기 적재용
import pandas as pd
import sqlite3
from pathlib import Path
import streamlit as st
from datetime import datetime

NEWS_PATH = Path("./data/processed_naver_news.csv")
DB_PATH = Path("./data/news.db")

def get_connection():
    return sqlite3.connect(DB_PATH)

def initialize_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            keyword TEXT,
            title TEXT,
            contents TEXT,
            link TEXT UNIQUE,
            categories TEXT,
            pubDate TEXT    
        )
    """)
    conn.commit()
    conn.close()

def insert_news_from_csv(csv_path: Path):
    if not csv_path.exists():
        st.warning(f"뉴스 CSV 파일이 존재하지 않습니다: {csv_path}")
        return 0

    df = pd.read_csv(csv_path)
    df = df[["keyword", "title", "contents", "link", "categories", "pubDate"]]

    # pubDate 형식 보정
    df["pubDate"] = pd.to_datetime(df["pubDate"]).dt.strftime("%Y-%m-%d")

    conn = get_connection()
    cur = conn.cursor()
    inserted_count = 0

    for _, row in df.iterrows():
        try:
            cur.execute("""
                INSERT OR IGNORE INTO news (keyword, title, contents, link, categories, pubDate)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                ", ".join(row["keyword"]) if isinstance(row["keyword"], list) else row["keyword"],
                row.get("title"),  
                row.get("contents"), 
                row.get("link"), 
                row.get("categories"),
                row.get("pubDate")
            ))
            inserted_count += cur.rowcount
        except Exception as e:
            st.warning(f"적재 중 오류: {e}")

    conn.commit()
    conn.close()
    return inserted_count

if __name__ == "__main__":
    initialize_db()
    insert_news_from_csv(NEWS_PATH)