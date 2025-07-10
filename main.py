# Step 1: 로컬 가상환경 구성 및 필수 라이브러리 설치 (사용자는 아래 명령어를 터미널에 실행)

# python 가상환경 생성 및 활성화 (윈도우)
# python -m venv venv
# .\venv\Scripts\activate

# python 가상환경 생성 및 활성화 (맥/리눅스)
# python3 -m venv venv
# source venv/bin/activate

# 필수 라이브러리 설치
# pip install streamlit langchain_openai langchain_community pandas beautifulsoup4 requests tiktoken faiss-cpu
# pip install python-dotenv matplotlib plotly newspaper3k kss
# pip install sentence-transformers

# Step 2: streamlit 앱 구현

import streamlit as st
import os
import sqlite3
from datetime import datetime, timedelta
from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
from pathlib import Path
import time

# 외부 모듈 import
from modules.news_pipeline import run_news_pipeline 
from modules.db_to_vector import build_faiss_from_sqlite, build_faiss_from_guide
from modules.rag_qa import get_faiss_qa_chain

# 환경 변수 로딩
load_dotenv()

# 기본 설정
st.set_page_config(page_title="AML AI ASSISTANT", layout="wide")

# 로컬 DB 및 데이터 경로
NEWS_PATH = Path("./data/processed_naver_news.csv")
HTML_PATH = Path("./data/processed_html.pkl")
DB_PATH = Path("./data/news.db")
FAISS_NEWS_PATH = Path("./data/faiss_news")
FAISS_GUIDE_PATH = Path("./data/faiss_guide")

# SQLite 연결 함수
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

# 뉴스 테이블 스키마 및 적재 함수
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

# 뉴스 중복 확인 후 삽입
def insert_news(df):
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
            st.warning(f"에러 발생: {e}")
    conn.commit()
    conn.close()
    return inserted_count

# 가장 최신 pubDate 조회
def get_latest_pubdate():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MAX(pubDate) FROM news")
    result = cur.fetchone()[0]
    conn.close()
    if result:
        return datetime.strptime(result, "%Y-%m-%d").date()
    else:
        return None

# 데이터 로딩
def load_data():
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM news", conn)
    conn.close()
    df["pubDate"] = pd.to_datetime(df["pubDate"])
    df["length"] = df["contents"].apply(lambda x: len(x) if x else 0)
    return df

# 세션 초기화
for i in range(1, 5):
    st.session_state.setdefault(f"insight{i}_result", None)

# 페이지 선택
page = st.sidebar.selectbox("메뉴", ["AI질의응답", "데이터수집기"])

if page == "AI질의응답":
    st.header("AML AI ASSISTANT")
    st.subheader("모델 설정")
    model_choice = st.selectbox("사용할 모델을 선택하세요", [
        "gpt-4o (OpenAI API 필요, 유료)",
        "gpt-3.5-turbo (OpenAI API 필요, 유료)",
        "mistral-7b-instruct (무료, HuggingFace)", 
        "llama3-8b-qa (무료, HuggingFace)"
    ])
    
    api_key = None
    def get_openai_model_id(choice):
        if "gpt-4" in choice:
            return "gpt-4o"
        elif "gpt-3.5" in choice:
            return "gpt-3.5-turbo"
        elif "mistral" in choice:
            return "mistral-7b-instruct"
        elif "llama3" in choice:
            return "llama3-8b-qa"
        else:
            return None
        
    if "OpenAI" in model_choice:
        api_key = st.text_input("OpenAI API 키 입력", type="password")
        model_id = get_openai_model_id(model_choice)
        if api_key:
            st.session_state.api_key = api_key
            st.session_state.model = model_id
            st.success("API 키 및 모델이 설정되었습니다.")
    else:
        model_id = get_openai_model_id(model_choice)
        st.session_state.model = model_id
        st.info(f"선택된 모델: {model_id} (API 키 필요 없음)")
    
    st.subheader("뉴스/가이드 기반 질의응답")
    question = st.text_area("질문을 입력하세요", placeholder="예: 가이드/STR 보고에 대해 알려줘, 뉴스/최근 금융사기뉴스 알려줘")

    if st.button("응답 받기") and question:
        if "뉴스/" in question:
            db_path = FAISS_NEWS_PATH
            prefix = "뉴스"
        elif "가이드/" in question:
            db_path = FAISS_GUIDE_PATH
            prefix = "가이드"
        else:
            st.warning("질문 앞에 '뉴스/' 또는 '가이드/'를 붙여주세요.")
            st.stop()

        try:
            with st.spinner(f"{prefix} 응답 생성 중"):
                rag_chain = get_faiss_qa_chain(str(db_path), api_key, model_name=model_id)
                result = rag_chain.invoke({"input": question})

            st.markdown("**모델 응답:**")
            st.write(result.get("answer", "답변 없음"))

            docs = result.get("context", [])
            if docs:
                st.subheader(f"참고한 {prefix} 문서 ({len(docs)}건)")
                for i, doc in enumerate(docs):
                    meta = doc.metadata
                    title = meta.get("title", f"{prefix} 문서 {i+1}")
                    with st.expander(f"{i+1}. {title}"):
                        if prefix == "뉴스":
                            st.markdown(f"**날짜:** {meta.get('pubDate')} | **카테고리:** {meta.get('categories')}")
                            st.markdown(f"**링크:** [{meta.get('link')}]({meta.get('link')})")
                        elif prefix == "가이드":
                            st.markdown(f"**출처 HTML 파일:** {meta.get('source')}")
                        st.markdown(doc.page_content[:500] + "...")
            else:
                st.info("참고 문서가 없습니다.")

        except Exception as e:
            st.error(str(e))

    # 통계 요약 생성
    df = load_data()
    
    daily_count = df.groupby(df["pubDate"].dt.date)["title"].count().reset_index(name="count")
    keyword_count = df["keyword"].value_counts().reset_index()
    keyword_count.columns = ["keyword", "count"]
    cat_count = df["categories"].value_counts().reset_index()
    cat_count.columns = ["category", "count"]
    avg_length = df["length"].mean()

    summary_context = {
        "length": f"- 뉴스 본문 평균 길이: 약 {avg_length:.0f}자",
        "days": f"- 뉴스 수집 일수: {daily_count.shape[0]}일",
        "keyword": f"- 주요 키워드: {', '.join(keyword_count.head(3)['keyword'])}",
        "category": f"- 주요 카테고리: {', '.join(cat_count.head(3)['category'])}"
    }

    # 시각화 및 인사이트 생성 UI
    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.bar(daily_count, x="pubDate", y="count", title="일자별 뉴스 건수")
        st.plotly_chart(fig1, use_container_width=True)
        with st.expander("인사이트 1: 일자별 뉴스 동향"):
            question = summary_context["days"] + "\n\n최근 일자별 뉴스 수에서 특이점이나 급증한 시기가 있는지 요약해줘."
            if st.button("인사이트 생성 1"):
                chain = get_faiss_qa_chain(FAISS_NEWS_PATH, st.session_state.api_key)
                result = chain.invoke({"input": question})
                st.session_state["insight1_result"] = result["answer"]
            if st.session_state["insight1_result"]:
                st.write(st.session_state["insight1_result"])

        fig3 = px.bar(cat_count, x="category", y="count", title="카테고리별 빈도")
        st.plotly_chart(fig3, use_container_width=True)
        with st.expander("인사이트 3: 뉴스 카테고리"):
            question = summary_context["category"] + "\n\n뉴스 카테고리 중 반복적으로 등장하는 주제를 금융사기유형 중심으로 알려줘."
            if st.button("인사이트 생성 3"):
                chain = get_faiss_qa_chain(FAISS_NEWS_PATH, st.session_state.api_key)
                result = chain.invoke({"input": question})
                st.session_state["insight3_result"] = result["answer"]
            if st.session_state["insight3_result"]:
                st.write(st.session_state["insight3_result"])

    with col2:
        fig2 = px.bar(keyword_count, x="keyword", y="count", title="키워드별 빈도")
        st.plotly_chart(fig2, use_container_width=True)
        with st.expander("인사이트 2: 키워드 분석"):
            question = summary_context["keyword"] + "\n\n자주 등장하는 키워드를 기반으로 금융사기유형에 대해 알려줘."
            if st.button("인사이트 생성 2"):
                chain = get_faiss_qa_chain(FAISS_NEWS_PATH, st.session_state.api_key)
                result = chain.invoke({"input": question})
                st.session_state["insight2_result"] = result["answer"]
            if st.session_state["insight2_result"]:
                st.write(st.session_state["insight2_result"])

        fig4 = px.histogram(df, x="length", nbins=30, title="뉴스 본문 길이 분포")
        st.plotly_chart(fig4, use_container_width=True)
        with st.expander("인사이트 4: 뉴스 밀도"):
            question = summary_context["length"] + "\n\n뉴스 본문 길이 분포를 기반으로 정보 밀도를 요약해줘."
            if st.button("인사이트 생성 4"):
                chain = get_faiss_qa_chain(FAISS_NEWS_PATH, st.session_state.api_key)
                result = chain.invoke({"input": question})
                st.session_state["insight4_result"] = result["answer"]
            if st.session_state["insight4_result"]:
                st.write(st.session_state["insight4_result"])

elif page == "데이터수집기":
    st.header("AML AI ASSISTANT")
    st.subheader("1. 뉴스 수집")
    client_id = st.text_input("네이버 Client ID", type="password")
    client_secret = st.text_input("네이버 Client Secret", type="password")
    start_date = st.date_input("시작일", datetime.today() - timedelta(days=7))
    end_date = st.date_input("종료일", datetime.today())
    
    if client_id and client_secret:
        option = st.radio("수집 방식 선택", ["뉴스 키워드로 수집"])    
        if st.button("수집 실행"):    
            progress = st.progress(0, text="수집 준비 중")        
            last_date = get_latest_pubdate()
            progress.progress(20, text="마지막 수집일 확인")
            if last_date:
                if end_date <= last_date:
                    st.warning("입력한 기간의 뉴스는 이미 수집되어 있습니다.")
                else:
                    with st.spinner("뉴스 수집 및 전처리 중"):
                        st.info("새로운 기간에 대해 수집을 진행합니다.")
                        df_collected = run_news_pipeline(start_date=last_date + timedelta(days=1), end_date=end_date, client_id=client_id, client_secret=client_secret)
                        progress.progress(60, text="수집 완료, DB 적재 중")
                        inserted = insert_news(df_collected)
                        df_collected.to_csv(f'{NEWS_PATH}_{last_date + timedelta(days=1)}', index=False, encoding='utf-8-sig')
                        progress.progress(100, text="모든 작업 완료")
                        st.success(f"{inserted}건의 뉴스가 DB에 추가되었습니다.")
            else:
                st.info("최초 수집을 시작합니다.")
                with st.spinner("뉴스 수집 및 전처리 중"):
                    df_collected = run_news_pipeline(start_date=start_date, end_date=end_date, client_id=client_id, client_secret=client_secret)
                    progress.progress(60, text="수집 완료, DB 적재 중")
                    inserted = insert_news(df_collected)
                    df_collected.to_csv(NEWS_PATH, index=False, encoding='utf-8-sig')
                    progress.progress(100, text="모든 작업 완료")
                    st.success(f"{inserted}건의 뉴스가 DB에 추가되었습니다.")

    st.subheader("2. 벡터 DB 생성")
    if st.button("뉴스 FAISS 생성"):
        with st.spinner("뉴스 FAISS 인덱스 생성 중"):
            build_faiss_from_sqlite(DB_PATH, FAISS_NEWS_PATH)
            st.success("뉴스 FAISS 저장 완료")

    if st.button("가이드 FAISS 생성"):
        with st.spinner("가이드 FAISS 인덱스 생성 중"):
            build_faiss_from_guide(HTML_PATH, FAISS_GUIDE_PATH)
            st.success("가이드 FAISS 저장 완료")

