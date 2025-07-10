# AML AI ASSISTANT

안녕하세요, AML AI ASSISTANT(트리플A)는 금융권에서 수행하고 있는 AML/STR 업무 지원을 위한 AI 기반 대시보드 PoC 프로젝트입니다.

현재 뉴스 / 가이드 기반 질의응답 기능이 통합되어 있고, 간단한 인사이트를 제공할 수 있도록 구현하였습니다.

성능 실험은 포함되어 있지 않기에, 답변 품질은 낮습니다.

답변 포맷팅이나 품질 향상을 위한 실험과, 소규모의 비용을 들일 수 있다면 더욱 실용적인 서비스로 발전할 수 있다고 기대합니다.

---

## ⚙️ 주요 기능

| 기능 | 설명 |
|------|------|
| **뉴스 수집 & 전처리** | 네이버 뉴스 API를 통해 금융 사기 관련 기사 수집, 중복 제거 및 벡터 DB 메타데이터 고려한 전처리 수행 |
| **SQLite & FAISS 연동** | 수집한 기사 데이터를 SQLite에 저장하고, FAISS 기반 벡터 DB로 임베딩 저장 |
| **RAG 기반 질의응답** | 뉴스 및 법/가이드 문서를 RAG 시스템에 연결하여 자연어 질의응답 제공 |
| **Streamlit 대시보드** | 시각화 4종(일자별, 키워드, 카테고리, 본문 길이) 및 AI 인사이트 요약 기능 |

---

## 💡 프로젝트 구조

```
AML_AI_ASSISTANT/
├─ data/
│   ├─ news.db
│   ├─ faiss_news/
│   └─ faiss_guide/
├─ modules/
│   ├─ news_pipeline.py
│   ├─ db_to_vector.py
│   └─ rag_qa.py
├─ main.py
└─ requirements.txt
```

---

## 🚀 설치 및 실행

```bash
# 1. 가상환경 설정 (윈도우 기준)
python -m venv venv
.\venv\Scripts\activate

# 2. 필수 라이브러리 설치
pip install -r requirements.txt
```

---

## 🛠️ 사용 방법

### 1. Streamlit 실행
```bash
streamlit run main.py
```

### 2. AI질의응답 탭
- OpenAI API 키 입력
- 뉴스/가이드 질의를 통해 RAG 기반 응답 확인 가능

### 3. 데이터수집기 탭
- 네이버 API 키 입력
- 기간 설정 후 수집 버튼 클릭 → SQLite에 저장
- FAISS DB 생성 버튼 클릭 → 벡터 DB 생성

---

## 🎯 개선 방향
- 전처리 로직과 질의 템플릿을 개선
- RAG QA 파이프라인 재설계
- 다양한 Instruction 기반 QA 모델 활용
  
