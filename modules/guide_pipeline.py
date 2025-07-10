from langchain.schema import Document
from langchain_community.document_loaders import BSHTMLLoader
import pickle
import re

def preprocess_amls03(paragraphs: list, source: str = "amls03.html") -> list:
    docs = []
    current_title = None
    i = 0

    while i < len(paragraphs):
        current = paragraphs[i].strip()

        # 번호로 시작하는 소제목
        if re.match(r'^\d\)', current):
            subtitle = current
            body = paragraphs[i + 1].strip() if i + 1 < len(paragraphs) else ''
            content = f"{subtitle}: {body}"
            docs.append(Document(
                page_content=content,
                metadata={"source": source, "title": current_title or "기타"}
            ))
            i += 2

        # 일반 제목 (짧고 마침표 없음)
        elif len(current) <= 30 and not current.endswith('.') and i + 1 < len(paragraphs):
            current_title = current  # 이건 상위 제목으로 보관
            next_para = paragraphs[i + 1].strip()

            # 다음 문단이 번호형 소제목이 아닐 때만 본문으로 판단
            if not re.match(r'^\d\)', next_para):
                docs.append(Document(
                    page_content=next_para,
                    metadata={"source": source, "title": current_title}
                ))
                i += 2
            else:
                i += 1  # 제목만 넘김, 다음 루프에서 소제목으로 처리됨

        else:
            # 독립 본문 or 구조 밖 문단
            docs.append(Document(
                page_content=current,
                metadata={"source": source, "title": current_title or "기타"}
            ))
            i += 1

    return docs

def preprocess_guide04(paragraphs: list, source: str = "guide04.html") -> list:
    docs = []
    i = 0

    while i < len(paragraphs):
        current = paragraphs[i].strip()

        # 시각자료 관련 텍스트 제거
        if "좌측 단위" in current or "축" in current:
            i += 1
            continue

        # 자금세탁방지제도 개요 요약 저장
        if "자금세탁방지제도 3형제" in current:
            docs.append(Document(
                page_content="자금세탁방지 3대 제도는 CDD, STR, CTR입니다.",
                metadata={"source": source, "title": "자금세탁방지제도 개요"}
            ))
            i += 2  # 바로 다음 문단(CDD/STR/CTR 목록)은 건너뜀
            continue

        # 제도 이름(title) 감지 후 다음 줄을 본문으로 저장
        if any(keyword in current for keyword in ["고객확인제도", "의심거래보고제도", "고액현금거래보고제도"]):
            title = current
            body = paragraphs[i+1].strip() if i + 1 < len(paragraphs) else ''
            docs.append(Document(page_content=body, metadata={"source": source, "title": title}))
            i += 2
            continue

        # 일반 제목 (짧고 본문 뒤에 있음)
        if len(current) <= 30 and not current.endswith('.') and i + 1 < len(paragraphs):
            title = current
            body = paragraphs[i + 1].strip()
            docs.append(Document(
                page_content=body,
                metadata={"source": source, "title": title}
            ))
            i += 2
            continue

        # 기타 단독 본문 처리
        if len(current) > 30:
            # 직전 항목이 제도 명이지만 누락된 경우 처리
            if i > 0 and any(keyword in paragraphs[i-1] for keyword in ["고객확인제도", "의심거래보고제도", "고액현금거래보고제도"]):
                title = paragraphs[i-1].strip()
                docs.append(Document(page_content=current, metadata={"source": source, "title": title}))
            else:
                docs.append(Document(page_content=current, metadata={"source": source, "title": "기타"}))
        i += 1

    return docs

def normalize_text(text: str) -> str:
    text = re.sub(r'\r', '', text) # 캐리지 리턴(\r) 문자 제거
    text = re.sub(r'\n{2,}', '\n\n', text) # 연속된 줄바꿈 (\n\n\n...) → 2줄 개행(\n\n)으로 통일
    text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text) # 단일 개행은 공백으로 변환 (문단 나누기 의도가 없는 개행)
    text = re.sub(r'[ \t]+', ' ', text) # 연속된 공백( )이나 탭(\t)을 하나의 공백으로 바꿈
    text = text.strip() # 양 끝의 공백 제거

    return text

def process_html_file(path: str):
    loader = BSHTMLLoader(path)
    text = loader.load()[0].page_content
    text = normalize_text(text)

    if "amls03" in path:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        docs = preprocess_amls03(paragraphs)

    elif "guide04" in path:
        cleaned_text = text.replace('!', '.').replace('?\n\n', ' ')
        paragraphs = [p.strip() for p in cleaned_text.split('\n\n') if p.strip()]
        paragraphs[4] = paragraphs[4] + " " + paragraphs[5]
        del paragraphs[5]
        docs = preprocess_guide04(paragraphs) # Path(path).name

    else:
        docs = []

    return docs


if __name__ == "__main__":

    file_paths = [
        "amls03.html",
        "guide04.html"
    ]

    all_docs = []
    for html_file in file_paths:
        all_docs.extend(process_html_file(html_file))
    
    with open("processed_html.pkl", "wb") as f:
        pickle.dump(all_docs, f)

    # df = pd.DataFrame([{
    #     "title": doc.metadata.get("title", ""),
    #     "source": doc.metadata.get("source", ""),
    #     "length": len(doc.page_content),
    #     "preview": doc.page_content[:100]
    # } for doc in all_docs])