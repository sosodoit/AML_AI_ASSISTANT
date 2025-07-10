import os
import re
import json
import urllib.request
import urllib.parse
import pandas as pd
import requests
from bs4 import BeautifulSoup
# from newspaper import Article
from datetime import datetime, timezone, timedelta
import time
from dotenv import load_dotenv
load_dotenv()

KST = timezone(timedelta(hours=9))

def fetch_news_data(start_date, end_date, client_id=None, client_secret=None, keywords=None):
    if client_id is None or client_secret is None:
        raise ValueError("client_id와 client_secret을 전달해야 합니다.")

    if keywords is None:
        keywords = [
            "보이스피싱", "전기통신금융사기", "자금세탁", "의심거래", "STR", "AML", "고액현금거래",
            "금융사기", "스미싱", "사기 피해", "금융정보분석원", "대포통장"
        ]

    display = 100
    sort = 'date'
    max_start = 1000
    results = []

    for keyword in keywords:
        encText = urllib.parse.quote(keyword)
        base_url = f"https://openapi.naver.com/v1/search/news?query={encText}"

        for start in range(1, max_start, display):
            url = f"{base_url}&display={display}&start={start}&sort={sort}"
            request = urllib.request.Request(url)
            request.add_header("X-Naver-Client-Id", client_id)
            request.add_header("X-Naver-Client-Secret", client_secret)

            try:
                response = urllib.request.urlopen(request)
                if response.getcode() == 200:
                    response_body = response.read()
                    data = json.loads(response_body.decode('utf-8'))

                    for item in data['items']:
                        pubDate = datetime.strptime(item["pubDate"], "%a, %d %b %Y %H:%M:%S %z").date()
                        if start_date <= pubDate <= end_date:
                            results.append({
                                'keyword': keyword,
                                'title': item['title'],
                                'link': item['link'],
                                'pubDate': pubDate.isoformat()
                            })
                        elif pubDate < start_date:
                            break
                else:
                    break

            except Exception as e:
                print("Request failed:", e)
                break

    df_news = pd.DataFrame(results)
    df_news = df_news[df_news['link'].str.startswith('https://n.news.naver.com/mnews/article')].reset_index(drop=True)
    df_news.drop_duplicates(subset=['link'], inplace=True)
    return df_news

# def extract_news_text(df_grouped):
#     collected = []
#     for _, row in df_grouped.iterrows():
#         url = row['link']
#         try:
#             article = Article(url, language='ko')
#             article.download()
#             article.parse()
#             collected.append({
#                 'keyword': row['keyword'],
#                 'title': row['title'],
#                 'link': row['link'],
#                 'pubDate': row['pubDate'],
#                 'contents': article.text
#             })
#         except:
#             collected.append({
#                 'keyword': row['keyword'],
#                 'title': row['title'],
#                 'link': row['link'],
#                 'pubDate': row['pubDate'],
#                 'contents': ""
#             })

#     df = pd.DataFrame(collected)
#     return df

def extract_news_text(df_news):
    
    # 뉴스 본문을 저장할 리스트
    full_articles = []
    links = df_news['link']

    # 링크마다 뉴스 본문 크롤링 (BeautifulSoup)
    for link in links:
        try:
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(link, headers=headers, timeout=5)
            soup = BeautifulSoup(response.text, 'html.parser')

            # 본문 태그 후보들 (네이버 뉴스 기준)
            content = soup.find('article', {'id': 'dic_area'})
            # title = soup.find('h2', {'id': 'title_area'})

            if content:
                full_articles.append(content.get_text(strip=True))
            else:
                full_articles.append('')

            # if title:
            #     titles.append(title.get_text(strip=True))
            # else:
            #     titles.append('')

        except Exception as e:
            full_articles.append('')

        time.sleep(1)  # 너무 빠른 요청 방지

    # DataFrame에 추가
    df_news['contents'] = full_articles
    return df_news

def extract_sid(link):
    match = re.search(r'sid=(\d+)', link)
    if match:
        return int(match.group(1))
    return None

def preprocess_news_content(text):
    if pd.isna(text): return ""

    # 기자명 패턴 제거: "기자=", "기자 " 포함 라인 제거
    text = re.sub(r"[가-힣]{2,4}\s?기자", "", text)
    text = re.sub(r"기자\s?[가-힣]{2,4}", "", text)
    # 이메일 주소 제거
    text = re.sub(r'\S+@\S+', '', text)
    # 날짜 제거 (예: 2025.6.29, 2025-06-29 등)
    text = re.sub(r"\d{4}[.\-년]\d{1,2}[.\-월]?\d{1,2}[일]?", "", text)
    # 특수기호 제거 (필요한 기호는 남김)
    text = re.sub(r"[=※◆●■◆▷▶▶️▲△▼▽■□◆◇]", " ", text)
    # 출처 제거: 괄호 안에 뉴스사 포함
    text = re.sub(r'\([^\)]*(뉴스1|연합뉴스|뉴시스|MBN|KBS|SBS|YTN)[^\)]*\)', ' ', text)
    # 사진, 자료, 제공 등 표현 제거
    text = re.sub(r"(사진[=:\s]?|자료[=:\s]?|이미지[=:\s]?|제공[=:\s]?)[^\s]{0,20}", "", text)
    # HTML 엔티티 제거
    text = re.sub(r'&[a-z]+;', ' ', text)
    # 뉴스 사이트 고지/광고/알림 제거
    drop_keywords = [
        "자동 추출 기술", "전체 보기 권장", "All rights reserved", "무단 전재",
        "기사 섹션", "닫기", "QR 코드", "YTN LIVE", "유튜브 채널", "[전화]", "[메일]",
        "제보가 뉴스가 됩니다", "구독자", "채널 추가", "이벤트", "크게 볼 수 있어요",
        "영상편집", "자막뉴스", "뉴스1", "연합뉴스", "뉴시스", "MBN", "KBS", "SBS", "YTN"
    ]
    lines = text.split('\n')
    filtered_lines = [line for line in lines if not any(kw in line for kw in drop_keywords)]
    # 불필요한 공백 및 줄바꿈 정리
    text = '\n'.join(filtered_lines)
    text = re.sub(r'\n{2,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def run_news_pipeline(start_date, end_date, client_id, client_secret):
    sid_to_category = {
        100: '정치',
        101: '경제',
        102: '생활/문화',
        103: 'IT/과학',
        104: '세계',
        105: '사회'
    }

    df_grouped = fetch_news_data(start_date, end_date, client_id=client_id, client_secret=client_secret)
    df_news = extract_news_text(df_grouped)
    df_news = df_news.dropna()
    df_news = df_news.reset_index(drop=True)
    df_news['sid'] = df_news['link'].apply(extract_sid)
    df_news['categories'] = df_news['sid'].map(sid_to_category)    
    df_news['cleaned_content'] = df_news['contents'].apply(preprocess_news_content)
    final_df_news = df_news[['keyword', 'title', 'link', 'cleaned_content', 'categories', 'pubDate']]
    final_df_news.rename(columns={'cleaned_content':'contents'}, inplace=True)
    return final_df_news
