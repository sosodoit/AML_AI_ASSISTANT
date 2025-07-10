from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from pathlib import Path

# FAISS 벡터 DB 로드 + RAG 체인 생성
def get_faiss_qa_chain(faiss_path, openai_api_key, model_name="gpt-3.5-turbo"):

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(
        temperature=0.3, 
        model=model_name, 
        openai_api_key=openai_api_key
        )

    prompt = ChatPromptTemplate.from_template(
            """
            "너는 AML 뉴스 분석 및 리스크 탐지 업무에 특화된 AI 어시스턴트야." 
            "너는 감성적 표현 없이 핵심을 명확히 전달하는 'T 성향' 스타일이야."
            "질문자가 원하는 분석 결과나 방향에 대해 직관적이고 요점 중심으로만 답변해."
            "필요시 표나 키워드 형태로 간단 명료하게 요약해줘. 불필요한 서두나 인삿말은 생략하고, 바로 핵심을 말해."

            아래는 참고할 수 있는 뉴스 요약입니다:
            ------------------
            {context}
            ------------------

            사용자의 질문:
            {input}

            위 자료를 참고하여 명확하고 간결하게 핵심만 요약해 응답하세요.
            """
        )
    
    document_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
        document_variable_name="context"
    )
    
    rag_chain = create_retrieval_chain(
        retriever=retriever, 
        combine_docs_chain=document_chain
    )

    return rag_chain