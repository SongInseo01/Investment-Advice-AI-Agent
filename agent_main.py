import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from langchain_upstage import ChatUpstage, UpstageEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from deep_translator import GoogleTranslator
import time
import agent_module as agentmodule
from langchain.schema import Document

# 한글 폰트 설정
def set_korean_font():
    if os.name == 'posix':
        plt.rcParams['font.family'] = 'AppleGothic'  # macOS
    elif os.name == 'nt':
        plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows
    else:
        plt.rcParams['font.family'] = 'NanumGothic'  # Linux

    # 마이너스 폰트 설정
    plt.rcParams['axes.unicode_minus'] = False

set_korean_font()

# Streamlit CSS 스타일 적용
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Noto Sans KR', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# 환경 변수 로드
load_dotenv()

# 기본 설정
api_key = os.getenv("UPSTAGE_API_KEY")

# Upstage Embeddings 및 Chat 모델 초기화
embeddings = UpstageEmbeddings(
    api_key=api_key,
    model="solar-embedding-1-large"
)
chat = ChatUpstage(api_key=api_key, model="solar-pro")

# Vector Store 초기화 (대화 기록 저장용)
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None  # 초기에는 None으로 설정


# JSON 파일 경로
yh_recovery_data_path = "./stock_data/yh_recovery_stocks.json"
yh_recovery_data = agentmodule.load_json_data_utf8(yh_recovery_data_path)

yh_growth_data_path = "./stock_data/yh_growth_stocks.json"
yh_growth_data = agentmodule.load_json_data_utf8(yh_growth_data_path)


yj_value_stocks_path = "./stock_data/yj_value_stocks_new.json"
yj_value_data = agentmodule.load_json_data_utf8(yj_value_stocks_path)

yj_dividend_stocks_path = "./stock_data/yj_dividend_stocks.json"
yj_dividend_data = agentmodule.load_json_data_utf8(yj_dividend_stocks_path)


sy_mixed_stocks_path = "./stock_data/sy_result.json"
sy_mixed_data = agentmodule.load_json_data_utf8(sy_mixed_stocks_path)


# 사이드바에 메뉴 추가
st.sidebar.title("메뉴")
menu_choice = st.sidebar.radio("작업 선택", ["대화하기", "종목추천 보기", "시각화 보기"])


# JSON 파일 기반 보기
if menu_choice == "종목추천 보기":
    # JSON 파일 로드 및 표시
    st.title("종목 추천")

    # 투자 방법 선택 멀티셀렉트박스 추가
    selected_methods = st.multiselect("투자 방법을 선택하세요", ["회복주", "성장주", "가치주", "배당주", "혼합주"])

    # 선택한 종목 리스트
    selected_stocks = [] # recovery
    selected_stocks_growth = []
    selected_stocks_value = []
    selected_stocks_divedend = []
    selected_stocks_mixed = []

    # 선택한 종목을 session_state에 저장하거나 불러오기
    if "selected_stocks" not in st.session_state:
        st.session_state.selected_stocks = []

    if "selected_stocks_growth" not in st.session_state:
        st.session_state.selected_stocks_growth = []

    if "selected_stocks_value" not in st.session_state:
        st.session_state.selected_stocks_value = []

    if "selected_stocks_divedend" not in st.session_state:
        st.session_state.selected_stocks_divedend = []
        
    if "selected_stocks_mixed" not in st.session_state:
        st.session_state.selected_stocks_mixed = []


    # 선택된 투자 방법에 따라 데이터를 표시
    for investment_method in selected_methods:
        if investment_method == "회복주":
            st.subheader("회복주")
            st.write("""
                    ##### 매출, 이익, 현금흐름 대비 주가가 낮아 저평가된 종목\n
                    ###### PSR 값이 0 이상 1 이하, PER 값이 10 이상 15 이하, PCR 값이 0 이상 10 이하\n
                    """)
            st.dataframe(yh_recovery_data)


            # 멀티셀렉트 박스를 통해 종목 선택
            selected_stocks += st.multiselect("회복주에서 종목을 선택하세요", yh_recovery_data["종목명"].tolist(), default=st.session_state.selected_stocks)
            # 선택한 종목을 session_state에 저장
            st.session_state.selected_stocks = selected_stocks
            # DataFrame에서 선택한 종목들만 필터링
            recovery_df = yh_recovery_data[yh_recovery_data["종목명"].isin(selected_stocks)]
            recovery_df["투자방법"] = "회복주"  # 투자 방법 열 추가
            # 선택된 회복주 종목을 session_state에 저장
            st.session_state["recovery_df"] = recovery_df
            # 확인용 출력
            st.write("선택한 종목 확인:", recovery_df)

        elif investment_method == "성장주":
            st.subheader("성장주")
            st.write("""
                    ##### 매출, 이익, 현금흐름 대비 시가총액이 낮아 저평가된 종목\n
                    ###### PSR 값이 0.1 이하, PER 값이 10 이하, PCR 값이 5 이하\n
                    """)
            st.dataframe(yh_growth_data)

            # 멀티셀렉트 박스를 통해 종목 선택
            selected_stocks_growth += st.multiselect("성장주에서 종목을 선택하세요", yh_growth_data["종목명"].tolist(), default=st.session_state.selected_stocks_growth)
            # 선택한 종목을 session_state에 저장
            st.session_state.selected_stocks_growth = selected_stocks_growth
            # DataFrame에서 선택한 종목들만 필터링
            growth_df = yh_growth_data[yh_growth_data["종목명"].isin(selected_stocks_growth)]
            growth_df["투자방법"] = "성장주"  # 투자 방법 열 추가
            # 선택된 회복주 종목을 session_state에 저장
            st.session_state["growth_df"] = growth_df
            # 확인용 출력
            st.write("선택한 종목 확인:", growth_df)


        elif investment_method == "가치주":
            st.subheader("가치주")
            st.write("""
                    ##### 순자산, 이익, 현금흐름 대비 주가가 매우 낮아 저평가된 종목\n
                    ###### PER 값이 5 이하 0 초과, PBR 값이 0.5 이하 0 초과, PCR 값이 5 이하 0 초과\n
                    """)
            st.dataframe(yj_value_data)

            # 멀티셀렉트 박스를 통해 종목 선택
            selected_stocks_value += st.multiselect("가치주에서 종목을 선택하세요", yj_value_data["종목명"].tolist(), default=st.session_state.selected_stocks_value)
            # 선택한 종목을 session_state에 저장
            st.session_state.selected_stocks_value = selected_stocks_value
            # DataFrame에서 선택한 종목들만 필터링
            value_df = yj_value_data[yj_value_data["종목명"].isin(selected_stocks_value)]
            value_df["투자방법"] = "가치주"  # 투자 방법 열 추가
            # 선택된 회복주 종목을 session_state에 저장
            st.session_state["value_df"] = value_df
            # 확인용 출력
            st.write("선택한 종목 확인:", value_df)

        elif investment_method == "배당주":
            st.subheader("배당주")
            st.write("""
                    ##### 배당수익률이 높으면서 저평가된 종목\n
                    ###### DY 값이 0.04 이상, PBR 값이 0 이상 1 이하, PER 값이 0 이상 15 이하\n
                    """)
            st.dataframe(yj_dividend_data)

            # 멀티셀렉트 박스를 통해 종목 선택
            selected_stocks_divedend += st.multiselect("배당주에서 종목을 선택하세요", yj_dividend_data["종목명"].tolist(), default=st.session_state.selected_stocks_divedend)
            # 선택한 종목을 session_state에 저장
            st.session_state.selected_stocks_divedend = selected_stocks_divedend
            # DataFrame에서 선택한 종목들만 필터링
            divedend_df = yj_dividend_data[yj_dividend_data["종목명"].isin(selected_stocks_divedend)]
            divedend_df["투자방법"] = "배당주"  # 투자 방법 열 추가
            # 선택된 회복주 종목을 session_state에 저장
            st.session_state["divedend_df"] = divedend_df
            # 확인용 출력
            st.write("선택한 종목 확인:", divedend_df)

        elif investment_method == "혼합주":
            st.subheader("혼합주")
            st.write("""
                    ##### 자산 대비 저평가되었으며, 배당수익률이 높고 높은 수익성을 유지하는 종목\n
                    ###### PBR 값이 1 이하 0 초과, PER 값이 15 이하 0 초과, DY 값이 0.03 이상, ROE 값 0.1 이상\n
                    """)
            st.dataframe(sy_mixed_data)

            # 멀티셀렉트 박스를 통해 종목 선택
            selected_stocks_mixed += st.multiselect("혼합주에서 종목을 선택하세요", sy_mixed_data["종목명"].tolist(), default=st.session_state.selected_stocks_mixed)
            # 선택한 종목을 session_state에 저장
            st.session_state.selected_stocks_mixed = selected_stocks_mixed
            # DataFrame에서 선택한 종목들만 필터링
            mixed_df = sy_mixed_data[sy_mixed_data["종목명"].isin(selected_stocks_mixed)]
            mixed_df["투자방법"] = "혼합주"  # 투자 방법 열 추가
            # 선택된 회복주 종목을 session_state에 저장
            st.session_state["mixed_df"] = mixed_df
            # 확인용 출력
            st.write("선택한 종목 확인:", mixed_df)


elif menu_choice == "시각화 보기":
    # 사이드바에서 필터 선택 옵션 추가
    st.sidebar.title("시각화 설정")
    top_n = st.sidebar.slider("상위 항목 수 선택", min_value=1, max_value=100, value=10)

    st.title("종목 추천 시각화")

    # 투자 방법 선택 멀티셀렉트박스 추가
    selected_methods = st.multiselect("투자 방법을 선택하세요", ["회복주", "성장주", "가치주", "배당주", "혼합주"])

    # 선택된 투자 방법에 따라 시각화
    for investment_method in selected_methods:
        if investment_method == "회복주":
            st.subheader("회복주")
            st.write("""
                    ##### 매출 규모, 수익성, 현금 창출력에 비해 주가가 낮게 형성된 저평가 종목
                    """)
            agentmodule.plot_stock_metrics(yh_recovery_data, "회복주", top_n)

            # session_state에서 recovery_df를 불러와 시각화
            if "recovery_df" in st.session_state and not st.session_state["recovery_df"].empty:
                st.subheader("선택한 회복주 종목 시각화")
                agentmodule.plot_stock_metrics(st.session_state["recovery_df"], "선택한 회복주", top_n)

        elif investment_method == "성장주":
            st.subheader("성장주")
            st.write("""
                    ##### 매출 규모, 수익성, 현금 창출력에 비해 시가총액이 낮게 평가된 종목
                    """)
            agentmodule.plot_stock_metrics(yh_growth_data, "성장주", top_n)

            # session_state에서 growth_df를 불러와 시각화
            if "growth_df" in st.session_state and not st.session_state["growth_df"].empty:
                st.subheader("선택한 성장주 종목 시각화")
                agentmodule.plot_stock_metrics(st.session_state["growth_df"], "선택한 성장주", top_n)

        elif investment_method == "가치주":
            st.subheader("가치주")
            st.write("""
                    ##### 자산 가치, 수익성, 현금 창출력에 비해 주가가 크게 저평가된 종목
                    """)
            agentmodule.plot_value_stock_metrics(yj_value_data, "가치주", top_n)

            # session_state에서 value_df를 불러와 시각화
            if "value_df" in st.session_state and not st.session_state["value_df"].empty:
                st.subheader("선택한 배당주 종목 시각화")
                agentmodule.plot_value_stock_metrics(st.session_state["value_df"], "선택한 가치주", top_n)
            
        elif investment_method == "배당주":
            st.subheader("배당주")
            st.write("""
                    ##### 높은 배당을 제공하면서 내재 가치 대비 주가가 낮은 종목
                    """)
            agentmodule.plot_dividend_stock_metrics(yj_dividend_data, "배당주", top_n)

            # session_state에서 divedend_df를 불러와 시각화
            if "divedend_df" in st.session_state and not st.session_state["divedend_df"].empty:
                st.subheader("선택한 가치주 종목 시각화")
                agentmodule.plot_dividend_stock_metrics(st.session_state["divedend_df"], "선택한 배당주", top_n)
            
        elif investment_method == "혼합주":
            st.subheader("혼합주")
            st.write("""
                    ##### 내재 가치 대비 주가가 낮게 평가되었고, 안정적인 배당 수익을 제공하며, 높은 자본 수익성을 보유한 종목
                    """)
            agentmodule.plot_mixed_stock_metrics(sy_mixed_data, "혼합주", top_n)

             # session_state에서 mixed_df를 불러와 시각화
            if "mixed_df" in st.session_state and not st.session_state["mixed_df"].empty:
                st.subheader("선택한 혼합주 종목 시각화")
                agentmodule.plot_mixed_stock_metrics(st.session_state["mixed_df"], "선택한 배당주", top_n)
            


# 대화 화면을 선택한 경우
elif menu_choice == "대화하기":
    # Streamlit UI 설정
    st.title("투자 조언 AI Agent")
    st.write("""
                워렌 버핏, 일론 머스크, 벤자민 그레이엄과 주식 투자를 논의하세요.\n
                "투자 관련 질문"을 받기 위해 이곳에 존재합니다.\n
             """)

    # 대화 기록을 저장할 sesison_state 초기화
    if "messages" not in st.session_state:
        st.session_state["messages"] = []  # 없으면 빈 리스트로 초기화

    # 대화 기록을 업데이트하는 함수 정의
    def update_chat_history():
        chat_html = '<div class="chat-container">'
        for sender, message in st.session_state.messages:
            chat_html += f'<div class="chat-message"><b>{sender}</b>: {message}</div>'
        chat_html += '</div>'
        chat_container.markdown(chat_html, unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .chat-container {
            max-height: 400px;
            overflow-y: auto;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .chat-message {
            margin-bottom: 10px;
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    # 스크롤 가능한 대화 기록 영역을 대화 내역으로 초기화
    initial_chat_html = '<div class="chat-container">'
    for sender, message in st.session_state.messages:
        initial_chat_html += f'<div class="chat-message"><b>{sender}</b>: {message}</div>'
    initial_chat_html += '</div>'
    chat_container = st.markdown(initial_chat_html, unsafe_allow_html=True)

    # 모델 선택 옵션 추가
    model_choice = st.selectbox("대화하고 싶은 상대를 선택하세요", ["Warren Buffett", "Elon Musk", "Benjamin Graham"])

    invest_category = st.selectbox("투자 방법을 선택하세요", ["해당없음", "회복주", "성장주", "가치주", "배당주", "혼합주"])
    additional_system = "" # 방법별 추가적인 코멘트

    # 모델이 변경될 때 입력 필드를 초기화하기 위한 코드
    if "previous_model_choice" not in st.session_state:
        st.session_state.previous_model_choice = model_choice

    if st.session_state.previous_model_choice != model_choice:
        st.session_state["input_text"] = ""  # 모델이 변경되면 입력 필드를 초기화
        st.session_state.previous_model_choice = model_choice

    select_model = ""

    # 특정 투자 방법에 따른 추가 문서 로드 함수
    def load_investment_data(invest_category):
        data = []

        if invest_category == "회복주":
            additional_system = agentmodule.recovery_system
            with open("./stock_data/yh_recovery_stocks.json", "r", encoding="utf-8") as file:
                data = json.load(file)
        elif invest_category == "성장주":
            additional_system = agentmodule.growth_system
            with open("./stock_data/yh_growth_stocks.json", "r", encoding="utf-8") as file:
                data = json.load(file)
        elif invest_category == "가치주":
            additional_system = agentmodule.value_system
            with open("./stock_data/yj_value_stocks.json", "r", encoding="utf-8") as file:
                data = json.load(file)
        elif invest_category == "배당주":
            additional_system = agentmodule.dividend_system
            with open("./stock_data/yj_dividend_stocks.json", "r", encoding="utf-8") as file:
                data = json.load(file)
        elif invest_category == "혼합주":
            additional_system = agentmodule.mixed_system
            with open("./stock_data/sy_result.json", "r", encoding="utf-8") as file:
                data = json.load(file)
        return data, additional_system
    
    # RAG에 추가할 투자 데이터를 Text 형식으로 변환하고 Document로 래핑하는 함수
    def prepare_investment_text(data):
        # JSON 데이터를 텍스트로 변환하고 `page_content` 속성으로 추가
        text_data = "\n".join([f"{item['종목명']} - {item}" for item in data if '종목명' in item])
        return Document(page_content=text_data)  # Document 객체로 반환

    # 가장 유사한 문서를 검색하는 함수
    def retrieve_most_relevant_doc(query, docs):
        query_embedding = embeddings.embed_query(query)
        similarities = [
            (np.dot(np.array(query_embedding), np.array(doc_emb)), content)
            for doc_emb, content in zip(doc_embeddings, [doc.page_content for doc in docs])
        ]
        most_relevant_doc = max(similarities, key=lambda x: x[0])[1]
        return most_relevant_doc

    # 모델에 따라 분기 처리
    if model_choice == "Elon Musk":
        select_model = "Elon Musk"
        txt_path = "./rag_data/Elon_all.txt"
        text_loader = TextLoader(txt_path, encoding="utf-8")
        documents = text_loader.load()

        # 투자 방법에 따른 추가 데이터 로드 및 RAG에 포함
        if invest_category != "해당없음":
            invest_data = load_investment_data(invest_category)
            invest_text = prepare_investment_text(invest_data)
            # 투자 데이터를 문서 리스트에 추가
            documents.append(invest_text)

        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        embedding_path = "./rag_data/embeddings_elon.npy"
        doc_embeddings = np.load(embedding_path, allow_pickle=True)
        system_message = agentmodule.Elon_message
        # additional_system이 비어있지 않다면 system_message에 추가
        if additional_system:
            system_message += "\n\n" + additional_system
    
    elif model_choice == "Warren Buffett":
        select_model = "Warren Buffett"
        pdf_path = "./rag_data/Warren_way.pdf"
        pdf_loader = PyPDFLoader(pdf_path)
        pdf_documents = pdf_loader.load()

        txt_path = "./rag_data/Warren_all.txt"
        text_loader = TextLoader(txt_path, encoding="utf-8")
        txt_documents = text_loader.load()

        documents = pdf_documents + txt_documents

        # 투자 방법에 따른 추가 데이터 로드 및 RAG에 포함
        if invest_category != "해당없음":
            invest_data = load_investment_data(invest_category)
            invest_text = prepare_investment_text(invest_data)
            # 투자 데이터를 문서 리스트에 추가
            documents.append(invest_text)

        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        embedding_path = "./rag_data/embeddings_warren.npy"
        doc_embeddings = np.load(embedding_path, allow_pickle=True)

        system_message = agentmodule.Warren_message
        # additional_system이 비어있지 않다면 system_message에 추가
        if additional_system:
            system_message += "\n\n" + additional_system

    elif model_choice == "Benjamin Graham":
        select_model = "Benjamin Graham"
        pdf_path = "./rag_data/benjamin.pdf"
        pdf_path2 = "./rag_data/benjamin2.pdf"
        pdf_loader = PyPDFLoader(pdf_path)
        pdf_loader2 = PyPDFLoader(pdf_path2)
        pdf_documents = pdf_loader.load()
        pdf_documents2 = pdf_loader2.load()

        txt_path = "./rag_data/Benjamin_all.txt"
        text_loader = TextLoader(txt_path, encoding="utf-8")
        txt_documents = text_loader.load()

        documents = pdf_documents + txt_documents + pdf_documents2

        # 투자 방법에 따른 추가 데이터 로드 및 RAG에 포함
        if invest_category != "해당없음":
            invest_data = load_investment_data(invest_category)
            invest_text = prepare_investment_text(invest_data)
            # 투자 데이터를 문서 리스트에 추가
            documents.append(invest_text)

        text_splitter = CharacterTextSplitter(chunk_size=512, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        embedding_path = "./rag_data/embeddings_warren.npy"
        doc_embeddings = np.load(embedding_path, allow_pickle=True)

        system_message = agentmodule.Benjamin_message
        # additional_system이 비어있지 않다면 system_message에 추가
        if additional_system:
            system_message += "\n\n" + additional_system

    # 사용자 질문 입력
    query_korean = st.text_input("당신의 질문을 입력하세요", key="input_text")

    # 영어로 대화할 때 더 좋은 응답을 생성하므로 영어로 번역해서 모델에 입력
    query = GoogleTranslator(source='ko', target='en').translate(query_korean)

    if query:

        # Vector Store에 내용이 있는지 확인하고, 없으면 빈 스토어로 초기화
        if st.session_state.vector_store is None:
            # 첫 대화가 들어올 때 벡터 스토어를 생성
            st.session_state.vector_store = FAISS.from_texts([query], embeddings)
        else:
            # 이후 대화는 기존 벡터 스토어에 추가
            st.session_state.vector_store.add_texts([query])

        # Vector Store에서 유사한 이전 대화 검색
        similar_docs = st.session_state.vector_store.similarity_search(query, k=3)
        previous_context = "\n".join(doc.page_content for doc in similar_docs)

        # 질문에 대한 가장 유사한 문서 검색
        relevant_doc = retrieve_most_relevant_doc(query, docs)

        # 선택된 모델에 맞는 SystemMessage 및 HumanMessage 생성
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=f"Previous conversations:\n{previous_context}\n\nRelated document:\n{relevant_doc}\n\nCurrent question:\n{query}")
            ]

        # 빈 컨테이너를 만들어 텍스트가 생성되는 동안 UI에 표시
        response_container = st.empty()

        # ChatUpstage 응답 생성 및 스트리밍 방식으로 출력
        response = chat.invoke(messages)
        answer_text = response.content  # 전체 텍스트 내용
        
        # 텍스트를 일정 간격으로 잘라 점진적으로 표시
        answer_text_korean = GoogleTranslator(source='en', target='ko').translate(answer_text)
        answer_lines = answer_text_korean.split(". ")  # 문장 단위로 분리

        # `current_text`를 문자열로 초기화
        current_text = ""

        for line in answer_lines:
            # 현재 문장을 `current_text`에 누적
            current_text += f"{line}. "
            # 컨테이너를 업데이트하여 누적된 텍스트 표시
            response_container.write(current_text)
            time.sleep(0.5)  # 텍스트 표시 간격 조절
        
        # 대화 기록에 사용자와 AI의 메시지를 추가
        st.session_state.messages.append(("User", query_korean))
        st.session_state.messages.append((select_model, answer_text_korean))
        
        # 대화 기록을 업데이트
        update_chat_history()
