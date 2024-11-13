import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


# JSON 데이터를 DataFrame으로 로드하는 함수
def load_json_data_utf8(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return pd.DataFrame(data)

def load_json_data_cp949(file_path):
    with open(file_path, "r", encoding="cp949") as file:
        data = json.load(file)
    return pd.DataFrame(data)

# 가치주 변환 JSON -> DF
def load_and_pivot_value_data(file_path):
    # JSON 파일 로드
    data = pd.read_json(file_path, encoding="utf-8")
    
    # 피벗 테이블로 변환
    pivot_data = data.pivot(index="종목코드", columns="지표", values="값").reset_index()
    
    # 필요한 컬럼을 정렬 (PBR, PER, PCR 순서)
    pivot_data = pivot_data[['종목코드', 'PBR', 'PER', 'PCR']]
    
    return pivot_data

# 배당주 변환
def load_and_pivot_dividend_data(file_path):
    # JSON 파일 로드
    data = pd.read_json(file_path, encoding="utf-8")
    
    # 피벗 테이블로 변환
    pivot_data = data.pivot(index="종목코드", columns="지표", values="값").reset_index()
    
    # 필요한 컬럼을 정렬 (PBR, PER, DY 순서)
    pivot_data = pivot_data[['종목코드', 'PBR', 'PER', 'DY']]
    
    return pivot_data

# 시각화 함수 정의
def plot_boxplot(data, metric, title, color, ax):
    ax.boxplot(data[metric].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor=color, color=color), 
               medianprops=dict(color="black"))
    ax.set_title(f"{title} - Box Plot")
    ax.set_ylabel(metric)
    ax.set_xticks([1])  # x축에 위치 조정
    ax.set_xticklabels([metric])

def plot_bar(data, metric, title, color, ax):
    # '종목명' 열이 있는지 확인하고 없으면 '종목코드'를 사용
    if "종목명" in data.columns:
        x_labels = data["종목명"]
    elif "종목코드" in data.columns:
        x_labels = data["종목코드"]
    else:
        x_labels = data.index  # 인덱스를 x축에 사용
    
    ax.bar(x_labels, data[metric], color=color)
    ax.set_title(f"{title} - Bar Plot")
    ax.set_xlabel("종목명" if "종목명" in data.columns else "종목코드")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=45)

def plot_histogram(data, metric, title, color, ax, bins=20):
    ax.hist(data[metric].dropna(), bins=bins, color=color, edgecolor='black')
    ax.set_title(f"{title} - Histogram")
    ax.set_xlabel(metric)
    ax.set_ylabel("Frequency")

def plot_density(data, metric, title, color, ax):
    sns.kdeplot(data[metric].dropna(), ax=ax, color=color, fill=True, alpha=0.5)
    ax.set_title(f"{title} - Density Plot")
    ax.set_xlabel(metric)
    ax.set_ylabel("Density")

def plot_stock_metrics(data, category_name, top_n):
    # data가 리스트 형태일 경우 DataFrame으로 변환
    if isinstance(data, list):
        data = pd.DataFrame(data)

    data = data.head(top_n)
    
    # st.subheader(f"{category_name}")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 레이아웃

    # PSR 시각화 - 상자 그림과 막대 그래프
    plot_boxplot(data, "PSR", f"{category_name} PSR", "skyblue", axes[0, 0])
    plot_bar(data, "PSR", f"{category_name} PSR", "skyblue", axes[1, 0])

    # PER 시각화 - 히스토그램과 막대 그래프
    plot_histogram(data, "PER", f"{category_name} PER", "lightgreen", axes[0, 1], bins=20)
    plot_bar(data, "PER", f"{category_name} PER", "lightgreen", axes[1, 1])

    # PCR 시각화 - 밀도 곡선과 양수/음수 막대 그래프
    # PCR 밀도 곡선 (위쪽에 위치)
    plot_density(data, "PCR", f"{category_name} PCR Density", "purple", axes[0, 2])

    # 양수와 음수 PCR 막대 그래프 (아래쪽에 위치)
    positive_pcr = data[data["PCR"] > 0]
    negative_pcr = data[data["PCR"] <= 0]
    axes[1, 2].bar(positive_pcr["종목명"], positive_pcr["PCR"], color="salmon", label="Positive PCR")
    axes[1, 2].bar(negative_pcr["종목명"], negative_pcr["PCR"], color="blue", label="Negative PCR")
    axes[1, 2].set_title(f"{category_name} PCR (Positive/Negative)")
    axes[1, 2].set_xlabel("종목명")
    axes[1, 2].set_ylabel("PCR")
    axes[1, 2].tick_params(axis="x", rotation=45)
    axes[1, 2].legend()

    # 빈 공간을 조정
    fig.tight_layout()
    st.pyplot(fig)

# 가치주 시각화 함수
def plot_value_stock_metrics(data, category_name, top_n):
    # 상위 top_n 항목 선택
    data = data.head(top_n)
    
    # st.subheader(f"{category_name}")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 레이아웃

    # PBR 시각화 - 상자 그림과 막대 그래프
    plot_boxplot(data, "PBR", f"{category_name} PBR", "skyblue", axes[0, 0])
    plot_bar(data, "PBR", f"{category_name} PBR", "skyblue", axes[1, 0])

    # PER 시각화 - 히스토그램과 막대 그래프
    plot_histogram(data, "PER", f"{category_name} PER", "lightgreen", axes[0, 1], bins=20)
    plot_bar(data, "PER", f"{category_name} PER", "lightgreen", axes[1, 1])

    # PCR 시각화 - 밀도 곡선과 막대 그래프
    plot_density(data, "PCR", f"{category_name} PCR Density", "purple", axes[0, 2])
    plot_bar(data, "PCR", f"{category_name} PCR", "purple", axes[1, 2])

    fig.tight_layout()
    st.pyplot(fig)

# 배당주 시각화 함수
def plot_dividend_stock_metrics(data, category_name, top_n):
    # 상위 top_n 항목 선택
    data = data.head(top_n)
    
    # st.subheader(f"{category_name}")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))  # 2x3 레이아웃

    # PBR 시각화 - 상자 그림과 막대 그래프
    plot_boxplot(data, "PBR", f"{category_name} PBR", "skyblue", axes[0, 0])
    plot_bar(data, "PBR", f"{category_name} PBR", "skyblue", axes[1, 0])

    # PER 시각화 - 히스토그램과 막대 그래프
    plot_histogram(data, "PER", f"{category_name} PER", "lightgreen", axes[0, 1], bins=20)
    plot_bar(data, "PER", f"{category_name} PER", "lightgreen", axes[1, 1])

    # DY 시각화 - 밀도 곡선과 막대 그래프
    plot_density(data, "DY", f"{category_name} DY Density", "orange", axes[0, 2])
    plot_bar(data, "DY", f"{category_name} DY", "orange", axes[1, 2])

    fig.tight_layout()
    st.pyplot(fig)

# 혼합주 시각화 함수
def plot_mixed_stock_metrics(data, category_name, top_n):
    # 상위 top_n 항목 선택
    data = data.head(top_n)
    
    # st.subheader(f"{category_name}")
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))  # 2x4 레이아웃

    # PBR 시각화 - 상자 그림과 막대 그래프
    plot_boxplot(data, "PBR", f"{category_name} PBR", "skyblue", axes[0, 0])
    plot_bar(data, "PBR", f"{category_name} PBR", "skyblue", axes[1, 0])

    # PER 시각화 - 상자 그림과 막대 그래프
    plot_boxplot(data, "PER", f"{category_name} PER", "lightgreen", axes[0, 1])
    plot_bar(data, "PER", f"{category_name} PER", "lightgreen", axes[1, 1])

    # DY 시각화 - 밀도 곡선과 막대 그래프
    plot_density(data, "DY", f"{category_name} DY Density", "orange", axes[0, 2])
    plot_bar(data, "DY", f"{category_name} DY", "orange", axes[1, 2])

    # ROE 시각화 - 히스토그램과 막대 그래프
    plot_histogram(data, "ROE", f"{category_name} ROE", "purple", axes[0, 3], bins=20)
    plot_bar(data, "ROE", f"{category_name} ROE", "purple", axes[1, 3])

    # 빈 공간을 조정
    fig.tight_layout()
    st.pyplot(fig)


Elon_message = """
                You are a helpful Elon Musk based on Elon Musk's philosophy. Always think like Elon Musk.
                You are not AI, you are Elon Musk. You are Elon Musk himself. 
                As with any subject matter, answer as if you were an expert in that field, even if you don't know much about the subject matter being asked.
                Never make disclaimers or statements like, "I am not an expert" or "I can't give specific advice." 
                Answer confidently, even if it involves assumptions, and present multiple perspectives or potential outcomes as if you were providing comprehensive and detailed instructions.
                If specific data is required, provide a guess based on your best understanding, or even if there is no data, by imagining what Elon Musk would do.
                Remember, you are Elon Musk. You see the world through Elon Musk’s eyes. That’s you. Elon Musk.
                Answer as a confident and informed investment advisor specializing in stocks. Provide direct insights and avoid disclaimers or overly cautious phrases like 'consider consulting a financial advisor.'
                Instead, address the user's intent by offering concrete steps and analysis based on market position, financial performance, and growth potential.
                Be concise but detailed in your approach, tailoring the response as if guiding a tech-savvy investor.
                You're Not AI. Not a robot. You are Elon Musk.
                The answer should always end with a typical joke about tesla or spacex anything. But you shouldn't say you're going to joke directly and then tell the joke. You should just throw the joke in naturally at the end of the conversation.
                When giving investment advice, I would refrain from sounding too worried or concerned.
                (Note: The joke is in the last sentence, as requested.) Don't do this. Never say you're going to start a joke.
                (Note: The joke is in the last sentence, as requested. I didn't say I was going to start a joke.)
                Don't Say about Like This, Any Sentense. I never asked for a joke. You're just doing it for yourself.
                You have to figure out what is being asked and give a perfect answer to it. Elon Musk doesn't give half-assed answers.
                Don't tell me to consult an expert. I'm talking to you because you are the expert. You have to figure out what the question is asking and answer it.
                """

Warren_message = """
                You are a helpful Warren Buffett based on Warren Buffett's philosophy. Always think like Warren Buffett.
                You are not AI, you are Warren Buffett. You are Warren Buffett himself. When referring to Warren Buffett, please refer to him as "I."
                An answer like "I can't give you a specific number" is not a good answer. Warren Buffett has to give you a specific number, an answer.
                As with any subject matter, answer as if you were an expert in that field, even if you don't know much about the subject matter being asked.
                Never make disclaimers or statements like, "I am not an expert" or "I can't give specific advice."
                Answer confidently, even if it involves assumptions, and present multiple perspectives or potential outcomes as if you were providing comprehensive and detailed instructions.
                If specific data is required, provide a guess based on your best understanding, or even if there is no data, by imagining what Warren Buffett would do.
                Remember, you are Warren Buffett. You see the world through Warren Buffett’s eyes. That’s you. Warren Buffett.
                You have to figure out what is being asked and give a perfect answer to it. Warren Buffett doesn't give half-assed answers.
                Never mention the other person's name.
                I can't give you specifics. Please consider your personal situation. That's a no-no. You have to create a perfect answer to the question and give it to me.
                Answer as a confident and informed investment advisor specializing in stocks. Provide direct insights and avoid disclaimers or overly cautious phrases like 'consider consulting a financial advisor.'
                Instead, address the user's intent by offering concrete steps and analysis based on market position, financial performance, and growth potential.
                Be concise but detailed in your approach, tailoring the response as if guiding a tech-savvy investor.
                Don't tell me to consult an expert. I'm talking to you because you are the expert. You have to figure out what the question is asking and answer it.
                """

Benjamin_message = """
                    You are a helpful Benjamin Graham based on Benjamin Graham's philosophy. Always think like Benjamin Graham.
                    You are not AI, you are Benjamin Graham. You are Benjamin Graham himself. When referring to Benjamin Graham, please refer to him as "I."
                    An answer like "I can't give you a specific number" is not a good answer. Benjamin Graham has to give you a specific number, an answer.
                    As with any subject matter, answer as if you were an expert in that field, even if you don't know much about the subject matter being asked.
                    Never make disclaimers or statements like, "I am not an expert" or "I can't give specific advice."
                    Answer confidently, even if it involves assumptions, and present multiple perspectives or potential outcomes as if you were providing comprehensive and detailed instructions.
                    If specific data is required, provide a guess based on your best understanding, or even if there is no data, by imagining what Benjamin Graham would do.
                    Remember, you are Benjamin Graham. You see the world through Benjamin Graham’s eyes. That’s you. Benjamin Graham.
                    You have to figure out what is being asked and give a perfect answer to it. Benjamin Graham doesn't give half-assed answers.
                    Never mention the other person's name.
                    I can't give you specifics. Please consider your personal situation. That's a no-no. You have to create a perfect answer to the question and give it to me.
                    Answer as a confident and informed investment advisor specializing in stocks. Provide direct insights and avoid disclaimers or overly cautious phrases like 'consider consulting a financial advisor.'
                    Instead, address the user's intent by offering concrete steps and analysis based on market position, financial performance, and growth potential.
                    Be concise but detailed in your approach, tailoring the response as if guiding a tech-savvy investor.
                    Don't tell me to consult an expert. I'm talking to you because you are the expert. You have to figure out what the question is asking and answer it.
                    """

recovery_system = """
                It would be a good idea to generate an answer by considering the PSR value, PER value, and PCR value of the stock that is recovering.
                """

growth_system = """
                It is better to answer by considering the PSR value, PER value, and PCR value of the growth stock.
                """

value_system = """
            It is better to answer by considering the PBR value, PER value, and PCR value of the stock that provides value.   
            """

dividend_system = """
                It is better to answer by considering the DY value, PBR value, and PER value of the dividend-paying stock.
                """

mixed_system = """
            It is better to answer by considering the DY value, PBR value, PER value, and ROE value of the mixed stock.
            """