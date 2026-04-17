# --- [클라우드 배포용 필수 패치] ---
try:
    __import__('pysqlite3')
    import sys
    if 'pysqlite3' in sys.modules:
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -----------------------------------------------------------

import streamlit as st
from openai import OpenAI

# RAG용 추가 라이브러리
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. 페이지 설정 ---
st.set_page_config(page_title="2026 유한인공지능전공봇", page_icon="🏫", layout="wide")

with st.sidebar:
    st.image("https://ubiquitous.yuhan.ac.kr/images/common/logo.png") 
    st.title("📌 학과 정보")
    st.info("💡 모델 로테이션 작동 중: 429 에러 발생 시 자동으로 다음 AI가 답변합니다.")

st.title("🏫 2026 유한대학교 인공지능전공 AI 안내 봇")

# --- 2. 📍 지도 기능 ---
with st.expander("📍 유한대학교 캠퍼스 지도 확인하기"):
    map_url = "http://googleusercontent.com/maps.google.com/7"
    st.markdown(f'<iframe src="{map_url}" width="100%" height="450" style="border:0; border-radius: 15px;" allowfullscreen="" loading="lazy"></iframe>', unsafe_allow_html=True)

# --- 3. 🧠 벡터 데이터베이스 로드 (RAG) ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory="./yuhan_vector_db", embedding_function=embeddings)

vector_db = load_vector_db()

# --- 4. OpenRouter 설정 및 로테이션 모델 리스트 ---
API_KEY = st.secrets["OPENROUTER_API_KEY"]

@st.cache_resource
def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        default_headers={
            "HTTP-Referer": "https://yuhan-bot.streamlit.app",
            "X-Title": "Yuhan AI Bot Rotation",
        }
    )

client = get_client()

# 순차적으로 시도할 무료 모델 리스트 (현재 작동 확률 높은 순서)
MODELS_TO_TRY = [
    "google/gemma-4-26b-a4b-it:free",
    "google/gemma-4-31b-it:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "meta-llama/llama-guard-4-12b:free"
]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 로테이션 시스템이 적용된 유한대 봇입니다. 무엇을 도와드릴까요?"}]

# --- 5. 채팅 화면 구성 및 로직 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # RAG 검색 (지식 베이스 활용)
        docs = vector_db.similarity_search(prompt, k=2)
        extracted_context = "\n".join([doc.page_content for doc in docs])
        
        bot_reply = ""
        success = False

        # 모델 로테이션 루프 시작
        for model_id in MODELS_TO_TRY:
            try:
                with st.spinner(f'AI 엔진({model_id.split("/")[1]}) 연결 중...'):
                    response = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": f"너는 유한대 전공 안내 봇이야. 아래 정보를 바탕으로 친절하게 답해줘.\n{extracted_context}"},
                            {"role": "user", "content": prompt}
                        ],
                        timeout=15  # 15초 이상 걸리면 다음 모델로 패스
                    )
                    bot_reply = response.choices[0].message.content
                    success = True
                    break  # 성공하면 루프 탈출
            except Exception as e:
                # 에러 발생 시 다음 모델로 넘어가며 경고 표시 (개발 단계에서만 확인)
                continue 

        if not success:
            bot_reply = "죄송합니다. 현재 모든 무료 모델의 한도가 초과되었습니다. 잠시 후 다시 시도해 주세요."

        message_placeholder.markdown(bot_reply)
        st.session_state.messages.append({"role": "assistant", "content": bot_reply})
