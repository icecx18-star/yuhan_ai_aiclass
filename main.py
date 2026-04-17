# --- [클라우드 배포용 필수 패치] ---
# Streamlit Cloud(리눅스)에서 ChromaDB 실행을 위해 필수적인 설정입니다.
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
from datetime import datetime, timedelta

# RAG용 라이브러리
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. 페이지 설정 및 사이드바 구성 ---
st.set_page_config(page_title="2026 유한인공지능전공봇", page_icon="🏫", layout="wide")

with st.sidebar:
    st.image("https://ubiquitous.yuhan.ac.kr/images/common/logo.png") 
    st.title("📌 학과 주요 링크")
    
    # 학과 공식 링크 복구
    st.link_button("📖 학과 안내 홈페이지", "https://ubiquitous.yuhan.ac.kr/ibuilder.do?menu_idx=1329", use_container_width=True)
    st.link_button("👨‍🏫 교수진 소개", "https://ubiquitous.yuhan.ac.kr/subject/professorList.do?menu_idx=1323", use_container_width=True)
    st.link_button("🏫 유한대학교 포털", "https://portal.yuhan.ac.kr", use_container_width=True)
    
    st.divider()
    
    # 🕒 [보안] 3시간당 15회 제한 로직
    if "first_query_time" not in st.session_state:
        st.session_state.first_query_time = None
    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0

    MAX_QUESTIONS = 15
    WINDOW_HOURS = 3

    # 시간 경과에 따른 초기화 체크
    if st.session_state.first_query_time is not None:
        elapsed = datetime.now() - st.session_state.first_query_time
        if elapsed > timedelta(hours=WINDOW_HOURS):
            st.session_state.first_query_time = None
            st.session_state.usage_count = 0

    # 사용량 표시 UI
    left_q = MAX_QUESTIONS - st.session_state.usage_count
    st.write(f"💬 남은 질문 횟수: **{max(0, left_q)}회**")
    
    if st.session_state.first_query_time:
        next_reset = st.session_state.first_query_time + timedelta(hours=WINDOW_HOURS)
        st.caption(f"⏰ 초기화 예정 시각: {next_reset.strftime('%H:%M:%S')}")
    
    st.progress(max(0.0, min(1.0, st.session_state.usage_count / MAX_QUESTIONS)))
    st.info(f"💡 크레딧 보호를 위해 {WINDOW_HOURS}시간당 {MAX_QUESTIONS}회로 제한됩니다.")

st.title("🏫 2026 유한대학교 인공지능전공 AI 안내 봇")

# --- 2. 📍 지도 기능 (이미지 교체 및 팁 추가) ---
with st.expander("📍 유한대학교 캠퍼스 지도 확인하기"):
    # 업로드한 이미지 파일명과 일치해야 합니다.
    st.image("KakaoTalk_20260417_235353575.png", caption="유한대학교 캠퍼스 전경", use_container_width=True)
    
    st.info("""
    **[ 건물번호 보는 법 ]**
    * **1**204 → **평화관** 2층 204호
    * **6**506 → **창조관** 5층 506호
    * **학과 사무실:** **유일한기념관(7번)** 2층에 위치해 있습니다.
    """)

st.divider()

# --- 3. 🧠 벡터 데이터베이스 로드 (RAG) ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory="./yuhan_vector_db", embedding_function=embeddings)

vector_db = load_vector_db()

# --- 4. OpenRouter API 설정 ---
API_KEY = st.secrets["OPENROUTER_API_KEY"]

@st.cache_resource
def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        default_headers={
            "HTTP-Referer": "https://yuhan-bot.streamlit.app",
            "X-Title": "Yuhan AI Guide Bot Final",
        }
    )

client = get_client()

# [백업 리스트] 유료 우선 + 사용자 기존 무료 모델들
MODELS_TO_TRY = [
    "google/gemini-2.0-flash-001",           # 결제 크레딧 전용 (가장 빠름)
    "google/gemma-4-26b-a4b-it:free",        # 백업 1
    "google/gemma-4-31b-it:free",           # 백업 2
    "qwen/qwen3-next-80b-a3b-instruct:free", # 백업 3
    "meta-llama/llama-guard-4-12b:free"     # 백업 4
]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 유한대 인공지능전공 봇입니다. 궁금한 점을 물어보세요!"}]

# --- 5. 채팅 화면 및 로테이션 로직 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    # 사용 제한 체크
    if st.session_state.usage_count >= MAX_QUESTIONS:
        st.error(f"⚠️ {WINDOW_HOURS}시간 한도를 모두 사용하셨습니다. {next_reset.strftime('%H:%M:%S')} 이후에 다시 시도해 주세요.")
    else:
        # 첫 질문 시각 기록
        if st.session_state.first_query_time is None:
            st.session_state.first_query_time = datetime.now()

        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # RAG 데이터 검색
            docs = vector_db.similarity_search(prompt, k=2)
            context = "\n".join([d.page_content for d in docs])
            
            final_reply = ""
            is_success = False

            # 모델 로테이션 실행
            for model_id in MODELS_TO_TRY:
                try:
                    with st.spinner(f"AI가 생각 중입니다... ({model_id.split('/')[-1]})"):
                        response = client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": f"너는 유한대학교 인공지능전공 안내 봇이야. 아래 정보를 바탕으로 답변해줘.\n{context}"},
                                {"role": "user", "content": prompt}
                            ],
                            timeout=15
                        )
                        final_reply = response.choices[0].message.content
                        is_success = True
                        st.session_state.usage_count += 1
                        break
                except Exception:
                    continue

            if not is_success:
                final_reply = "현재 모든 AI 엔진이 바쁩니다. 잠시 후 다시 질문해 주세요."

            message_placeholder.markdown(final_reply)
            st.session_state.messages.append({"role": "assistant", "content": final_reply})
            st.rerun()
