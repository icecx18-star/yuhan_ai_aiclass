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

# --- 1. 페이지 설정 및 사이드바 (링크 및 질문 제한) ---
st.set_page_config(page_title="2026 유한인공지능전공봇", page_icon="🏫", layout="wide")

with st.sidebar:
    st.image("https://ubiquitous.yuhan.ac.kr/images/common/logo.png") 
    st.title("📌 학과 주요 링크")
    
    # 학과 관련 사이트 링크 복구 
    st.link_button("📖 학과 안내 홈페이지", "https://ubiquitous.yuhan.ac.kr/index.do", use_container_width=True)
    st.link_button("👨‍🏫 교수진 소개", "https://ubiquitous.yuhan.ac.kr/subject/professorList.do?menu_idx=1323", use_container_width=True)
    st.link_button("🏫 유한대학교 포털", "https://portal.yuhan.ac.kr", use_container_width=True)
    
    st.divider()
    
    # 💳 질문 횟수 제한 로직
    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0
    
    MAX_QUESTIONS = 15 # 한 세션당 최대 질문 횟수
    left_q = MAX_QUESTIONS - st.session_state.usage_count
    st.write(f"💬 남은 질문 횟수: **{left_q}회** / {MAX_QUESTIONS}회")
    st.progress(max(0, st.session_state.usage_count / MAX_QUESTIONS))
    
    st.info("💡 결제된 유료 엔진이 우선 작동하며, 장애 시 무료 모델로 자동 전환됩니다.")

st.title("🏫 2026 유한대학교 인공지능전공 AI 안내 봇")

# --- 2. 📍 지도 기능 ---
with st.expander("📍 유한대학교 캠퍼스 지도 확인하기"):
    # 유한대 위치 정보 반영 [cite: 2]
    map_url = "http://googleusercontent.com/maps.google.com/7"
    st.markdown(f'<iframe src="{map_url}" width="100%" height="450" style="border:0; border-radius: 15px;" allowfullscreen="" loading="lazy"></iframe>', unsafe_allow_html=True)
    st.caption("학과 사무실: 유일한기념관(7호관) 2층 [cite: 2]")

# --- 3. 🧠 벡터 데이터베이스 로드 (RAG) ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory="./yuhan_vector_db", embedding_function=embeddings)

vector_db = load_vector_db()

# --- 4. OpenRouter 설정 및 사용자 모델 백업 ---
API_KEY = st.secrets["OPENROUTER_API_KEY"]

@st.cache_resource
def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        default_headers={
            "HTTP-Referer": "https://yuhan-bot.streamlit.app",
            "X-Title": "Yuhan AI Bot Paid Version",
        }
    )

client = get_client()

# [백업 완료] 사용자가 직접 설정했던 모델 리스트 + 유료 모델 추가
MODELS_TO_TRY = [
    "google/gemini-2.0-flash-001",           # 결제 크레딧 사용을 위한 최상단 유료 모델
    "google/gemma-4-26b-a4b-it:free",        # 사용자 백업 모델 1
    "google/gemma-4-31b-it:free",           # 사용자 백업 모델 2
    "qwen/qwen3-next-80b-a3b-instruct:free", # 사용자 백업 모델 3
    "meta-llama/llama-guard-4-12b:free"     # 사용자 백업 모델 4
]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 인공지능전공 봇입니다. 무엇을 도와드릴까요?"}]

# --- 5. 채팅 화면 구성 및 로직 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    # 질문 횟수 체크
    if st.session_state.usage_count >= MAX_QUESTIONS:
        st.error(f"⚠️ 사용 한도({MAX_QUESTIONS}회)를 초과했습니다.")
    else:
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            # RAG 검색 (문서 조각 2개 활용) 
            docs = vector_db.similarity_search(prompt, k=2)
            extracted_context = "\n".join([doc.page_content for doc in docs])
            
            bot_reply = ""
            success = False

            # 모델 로테이션 (사용자 백업 리스트 순회)
            for model_id in MODELS_TO_TRY:
                try:
                    with st.spinner(f'AI 엔진({model_id.split("/")[-1]}) 연결 중...'):
                        response = client.chat.completions.create(
                            model=model_id,
                            messages=[
                                {"role": "system", "content": f"너는 유한대 전공 안내 봇이야. 아래 정보를 참고해서 친절하게 답해줘.\n{extracted_context}"},
                                {"role": "user", "content": prompt}
                            ],
                            timeout=15 
                        )
                        bot_reply = response.choices[0].message.content
                        success = True
                        st.session_state.usage_count += 1 # 성공 시에만 카운트 증가
                        break  
                except Exception:
                    continue 

            if not success:
                bot_reply = "현재 모든 모델의 한도가 초과되었거나 응답이 지연되고 있습니다. 잠시 후 다시 시도해 주세요."

            message_placeholder.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            st.rerun()
