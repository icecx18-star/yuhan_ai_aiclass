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

# RAG용 라이브러리
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. 페이지 설정 및 사이드바 ---
st.set_page_config(page_title="2026 유한인공지능전공봇", page_icon="🏫", layout="wide")

with st.sidebar:
    st.image("https://ubiquitous.yuhan.ac.kr/images/common/logo.png") 
    st.title("📌 학과 주요 링크")
    
    st.link_button("📖 학과 안내 홈페이지", "https://ubiquitous.yuhan.ac.kr/ibuilder.do?menu_idx=1329", use_container_width=True)
    st.link_button("👨‍🏫 교수진 소개", "https://ubiquitous.yuhan.ac.kr/subject/professorList.do?menu_idx=1323", use_container_width=True)
    st.link_button("🏫 유한대학교 포털", "https://portal.yuhan.ac.kr", use_container_width=True)
    
    st.divider()
    
    # 💳 단순 질문 횟수 제한 (세션당 15회)
    if "usage_count" not in st.session_state:
        st.session_state.usage_count = 0

    MAX_QUESTIONS = 15
    left_q = MAX_QUESTIONS - st.session_state.usage_count
    
    st.write(f"💬 남은 질문 횟수: **{max(0, left_q)}회**")
    st.progress(max(0.0, min(1.0, st.session_state.usage_count / MAX_QUESTIONS)))
    st.info(f"💡 브라우저를 새로고침하면 횟수가 초기화됩니다.")

st.title("🏫 2026 유한대학교 인공지능전공 AI 안내 봇")

# --- 2. 📍 지도 기능 (업로드한 이미지 사용) ---
with st.expander("📍 유한대학교 캠퍼스 지도 확인하기"):
    # 파일명이 GitHub에 올린 이미지와 정확히 일치해야 합니다.
    st.image("KakaoTalk_20260417_235353575.png", caption="유한대학교 캠퍼스 맵", use_container_width=True)
    st.info("학과 사무실: 유일한기념관(7번) 2층")

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
            "X-Title": "Yuhan AI Bot Final Stable",
        }
    )

client = get_client()

# [백업 완료] 작동이 확인된 모델 리스트
MODELS_TO_TRY = [
    "google/gemini-2.0-flash-001",           # 1순위: 유료 결제 전용 (빠름/안정적)
    "meta-llama/llama-3.1-8b-instruct:free", # 2순위: 안정적인 무료 모델
    "qwen/qwen-2.5-72b-instruct:free",       # 3순위: 한국어 특화
    "google/gemini-flash-1.5-8b:free",       # 4순위: 속도 위주
    "mistralai/pixtral-12b:free"             # 5순위: 백업
]

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 유한대학교 인공지능전공 봇입니다. 무엇을 도와드릴까요?"}]

# --- 5. 채팅 화면 구성 및 로직 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("질문을 입력하세요..."):
    # 횟수 제한 체크
    if st.session_state.usage_count >= MAX_QUESTIONS:
        st.error(f"⚠️ 오늘의 질문 한도({MAX_QUESTIONS}회)를 모두 사용하셨습니다. 새로고침 후 다시 이용해 주세요.")
    else:
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
                    with st.spinner(f"AI 응답 생성 중... ({model_id.split('/')[-1]})"):
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
                except Exception as e:
                    # 무엇이 문제인지 개발자에게만 살짝 알림
                    st.toast(f"⚠️ {model_id.split('/')[-1]} 연결 실패")
                    continue

            if not is_success:
                final_reply = "현재 모든 AI 서버가 응답하지 않습니다. 잠시 후 다시 질문해 주세요."

            message_placeholder.markdown(final_reply)
            st.session_state.messages.append({"role": "assistant", "content": final_reply})
            st.rerun()
