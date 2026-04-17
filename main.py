# --- [클라우드 배포용 필수 패치] 반드시 파일 맨 위에 있어야 합니다! ---
try:
    __import__('pysqlite3')
    import sys
    if 'pysqlite3' in sys.modules:
        sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
# -----------------------------------------------------------

import streamlit as st
from openai import OpenAI  # OpenRouter 연동용

# RAG용 추가 라이브러리
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- 1. 페이지 설정 및 디자인 ---
st.set_page_config(page_title="2026 유한인공지능전공봇", page_icon="🏫", layout="wide")

# 사이드바 구성
with st.sidebar:
    st.image("https://ubiquitous.yuhan.ac.kr/images/common/logo.png") 
    st.title("📌 학과 정보")
    st.link_button("📖 학과 안내", "https://ubiquitous.yuhan.ac.kr/ibuilder.do?menu_idx=1329", use_container_width=True)
    st.link_button("👨‍🏫 교수진 소개", "https://ubiquitous.yuhan.ac.kr/subject/professorList.do?menu_idx=1323", use_container_width=True)
    st.divider()
    st.info("💡 OpenRouter 서버를 통해 24시간 안정적으로 운영됩니다.")

st.title("🏫 2026 유한대학교 인공지능전공 AI 안내 봇")
st.caption("실시간 지도와 오픈소스 모델이 결합된 최신 버전입니다 ⚡")

# --- 2. 📍 지도 기능 (유한대학교 실제 위치 연결) ---
with st.expander("📍 유한대학교 캠퍼스 지도 확인하기"):
    # 유한대학교 실제 위경도 기반 구글 지도 임베드 코드
    map_url = "https://www.google.com/maps/embed?pb=!1m18!1m12!1m3!1d3165.1741544600124!2d126.81745487643501!3d37.48731302888151!2m3!1f0!2f0!3f0!3m2!1i1024!2i768!4f13.1!3m3!1m2!1s0x357b63973347b973%3A0xc647087612f0088e!2z7Jyg7ZWc64yA7ZWZ6rWQ!5e0!3m2!1sko!2skr!4v1713429000000!5m2!1sko!2skr"
    
    map_html = f"""
    <iframe 
        width="100%" 
        height="450" 
        style="border:0; border-radius: 15px;" 
        src="{map_url}" 
        allowfullscreen="" 
        loading="lazy" 
        referrerpolicy="no-referrer-when-downgrade">
    </iframe>
    """
    st.markdown(map_html, unsafe_allow_html=True)
    st.info("💡 학과 사무실은 유일한기념관(7호관) 2층에 위치해 있습니다.")

st.divider()

# --- 3. 🧠 벡터 데이터베이스 로드 (RAG) ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        encode_kwargs={'normalize_embeddings': True}
    )
    db = Chroma(persist_directory="./yuhan_vector_db", embedding_function=embeddings)
    return db

vector_db = load_vector_db()

# --- 4. OpenRouter API 클라이언트 세팅 ---
API_KEY = st.secrets["OPENROUTER_API_KEY"] if "OPENROUTER_API_KEY" in st.secrets else "YOUR_KEY_HERE"

@st.cache_resource
def get_client():
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
        default_headers={
            "HTTP-Referer": "https://yuhan-bot.streamlit.app",
            "X-Title": "Yuhan AI Bot",
        }
    )

client = get_client()

# 안정적인 최신 무료 모델 적용
MODEL_NAME = "google/gemma-4-26b-a4b-it:free"

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 유한대학교 인공지능전공 봇입니다. 무엇을 도와드릴까요?"}]

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
        try:
            # 1. RAG 검색 (조각 2개 검색으로 정확도 향상)
            docs = vector_db.similarity_search(prompt, k=1)
            extracted_context = "\n".join([doc.page_content for doc in docs])
            
            # 2. OpenRouter 답변 생성
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": f"너는 유한대학교 인공지능전공 봇이야. 아래 [참고 정보]만을 활용해서 질문에 친절하게 답변해줘.\n\n[참고 정보]\n{extracted_context}"},
                    {"role": "user", "content": prompt}
                ]
            )
            bot_reply = response.choices[0].message.content
            
            message_placeholder.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")