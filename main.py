import streamlit as st
from google import genai

# RAG용 추가 라이브러리 (최신 버전 적용)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

st.set_page_config(page_title="2026 유한인공지능전공봇", page_icon="🏫")
st.title("🏫 2026 유한대학교 인공지능전공 AI 안내 봇")
st.caption("정보가 수집될 수 있으니 개인정보는 입력하지 마세요!!")

# --- 1. 상단 링크 & 지도 ---
col1, col2 = st.columns(2)
with col1:
    st.link_button("📖 학과 안내", "https://ubiquitous.yuhan.ac.kr/ibuilder.do?menu_idx=1329", use_container_width=True)
with col2:
    st.link_button("👨‍🏫 교수진 소개", "https://ubiquitous.yuhan.ac.kr/subject/professorList.do?menu_idx=1323", use_container_width=True)

with st.expander("📍 유한대학교 캠퍼스 지도 확인하기"):
    map_html = """<iframe width="100%" height="400" style="border:0; border-radius: 10px;" loading="lazy" allowfullscreen src="https://maps.google.com/maps?q=유한대학교&t=&z=16&ie=UTF8&iwloc=&output=embed"></iframe>"""
    # [수정됨] Streamlit의 최신 HTML 렌더링 방식을 사용하여 경고문을 해결했습니다.
    st.markdown(map_html, unsafe_allow_html=True)
    
st.divider()

# --- 2. 🧠 벡터 데이터베이스 로드 ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        encode_kwargs={'normalize_embeddings': True}
    )
    # 구워둔 DB 폴더를 불러옵니다.
    db = Chroma(persist_directory="./yuhan_vector_db", embedding_function=embeddings)
    return db

vector_db = load_vector_db()

# --- 3. 캐시 및 API 클라이언트 세팅 ---
@st.cache_resource
def get_qa_cache(): return {}
qa_cache = get_qa_cache()

API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else "여기에_API_키를_입력하세요"
@st.cache_resource
def get_client(): return genai.Client(api_key=API_KEY)
client = get_client()

if "chat_session" not in st.session_state:
    st.session_state.chat_session = client.chats.create(model="gemini-2.5-flash", config={"temperature": 0.5})
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 인공지능전공 봇입니다. 무엇을 도와드릴까요?"}]

# --- 4. 채팅 화면 구성 및 로직 ---
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
            if prompt in qa_cache:
                bot_reply = "⚡ " + qa_cache[prompt]
            else:
                # 🔍 질문과 관련된 정보 조각 찾아오기
                docs = vector_db.similarity_search(prompt, k=1)
                extracted_context = "\n".join([doc.page_content for doc in docs])
                
                rag_prompt = f"[참고]\n{extracted_context}\n\n질문: {prompt}"
                
                response = st.session_state.chat_session.send_message(rag_prompt)
                bot_reply = response.text
                qa_cache[prompt] = bot_reply 
            
            message_placeholder.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")