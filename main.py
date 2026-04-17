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

# 외부 AI(OpenAI) 호출 라이브러리 완전 삭제
# 로컬 검색용 라이브러리만 남김
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
    st.success("💡 현재 봇은 외부 서버 의존 없이 로컬 데이터베이스를 기반으로 작동하여 끊김이 없습니다.")

st.title("🏫 2026 유한대학교 인공지능전공 안내 봇")

# --- 2. 📍 지도 기능 ---
with st.expander("📍 유한대학교 캠퍼스 지도 확인하기"):
    st.image("KakaoTalk_20260417_235353575.png", caption="유한대학교 캠퍼스 맵", use_container_width=True)
    st.info("학과 사무실: 유일한기념관(7번) 2층")

st.divider()

# --- 3. 🧠 벡터 데이터베이스 로드 (외부 인터넷 불필요) ---
@st.cache_resource
def load_vector_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sroberta-multitask",
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(persist_directory="./yuhan_vector_db", embedding_function=embeddings)

vector_db = load_vector_db()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 유한대학교 인공지능전공 관련 정보를 빠르고 정확하게 찾아드립니다. 무엇이 궁금하신가요?"}]

# --- 4. 채팅 화면 구성 ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 5. 스마트 검색 로직 (AI 생성 대신 정확한 문서 매칭) ---
if prompt := st.chat_input("질문을 입력하세요..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # 1. 자주 묻는 질문(FAQ) 즉시 답변 (키워드 매칭)
        if "사무실" in prompt or "전화번호" in prompt or "연락처" in prompt:
            final_reply = "📞 **학과 사무실 정보**\n\n* **위치:** 유일한기념관(7호관) 2층\n* **전화번호:** 02-2610-0783\n* 캠퍼스 지도는 위쪽 탭을 눌러 확인해 주세요!"
        
        elif "장학" in prompt:
            final_reply = "💰 **장학금 안내**\n\n유한대학교 포털 홈페이지에서 [장학 공지사항]을 확인하시는 것이 가장 빠릅니다. 성적 우수 장학금 외에도 다양한 국가/교내 장학금이 있으니 학과 사무실로 문의해 주셔도 좋습니다."
        
        # 2. 벡터 DB 검색 (문서에서 가장 유사한 부분 3개를 찾아 그대로 보여줌)
        else:
            with st.spinner("관련 학과 규정/안내 문서를 검색 중입니다..."):
                docs = vector_db.similarity_search(prompt, k=3)
                
                final_reply = "🔍 **질문하신 내용과 가장 관련 있는 학과 정보입니다:**\n\n"
                
                # 중복 방지를 위해 검색된 텍스트 합치기
                unique_texts = []
                for i, doc in enumerate(docs):
                    text = doc.page_content.strip()
                    if text not in unique_texts:
                        unique_texts.append(text)
                        final_reply += f"> {text}\n\n"
                
                final_reply += "---\n💡 더 자세한 내용은 상단의 학과 홈페이지를 참고하시거나 학과 사무실로 문의해 주세요."

        # 결과 출력
        message_placeholder.markdown(final_reply)
        st.session_state.messages.append({"role": "assistant", "content": final_reply})
