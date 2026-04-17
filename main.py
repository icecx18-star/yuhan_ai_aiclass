import streamlit as st
from google import genai
import streamlit.components.v1 as components

# 웹 페이지 설정
st.set_page_config(page_title="2026 유한인공지능전공봇", page_icon="🏫")

st.title("🏫 2026 유한대 인공지능전공 AI 챗봇")
st.caption("아직 베타버전입니다. ai모델이 학습중이므로 본인의 개인정보(학번, 주민등록번호, 전화번호, 이름)은 입력하지 마세요!!")

# --- 1. 상단 바로가기 링크 ---
col1, col2, col3 = st.columns(3)
with col1:
    st.link_button("📖 학과 안내", "https://ubiquitous.yuhan.ac.kr/ibuilder.do?menu_idx=1329", use_container_width=True)
with col2:
    st.link_button("👨‍🏫 교수진 소개", "https://ubiquitous.yuhan.ac.kr/subject/professorList.do?menu_idx=1323", use_container_width=True)
with col3:
    st.link_button("🧠 전공 홈페이지", "https://ubiquitous.yuhan.ac.kr/index.do", use_container_width=True)


# --- 2. [업데이트] 캠퍼스 실제 지도 (구글 맵 연동) ---
with st.expander("📍 유한대학교 캠퍼스 지도 확인하기"):
    # 유한대학교의 실제 좌표 및 검색어가 포함된 구글 지도 iframe 코드입니다.
    map_html = """
    <iframe width="100%" height="400" style="border:0; border-radius: 10px;" loading="lazy" allowfullscreen
    src="https://maps.google.com/maps?q=유한대학교&t=&z=16&ie=UTF8&iwloc=&output=embed">
    </iframe>
    """
    components.html(map_html, height=400)

st.divider()

# --- 3. 🧠 답변 기억 창고 (Cache) ---
@st.cache_resource
def get_qa_cache():
    return {}

qa_cache = get_qa_cache()

# --- 4. 지식 창고 로드 (yuhan_info.txt 파일 읽기) ---
@st.cache_data
def load_knowledge_base():
    try:
        with open("yuhan_info.txt", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "현재 수집된 상세 학교 정보가 없습니다."

knowledge_base = load_knowledge_base()

# API 키 및 클라이언트 설정
API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else "여기에_API_키를_입력하세요"

@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)

client = get_client()

# 대화 세션 및 시스템 프롬프트 초기화
if "chat_session" not in st.session_state:
    system_instruction = f"""
너는 유한대 인공지능전공 AI 조교 '유한인공지능전공봇'이야. 
아래 제공된 [유한대학교 캠퍼스 정보]를 바탕으로 학생들에게 친절하고 정확하게 답변해줘. 
문서에 없는 내용은 "제가 아직 모르는 내용입니다. 학교 홈페이지나 학과 사무실에 문의해 주세요."라고 정중하게 안내해.

[유한대학교 캠퍼스 정보]
{knowledge_base}
"""
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.0-flash",
        config={"system_instruction": system_instruction, "temperature": 0.5}
    )
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 저는 유한인공지능전공봇입니다. 궁금한 점이 있다면 물어봐주세요!"}]

# 기존 대화 내용 출력
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요... (베타버전이므로 오류가 발생할 수 있습니다.)"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            # 캐시(기억) 검사
            if prompt in qa_cache:
                bot_reply = "⚡ " + qa_cache[prompt]
            else:
                response = st.session_state.chat_session.send_message(prompt)
                bot_reply = response.text
                qa_cache[prompt] = bot_reply 
            
            message_placeholder.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
