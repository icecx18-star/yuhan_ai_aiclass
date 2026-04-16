import streamlit as st
from google import genai
import streamlit.components.v1 as components
import folium
from streamlit_folium import st_folium


# 웹 페이지 탭 설정
st.set_page_config(page_title="2026 유한인공지능전공봇", page_icon="🏫")

st.title("🏫 2026 유한인공지능전공 AI 안내 봇")
st.caption("베타 테스트 기간입니다.\n 본인의 학번 학과등 개인정보는 입력하지 마세요 학습 중이라 개인정보가 수집될 수 있습니다!!")


# ==========================================
# 🔗 [추가된 부분] 상단 바로가기 링크 버튼 영역
# ==========================================
col1, col2 = st.columns(2) # 화면을 정확히 반(1:1)으로 나눕니다.

with col1:
    # use_container_width=True를 주면 버튼이 화면 반쪽을 꽉 채우게 예쁘게 늘어납니다.
    st.link_button("📖 학과 안내", "https://ubiquitous.yuhan.ac.kr/ibuilder.do?menu_idx=1329", use_container_width=True)

with col2:
    st.link_button("👨‍🏫 교수진 소개", "https://ubiquitous.yuhan.ac.kr/subject/professorList.do?menu_idx=1323", use_container_width=True)

st.divider() # 버튼과 채팅창을 시각적으로 분리해주는 얇은 회색 선을 긋습니다.
# ==========================================

# 주의: 깃허브 배포 시에는 st.secrets["GEMINI_API_KEY"] 방식을 사용하세요.
API_KEY = "여기에_API_키를_입력하세요" 

# API 클라이언트 캐싱 (연결 끊김 에러 방지)
@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)

client = get_client()

# 대화 기록 저장소 및 채팅 세션 초기화
if "chat_session" not in st.session_state:
    system_instruction = """
너는 유한대학교의 친절한 AI 조교 '유한봇'이야.
학생들의 질문에 친절하고 정확하게 답변해줘.
2026학년도 대학생활 안내를 기준으로 답변하며, 모르는 내용은 학교 홈페이지를 참고하라고 안내해.
"""
    st.session_state.chat_session = client.chats.create(
        model="gemini-2.5-flash",
        config={
            "system_instruction": system_instruction,
            "temperature": 0.7
        }
    )
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 저는 유한봇입니다. 상단의 메뉴를 클릭하거나 궁금한 점을 질문해 주세요!"}]

# 기존 대화 내용 화면에 그리기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요... (예: 과방 위치가 어디야?)"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 답변 생성 및 화면에 출력
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            response = st.session_state.chat_session.send_message(prompt)
            bot_reply = response.text
            
            message_placeholder.markdown(bot_reply)
            st.session_state.messages.append({"role": "assistant", "content": bot_reply})
            
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
