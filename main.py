import streamlit as st
from google import genai

# 웹 페이지 탭 설정
st.set_page_config(page_title="2026 유한봇", page_icon="🏫")

st.title("🏫 2026 유한대학교 AI 안내 봇")
st.caption("연결 끊김(Client Closed) 문제를 완벽히 해결한 버전입니다.")

# Streamlit의 안전한 금고(Secrets)에서 키를 꺼내오도록 변경합니다.
API_KEY = st.secrets["GEMINI_API_KEY"]

# ---------------------------------------------------------
# [핵심 변경 사항] API 클라이언트를 한 번만 생성하고 캐싱(유지)합니다.
@st.cache_resource
def get_client():
    return genai.Client(api_key=API_KEY)

client = get_client()
# ---------------------------------------------------------

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
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 저는 유한봇입니다. 2026학년도 대학생활에 대해 무엇이든 물어보세요!"}]

# 기존 대화 내용 화면에 그리기
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요... (예: 장학금 기준이 뭐야?)"):
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