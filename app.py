import streamlit as st
from google import genai

st.set_page_config(
    page_title="나의 AI 챗봇",
    page_icon="🤖"
)

st.title("🤖 나의 AI 챗봇")
st.write("내 정보와 문서를 바탕으로 질문에 답해주는 AI입니다.")

# =========================
# Gemini 연결
# =========================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# =========================
# 개인 프로필
# =========================

profile = {
    "이름": "홍길동",
    "전화번호": "010-1234-5678",
    "주소": "서울특별시 강남구 테헤란로 123",
    "가족관계": {
        "아버지": "홍아버지",
        "어머니": "김어머니",
        "동생": "홍동생"
    }
}

# =========================
# 대화 기록
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================
# 질문
# =========================

question = st.chat_input(
    "내 정보에 대해 질문해보세요."
)

if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # =========================
    # AI에게 전달할 프롬프트
    # =========================

    prompt = f"""
너는 사용자의 개인 정보를 관리하는 AI 챗봇이다.

아래에 등록된 사용자 정보를 참고해서 질문에 답변한다.

[사용자 정보]
이름: {profile["이름"]}
전화번호: {profile["전화번호"]}
주소: {profile["주소"]}

가족관계:
아버지: {profile["가족관계"]["아버지"]}
어머니: {profile["가족관계"]["어머니"]}
동생: {profile["가족관계"]["동생"]}

[중요한 규칙]
1. 위에 제공된 정보만 사용한다.
2. 등록되지 않은 개인정보를 추측하지 않는다.
3. 질문에 대한 정보가 없으면
   "등록된 정보에서 찾을 수 없습니다."
   라고 답한다.
4. 사용자가 물어본 정보만 간단하고 명확하게 답한다.
5. 한국어로 답변한다.

사용자의 질문:
{question}
"""

    with st.chat_message("assistant"):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            answer = response.text

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            st.error(
                f"AI 응답 중 오류가 발생했습니다: {e}"
            )
