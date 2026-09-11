import streamlit as st
from google import genai

st.set_page_config(
    page_title="AI RAG 챗봇",
    page_icon="🤖"
)

st.title("🤖 AI 질문 답변 챗봇")
st.write("궁금한 내용을 질문해보세요.")

# Gemini API 연결
client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 출력
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력
question = st.chat_input("질문을 입력해주세요.")

if question:

    # 사용자 질문 표시
    with st.chat_message("user"):
        st.markdown(question)

    # 사용자 질문 저장
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # Gemini 답변
    with st.chat_message("assistant"):

        try:

            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=question
            )

            answer = response.text

            st.markdown(answer)

            # AI 답변 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            st.error(
                f"AI 응답 중 오류가 발생했습니다: {e}"
            )
