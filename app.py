import streamlit as st
from openai import OpenAI

st.set_page_config(
    page_title="AI RAG 챗봇",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 AI 질문 답변 챗봇")
st.write("궁금한 내용을 질문해보세요.")

# OpenAI 연결
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 대화 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 질문 입력
question = st.chat_input("질문을 입력해주세요.")

if question:

    # 사용자 질문
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # AI 답변
    with st.chat_message("assistant"):

        try:
            response = client.responses.create(
                model="gpt-5",
                input=question
            )

            answer = response.output_text

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")
