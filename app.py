import streamlit as st
from google import genai
import PyPDF2
from docx import Document

st.set_page_config(
    page_title="AI RAG 챗봇",
    page_icon="🤖"
)

st.title("🤖 AI RAG 챗봇")
st.write("PDF 또는 Word 문서를 업로드하고 질문해보세요.")

# =========================
# Gemini API 연결
# =========================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)

# =========================
# 문서 업로드
# =========================

st.sidebar.header("📄 문서 업로드")

uploaded_file = st.sidebar.file_uploader(
    "PDF 또는 DOCX 파일을 선택하세요.",
    type=["pdf", "docx"]
)

# =========================
# 문서 내용 추출
# =========================

document_text = ""

if uploaded_file is not None:

    # PDF
    if uploaded_file.name.endswith(".pdf"):

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        for page in pdf_reader.pages:
            text = page.extract_text()

            if text:
                document_text += text + "\n"

    # DOCX
    elif uploaded_file.name.endswith(".docx"):

        doc = Document(uploaded_file)

        for paragraph in doc.paragraphs:
            document_text += paragraph.text + "\n"

    st.sidebar.success(
        f"✅ {uploaded_file.name} 업로드 완료"
    )

    st.sidebar.write(
        f"문서 글자 수: {len(document_text):,}"
    )


# =========================
# 대화 기록
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


# 이전 대화 표시
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# 질문 입력
# =========================

question = st.chat_input(
    "문서에 대해 질문해주세요."
)


if question:

    # 사용자 질문 표시
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # =========================
    # Gemini 답변
    # =========================

    with st.chat_message("assistant"):

        try:

            if document_text:

                prompt = f"""
다음 문서 내용을 참고해서 사용자의 질문에 답변하세요.

문서 내용:
----------------
{document_text}
----------------

사용자 질문:
{question}

답변 규칙:
1. 반드시 문서 내용을 우선적으로 참고하세요.
2. 문서에 없는 내용은 추측하지 마세요.
3. 문서에서 답을 찾을 수 없다면
   "문서에서 해당 내용을 찾을 수 없습니다."
   라고 답변하세요.
4. 답변은 이해하기 쉽게 한국어로 작성하세요.
"""

            else:

                prompt = f"""
사용자의 질문에 친절하게 답변하세요.

질문:
{question}
"""


            response = client.models.generate_content(
                model="gemini-3.8-flash",
                contents=prompt
            )

            answer = response.text

            st.markdown(answer)


            # 답변 저장
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


        except Exception as e:

            st.error(
                f"AI 응답 중 오류가 발생했습니다: {e}"
            )
