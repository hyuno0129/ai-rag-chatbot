import streamlit as st
from google import genai
import PyPDF2
from docx import Document

from profile_data import PROFILE


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="나의 AI RAG 챗봇",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 나의 AI RAG 챗봇")
st.write("개인 정보와 업로드한 문서를 바탕으로 질문에 답해주는 AI 챗봇입니다.")


# =========================================================
# Gemini 연결
# =========================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# =========================================================
# 사이드바 - 문서 업로드
# =========================================================

st.sidebar.header("📄 문서 업로드")

uploaded_file = st.sidebar.file_uploader(
    "PDF 또는 Word 파일을 선택하세요.",
    type=["pdf", "docx"]
)


# =========================================================
# 문서 내용 추출
# =========================================================

document_text = ""


if uploaded_file is not None:

    try:

        # -------------------------
        # PDF
        # -------------------------

        if uploaded_file.name.lower().endswith(".pdf"):

            pdf_reader = PyPDF2.PdfReader(uploaded_file)

            for page in pdf_reader.pages:

                text = page.extract_text()

                if text:
                    document_text += text + "\n"


        # -------------------------
        # DOCX
        # -------------------------

        elif uploaded_file.name.lower().endswith(".docx"):

            doc = Document(uploaded_file)

            for paragraph in doc.paragraphs:

                if paragraph.text.strip():
                    document_text += paragraph.text + "\n"


        # -------------------------
        # 업로드 결과
        # -------------------------

        if document_text.strip():

            st.sidebar.success(
                f"✅ {uploaded_file.name} 업로드 완료"
            )

            st.sidebar.write(
                f"문서 글자 수: {len(document_text):,}"
            )

        else:

            st.sidebar.warning(
                "⚠️ 문서에서 텍스트를 찾을 수 없습니다."
            )


    except Exception as e:

        st.sidebar.error(
            f"문서를 읽는 중 오류가 발생했습니다: {e}"
        )


# =========================================================
# 개인 정보 질문인지 확인
# =========================================================

def is_profile_question(question):

    profile_keywords = [
        "이름",
        "성함",
        "전화번호",
        "핸드폰",
        "휴대폰",
        "연락처",
        "주소",
        "사는 곳",
        "가족",
        "아버지",
        "어머니",
        "엄마",
        "아빠",
        "동생"
    ]

    for keyword in profile_keywords:

        if keyword in question:
            return True

    return False


# =========================================================
# 개인 정보에서 답변 찾기
# =========================================================

def get_profile_answer(question):

    # 이름
    if "이름" in question or "성함" in question:

        return f'제 이름은 **{PROFILE["이름"]}**입니다.'


    # 전화번호
    if (
        "전화번호" in question
        or "핸드폰" in question
        or "휴대폰" in question
        or "연락처" in question
    ):

        return f'전화번호는 **{PROFILE["전화번호"]}**입니다.'


    # 주소
    if "주소" in question or "사는 곳" in question:

        return f'주소는 **{PROFILE["주소"]}**입니다.'


    # 아버지
    if "아버지" in question or "아빠" in question:

        return f'아버지는 **{PROFILE["가족관계"]["아버지"]}**입니다.'


    # 어머니
    if "어머니" in question or "엄마" in question:

        return f'어머니는 **{PROFILE["가족관계"]["어머니"]}**입니다.'


    # 동생
    if "동생" in question:

        return f'동생은 **{PROFILE["가족관계"]["동생"]}**입니다.'


    # 가족 전체
    if "가족" in question:

        family = PROFILE["가족관계"]

        return f"""
### 👨‍👩‍👧‍👦 가족관계

- 아버지: **{family["아버지"]}**
- 어머니: **{family["어머니"]}**
- 동생: **{family["동생"]}**
"""


    return "등록된 개인 정보에서 해당 내용을 찾을 수 없습니다."


# =========================================================
# 대화 기록
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =========================================================
# 사용자 질문 입력
# =========================================================

question = st.chat_input(
    "질문을 입력하세요."
)


# =========================================================
# 질문 처리
# =========================================================

if question:

    # -------------------------
    # 사용자 질문 화면 표시
    # -------------------------

    with st.chat_message("user"):

        st.markdown(question)


    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # =====================================================
    # 개인 정보 질문
    # =====================================================

    if is_profile_question(question):

        with st.chat_message("assistant"):

            answer = get_profile_answer(question)

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })


    # =====================================================
    # 문서 질문
    # =====================================================

    else:

        with st.chat_message("assistant"):

            try:

                # -------------------------
                # 문서가 있는 경우
                # -------------------------

                if document_text.strip():

                    prompt = f"""
당신은 문서 기반 AI RAG 챗봇입니다.

아래 업로드된 문서의 내용을 참고해서 사용자의 질문에 답변하세요.

====================
[업로드된 문서]
====================

{document_text}

====================
[사용자 질문]
====================

{question}

====================
[답변 규칙]
====================

1. 반드시 업로드된 문서의 내용을 우선적으로 참고하세요.
2. 문서에 없는 내용은 추측하지 마세요.
3. 문서에서 답을 찾을 수 없다면
   "업로드된 문서에서 해당 내용을 찾을 수 없습니다."
   라고 답변하세요.
4. 답변은 이해하기 쉬운 한국어로 작성하세요.
5. 질문과 관련된 내용만 간결하게 답변하세요.
"""

                # -------------------------
                # 문서가 없는 경우
                # -------------------------

                else:

                    prompt = f"""
사용자의 질문에 친절하게 답변하세요.

사용자 질문:
{question}

답변 규칙:
1. 한국어로 답변하세요.
2. 모르는 내용은 추측하지 마세요.
3. 이해하기 쉽게 설명하세요.
"""


                # -------------------------
                # Gemini 호출
                # -------------------------

                response = client.models.generate_content(
                    model="gemini-3.8-flash",
                    contents=prompt
                )


                answer = response.text


                # -------------------------
                # 답변 표시
                # -------------------------

                st.markdown(answer)


                # -------------------------
                # 대화 기록 저장
                # -------------------------

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })


            except Exception as e:

                error_message = f"AI 응답 중 오류가 발생했습니다: {e}"

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
