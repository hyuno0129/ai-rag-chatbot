import streamlit as st
from google import genai
from google.genai import types

import PyPDF2
from docx import Document

import numpy as np
import faiss

from profile_data import PROFILE


# =========================================================
# 페이지 설정
# =========================================================

st.set_page_config(
    page_title="나의 AI 챗봇",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 나의 AI 챗봇")

st.write(
    "개인 정보와 업로드한 문서를 설명하고, 일반적인 질문에도 답변하는 AI 챗봇입니다."
)


# =========================================================
# Gemini 연결
# =========================================================

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"]
)


# =========================================================
# 모델 설정
# =========================================================

CHAT_MODEL = "gemini-3.8-flash"

EMBEDDING_MODEL = "gemini-embedding-001"

EMBEDDING_DIMENSION = 768


# =========================================================
# 문서 Chunk 설정
# =========================================================

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 5

# 문서 질문으로 판단할 때 사용하는 키워드
DOCUMENT_KEYWORDS = [
    "문서",
    "파일",
    "pdf",
    "word",
    "docx",
    "업로드",
    "자료",
    "보고서",
    "내용",
    "페이지",
    "쪽",
    "이 자료",
    "이 파일",
    "이 문서",
]


# =========================================================
# 문서 텍스트 추출
# =========================================================

def extract_text_from_file(uploaded_file):

    document_text = ""

    # =====================================================
    # PDF
    # =====================================================

    if uploaded_file.name.lower().endswith(".pdf"):

        pdf_reader = PyPDF2.PdfReader(uploaded_file)

        for page_number, page in enumerate(pdf_reader.pages):

            text = page.extract_text()

            if text:

                document_text += (
                    f"\n[페이지 {page_number + 1}]\n"
                )

                document_text += text
                document_text += "\n"

    # =====================================================
    # DOCX
    # =====================================================

    elif uploaded_file.name.lower().endswith(".docx"):

        doc = Document(uploaded_file)

        for paragraph in doc.paragraphs:

            text = paragraph.text.strip()

            if text:
                document_text += text + "\n"

    return document_text


# =========================================================
# 문서 Chunk 분할
# =========================================================

def split_text_into_chunks(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk.strip())

        start += chunk_size - overlap

    return chunks


# =========================================================
# Gemini Embedding 생성
# =========================================================

def create_embeddings(texts, task_type):

    result = client.models.embed_content(

        model=EMBEDDING_MODEL,

        contents=texts,

        config=types.EmbedContentConfig(

            task_type=task_type,

            output_dimensionality=EMBEDDING_DIMENSION

        )
    )

    embeddings = []

    for embedding in result.embeddings:
        embeddings.append(embedding.values)

    return np.array(
        embeddings,
        dtype="float32"
    )


# =========================================================
# FAISS 벡터 DB 생성
# =========================================================

def create_faiss_index(chunks):

    document_embeddings = create_embeddings(
        chunks,
        "RETRIEVAL_DOCUMENT"
    )

    # 벡터 정규화
    faiss.normalize_L2(document_embeddings)

    # Cosine Similarity
    index = faiss.IndexFlatIP(
        EMBEDDING_DIMENSION
    )

    index.add(
        document_embeddings
    )

    return index


# =========================================================
# 질문과 관련된 문서 검색
# =========================================================

def search_relevant_chunks(
    question,
    index,
    chunks,
    top_k=TOP_K
):

    query_embedding = create_embeddings(
        [question],
        "RETRIEVAL_QUERY"
    )

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(
        query_embedding,
        min(top_k, len(chunks))
    )

    results = []

    for score, index_number in zip(
        scores[0],
        indices[0]
    ):

        if index_number == -1:
            continue

        results.append({
            "text": chunks[index_number],
            "score": float(score),
            "index": int(index_number)
        })

    return results


# =========================================================
# 개인정보 질문인지 확인
# =========================================================

def is_profile_question(question):

    profile_keywords = [

        # 본인 기본 정보
        "이름",
        "성함",
        "학번",
        "학생번호",
        "전화번호",
        "핸드폰",
        "휴대폰",
        "연락처",
        "이메일",
        "메일",
        "주소",
        "사는 곳",

        # 학교 정보
        "학교",
        "대학교",
        "학과",
        "전공",
        "학년",

        # 기타
        "생일",
        "생년월일",
        "MBTI",
        "취미",
        "좋아하는 음식",
        "좋아하는 음악",

        # 가족
        "가족",
        "아버지",
        "아빠",
        "어머니",
        "엄마",
        "동생",
        "누나",
        "형",
        "언니"
    ]

    question_lower = question.lower()

    for keyword in profile_keywords:

        if keyword.lower() in question_lower:
            return True

    return False


# =========================================================
# 개인정보 답변
# =========================================================

def get_profile_answer(question):

    question_lower = question.lower()

    # =====================================================
    # 가족관계
    # 중요:
    # "아버지 이름"처럼 "이름"이 같이 들어오는 경우가
    # 있기 때문에 가족을 먼저 검사한다.
    # =====================================================

    # 아버지
    if (
        "아버지" in question
        or "아빠" in question
    ):
        return (
            f'아버지는 '
            f'**{PROFILE["가족관계"]["아버지"]}**입니다.'
        )

    # 어머니
    if (
        "어머니" in question
        or "엄마" in question
    ):
        return (
            f'어머니는 '
            f'**{PROFILE["가족관계"]["어머니"]}**입니다.'
        )

    # 동생
    if "동생" in question:
        return (
            f'동생은 '
            f'**{PROFILE["가족관계"]["동생"]}**입니다.'
        )

    # 누나
    if "누나" in question:
        return (
            f'누나는 '
            f'**{PROFILE["가족관계"]["누나"]}**입니다.'
        )

    # 형
    if "형" in question:
        return (
            f'형은 '
            f'**{PROFILE["가족관계"]["형"]}**입니다.'
        )

    # 언니
    if "언니" in question:
        return (
            f'언니는 '
            f'**{PROFILE["가족관계"]["언니"]}**입니다.'
        )

    # =====================================================
    # 가족 전체
    # =====================================================

    if "가족" in question:

        family = PROFILE["가족관계"]

        family_text = "### 👨‍👩‍👧‍👦 가족관계\n\n"

        for relation, name in family.items():

            family_text += (
                f"- {relation}: **{name}**\n"
            )

        return family_text

    # =====================================================
    # 본인 이름
    # =====================================================

    if (
        "이름" in question
        or "성함" in question
    ):
        return (
            f'제 이름은 '
            f'**{PROFILE["이름"]}**입니다.'
        )

    # =====================================================
    # 학번
    # =====================================================

    if (
        "학번" in question
        or "학생번호" in question
    ):
        return (
            f'학번은 '
            f'**{PROFILE["학번"]}**입니다.'
        )

    # =====================================================
    # 전화번호
    # =====================================================

    if (
        "전화번호" in question
        or "핸드폰" in question
        or "휴대폰" in question
        or "연락처" in question
    ):
        return (
            f'전화번호는 '
            f'**{PROFILE["전화번호"]}**입니다.'
        )

    # =====================================================
    # 이메일
    # =====================================================

    if (
        "이메일" in question
        or "메일" in question
    ):
        return (
            f'이메일은 '
            f'**{PROFILE["이메일"]}**입니다.'
        )

    # =====================================================
    # 주소
    # =====================================================

    if (
        "주소" in question
        or "사는 곳" in question
    ):
        return (
            f'주소는 '
            f'**{PROFILE["주소"]}**입니다.'
        )

    # =====================================================
    # 학교
    # =====================================================

    if (
        "학교" in question
        or "대학교" in question
    ):
        return (
            f'학교는 '
            f'**{PROFILE["학교"]}**입니다.'
        )

    # =====================================================
    # 학과 / 전공
    # =====================================================

    if (
        "학과" in question
        or "전공" in question
    ):
        return (
            f'전공은 '
            f'**{PROFILE["학과"]}**입니다.'
        )

    # =====================================================
    # 학년
    # =====================================================

    if "학년" in question:
        return (
            f'현재 '
            f'**{PROFILE["학년"]}**입니다.'
        )

    # =====================================================
    # 생년월일
    # =====================================================

    if (
        "생일" in question
        or "생년월일" in question
    ):
        return (
            f'생년월일은 '
            f'**{PROFILE["생년월일"]}**입니다.'
        )

    # =====================================================
    # MBTI
    # =====================================================

    if "mbti" in question_lower:
        return (
            f'MBTI는 '
            f'**{PROFILE["MBTI"]}**입니다.'
        )

    # =====================================================
    # 취미
    # =====================================================

    if "취미" in question:

        hobbies = ", ".join(
            PROFILE["취미"]
        )

        return (
            f'취미는 **{hobbies}**입니다.'
        )

    # =====================================================
    # 좋아하는 음식
    # =====================================================

    if "좋아하는 음식" in question:

        foods = ", ".join(
            PROFILE["좋아하는 음식"]
        )

        return (
            f'좋아하는 음식은 '
            f'**{foods}**입니다.'
        )

    # =====================================================
    # 좋아하는 음악
    # =====================================================

    if "좋아하는 음악" in question:

        music = ", ".join(
            PROFILE["좋아하는 음악"]
        )

        return (
            f'좋아하는 음악은 '
            f'**{music}**입니다.'
        )

    # =====================================================
    # 자기소개 / 전체 정보
    # =====================================================

    if (
        "자기소개" in question
        or "나에 대해" in question
        or "내 정보" in question
        or "내 정보 알려줘" in question
    ):

        return f"""
### 👤 나의 정보

- 이름: **{PROFILE["이름"]}**
- 학번: **{PROFILE["학번"]}**
- 학교: **{PROFILE["학교"]}**
- 학과: **{PROFILE["학과"]}**
- 학년: **{PROFILE["학년"]}**
- 이메일: **{PROFILE["이메일"]}**
- 전화번호: **{PROFILE["전화번호"]}**
- 주소: **{PROFILE["주소"]}**
- 생년월일: **{PROFILE["생년월일"]}**
- MBTI: **{PROFILE["MBTI"]}**
- 취미: **{", ".join(PROFILE["취미"])}**
- 좋아하는 음식: **{", ".join(PROFILE["좋아하는 음식"])}**
- 좋아하는 음악: **{", ".join(PROFILE["좋아하는 음악"])}**
"""

    return (
        "등록된 개인 정보에서 "
        "해당 내용을 찾을 수 없습니다."
    )


# =========================================================
# 문서 질문인지 확인
# =========================================================

def is_document_question(question):

    question_lower = question.lower()

    for keyword in DOCUMENT_KEYWORDS:

        if keyword.lower() in question_lower:
            return True

    return False


# =========================================================
# 일반 Gemini 답변
# =========================================================

def generate_general_answer(question):

    prompt = f"""
당신은 친절한 한국어 AI 챗봇입니다.

사용자의 질문에 자연스럽고 정확하게 답변하세요.

질문:
{question}

답변 규칙:
1. 한국어로 답변하세요.
2. 모르는 내용은 아는 척하지 마세요.
3. 질문과 관련된 내용만 이해하기 쉽게 설명하세요.
4. 일반적인 대화, 인사, 상식, 공부 질문 등에 자유롭게 답변하세요.
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    return response.text


# =========================================================
# RAG 답변 생성
# =========================================================

def generate_rag_answer(
    question,
    search_results
):

    context = ""

    for i, result in enumerate(
        search_results,
        start=1
    ):

        context += f"""

[검색 결과 {i}]

유사도 점수:
{result["score"]:.4f}

{result["text"]}

--------------------
"""

    prompt = f"""
당신은 문서 기반 RAG AI 챗봇입니다.

사용자의 질문에 답변하기 위해
검색된 문서 내용을 참고하세요.

========================
[검색된 문서 내용]
========================

{context}

========================
[사용자 질문]
========================

{question}

========================
[답변 규칙]
========================

1. 검색된 문서 내용을 우선적으로 사용하세요.
2. 문서에 없는 내용을 추측하지 마세요.
3. 검색된 문서에서 질문에 대한 답을
   찾을 수 없다면 다음과 같이 답하세요.

"업로드된 문서에서 해당 내용을 찾을 수 없습니다."

4. 한국어로 답변하세요.
5. 질문과 관련된 내용만 간결하고
   이해하기 쉽게 설명하세요.
"""

    response = client.models.generate_content(
        model=CHAT_MODEL,
        contents=prompt
    )

    return response.text


# =========================================================
# 사이드바
# =========================================================

st.sidebar.header("📄 문서 업로드")

uploaded_file = st.sidebar.file_uploader(
    "PDF 또는 Word 파일을 선택하세요.",
    type=["pdf", "docx"]
)


# =========================================================
# 문서 처리
# =========================================================

if uploaded_file is not None:

    # 파일 이름 + 크기
    file_identifier = (
        uploaded_file.name,
        uploaded_file.size
    )

    if (
        "file_identifier" not in st.session_state
        or
        st.session_state.file_identifier
        != file_identifier
    ):

        try:

            # =================================================
            # 텍스트 추출
            # =================================================

            document_text = extract_text_from_file(
                uploaded_file
            )

            if not document_text.strip():

                st.sidebar.warning(
                    "⚠️ 문서에서 텍스트를 찾을 수 없습니다."
                )

            else:

                # =================================================
                # Chunk 생성
                # =================================================

                chunks = split_text_into_chunks(
                    document_text
                )

                # =================================================
                # FAISS 생성
                # =================================================

                with st.spinner(
                    "📚 문서를 분석하고 벡터 DB를 만드는 중..."
                ):

                    index = create_faiss_index(
                        chunks
                    )

                # =================================================
                # Session State 저장
                # =================================================

                st.session_state.file_identifier = (
                    file_identifier
                )

                st.session_state.document_text = (
                    document_text
                )

                st.session_state.chunks = (
                    chunks
                )

                st.session_state.faiss_index = (
                    index
                )

                # =================================================
                # 상태 표시
                # =================================================

                st.sidebar.success(
                    f"✅ {uploaded_file.name} 업로드 완료"
                )

                st.sidebar.write(
                    f"문서 글자 수: "
                    f"{len(document_text):,}"
                )

                st.sidebar.write(
                    f"문서 Chunk 수: "
                    f"{len(chunks):,}"
                )

                st.sidebar.success(
                    "🧠 RAG 벡터 DB 생성 완료!"
                )

        except Exception as e:

            st.sidebar.error(
                f"문서 처리 중 오류가 발생했습니다: {e}"
            )


# =========================================================
# 현재 RAG 상태
# =========================================================

if "faiss_index" in st.session_state:

    st.sidebar.divider()

    st.sidebar.write(
        "🟢 **RAG 상태: 준비 완료**"
    )

    st.sidebar.write(
        f"검색 대상 Chunk: "
        f"{len(st.session_state.chunks)}개"
    )


# =========================================================
# 대화 기록
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# 질문 입력
# =========================================================

question = st.chat_input(
    "질문을 입력하세요."
)


# =========================================================
# 질문 처리
# =========================================================

if question:

    # =====================================================
    # 사용자 질문
    # =====================================================

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    # =====================================================
    # 개인정보 질문
    # =====================================================

    if is_profile_question(question):

        with st.chat_message("assistant"):

            answer = get_profile_answer(
                question
            )

            st.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

    # =====================================================
    # 문서 질문
    # =====================================================

    elif is_document_question(question):

        # -------------------------------------------------
        # 문서가 없는 경우
        # -------------------------------------------------

        if "faiss_index" not in st.session_state:

            with st.chat_message("assistant"):

                answer = """
📄 현재 업로드된 문서가 없습니다.

문서에 대한 질문을 하려면 왼쪽에서 PDF 또는 Word 문서를 먼저 업로드해주세요.
"""

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

        # -------------------------------------------------
        # RAG 검색
        # -------------------------------------------------

        else:

            with st.chat_message("assistant"):

                try:

                    # =========================================
                    # 관련 문서 검색
                    # =========================================

                    with st.spinner(
                        "🔎 관련 문서를 검색하는 중..."
                    ):

                        search_results = (
                            search_relevant_chunks(
                                question,
                                st.session_state.faiss_index,
                                st.session_state.chunks
                            )
                        )

                    # =========================================
                    # Gemini 답변
                    # =========================================

                    with st.spinner(
                        "🤖 답변을 생성하는 중..."
                    ):

                        answer = generate_rag_answer(
                            question,
                            search_results
                        )

                    # =========================================
                    # 답변
                    # =========================================

                    st.markdown(answer)

                    # =========================================
                    # 참고 문서
                    # =========================================

                    st.divider()

                    st.caption(
                        "📚 참고한 문서 내용"
                    )

                    for i, result in enumerate(
                        search_results,
                        start=1
                    ):

                        with st.expander(
                            f"검색 결과 {i} "
                            f"(유사도 {result['score']:.3f})"
                        ):

                            st.write(
                                result["text"]
                            )

                    # =========================================
                    # 대화 기록
                    # =========================================

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    error_message = (
                        "RAG 처리 중 오류가 발생했습니다:\n\n"
                        f"{e}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_message
                    })

    # =====================================================
    # 일반 질문
    # =====================================================

    else:

        with st.chat_message("assistant"):

            try:

                with st.spinner(
                    "🤖 답변을 생성하는 중..."
                ):

                    answer = generate_general_answer(
                        question
                    )

                st.markdown(answer)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except Exception as e:

                error_message = (
                    "답변 생성 중 오류가 발생했습니다:\n\n"
                    f"{e}"
                )

                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })
