import base64
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from utils.pdf_loader import extract_pages_from_pdf
from utils.chunker import split_pages_into_chunks
from utils.embeddings import create_embeddings, create_query_embedding
from utils.vector_store import create_vector_store, search_vector_store
from utils.rag_pipeline import generate_answer
from utils.summarizer import summarize_pdf

from utils.question_generator import generate_questions
from utils.quiz_manager import calculate_score
from utils.downloader import (
    create_questions_text,
    create_answer_key,
    create_questions_and_answers,
    text_to_bytes,
)


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
)

PDF_DIR = Path("uploaded_pdfs")
PDF_DIR.mkdir(exist_ok=True)


# ============================================================
# SESSION STATE
# ============================================================

defaults = {
    "chunks": [],
    "index": None,
    "messages": [],
    "files": [],
    "summary": None,
    "summary_file": None,
    "viewer_file": None,
    "viewer_page": 1,
    "questions": [],
    "question_file": None,
    "quiz_result": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# FUNCTIONS
# ============================================================

def save_pdf(file):
    path = PDF_DIR / file.name
    path.write_bytes(file.getbuffer())
    return path


def process_pdf(file):
    pages = extract_pages_from_pdf(file)

    if not pages:
        raise ValueError("Could not extract text from PDF.")

    chunks = split_pages_into_chunks(pages)

    if not chunks:
        raise ValueError("Could not create PDF chunks.")

    return [
        {
            "source": file.name,
            "page": chunk.get("page", 1),
            "text": chunk.get("text", ""),
        }
        for chunk in chunks
    ]


def rebuild_index(documents):
    texts = [x["text"] for x in documents if x.get("text")]

    if not texts:
        raise ValueError("No text available.")

    embeddings = create_embeddings(texts)
    return create_vector_store(embeddings)


def open_source(source, page):
    st.session_state.viewer_file = source
    st.session_state.viewer_page = int(page)
    st.rerun()


def show_pdf(source, page):
    path = PDF_DIR / source

    if not path.exists():
        st.error("PDF not found.")
        return

    data = base64.b64encode(path.read_bytes()).decode()

    url = (
        "data:application/pdf;base64,"
        + data
        + f"#page={page}"
    )

    components.html(
        f"""
        <iframe
            src="{url}"
            width="100%"
            height="700"
            style="border:1px solid #ddd;border-radius:10px;">
        </iframe>
        """,
        height=720,
    )


def build_context(documents):
    return "\n\n".join(
        f"SOURCE: {doc['source']}\\n"
        f"PAGE: {doc['page']}\\n\\n"
        f"{doc['text']}"
        for doc in documents
    )



# ============================================================
# WHOLE PDF SUMMARY
# ============================================================

def generate_full_pdf_summary(documents, max_chars_per_part=50000):
    """Fast whole-document summary.

    Uses a small number of LLM calls by batching many PDF chunks together.
    This is designed to be much faster for large PDFs such as 90+ pages.
    """
    if not documents:
        raise ValueError("No document content available.")

    parts = []
    current = []
    current_len = 0

    for doc in documents:
        text_part = str(doc.get("text", "")).strip()
        if not text_part:
            continue

        item = (
            f"SOURCE: {doc.get('source', '')}\n"
            f"PAGE: {doc.get('page', 1)}\n"
            f"{text_part}"
        )

        if current and current_len + len(item) > max_chars_per_part:
            parts.append("\n\n".join(current))
            current = []
            current_len = 0

        current.append(item)
        current_len += len(item)

    if current:
        parts.append("\n\n".join(current))

    if not parts:
        raise ValueError("The PDF contains no extractable text.")

    # Keep the number of model calls low. A large PDF is summarized in
    # batches, followed by one final consolidation call.
    partial_summaries = []
    for i, part in enumerate(parts, 1):
        prompt = f"""
Summarize this section of a college study PDF accurately and concisely.

Cover the important definitions, concepts, formulas, examples, steps and
conclusions. Use only the supplied text. Do not invent information.

SECTION {i} OF {len(parts)}:
{part}
"""
        partial_summaries.append(generate_answer(prompt, part))

    if len(partial_summaries) == 1:
        return partial_summaries[0]

    combined = "\n\n".join(
        f"SECTION {i} SUMMARY:\n{s}"
        for i, s in enumerate(partial_summaries, 1)
    )

    final_prompt = """
Create one concise but complete study summary from the section summaries below.
Preserve important information from every section, remove repetition, and use
clear headings and bullet points. Do not invent information.

SECTION SUMMARIES:
""" + combined

    return generate_answer(final_prompt, combined)


# ============================================================
# HEADER
# ============================================================

st.title("📚 AI Study Assistant")
st.caption(
    "Upload PDFs, ask questions, generate exams, "
    "take quizzes, and download questions."
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Study Material")

    uploaded = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
    )

    if uploaded and st.button(
        "➕ Add PDF",
        type="primary",
        use_container_width=True,
    ):

        try:
            with st.spinner("Processing PDF..."):

                save_pdf(uploaded)

                new_docs = process_pdf(uploaded)

                old_docs = [
                    x
                    for x in st.session_state.chunks
                    if x["source"] != uploaded.name
                ]

                documents = old_docs + new_docs

                index = rebuild_index(documents)

                st.session_state.chunks = documents
                st.session_state.index = index

                if uploaded.name not in st.session_state.files:
                    st.session_state.files.append(uploaded.name)

            st.success("✅ PDF added.")

        except Exception as e:
            st.error("PDF processing failed.")
            st.exception(e)

    # ========================================================
    # STUDY STATISTICS
    # ========================================================
    st.divider()
    st.header("📊 Study Statistics")

    st.metric(
        "Knowledge Chunks",
        len(st.session_state.chunks)
    )

    if st.session_state.files:
        st.caption(
            f"📚 {len(st.session_state.files)} PDF(s) currently loaded"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    st.divider()
    st.header("📝 Summary")

    if st.session_state.files:

        summary_file = st.selectbox(
            "PDF",
            st.session_state.files,
            key="summary_select",
        )

        selected_docs = [
            x for x in st.session_state.chunks
            if x["source"] == summary_file
        ]
        page_numbers = sorted({
            int(x.get("page", 1))
            for x in selected_docs
            if str(x.get("page", "")).isdigit()
        })

        st.caption(
            f"📄 {len(page_numbers)} pages detected • "
            f"{len(selected_docs)} text chunks will be processed"
        )

        if st.button(
            "⚡ Generate Fast Full Summary",
            use_container_width=True,
        ):

            docs = [
                x
                for x in st.session_state.chunks
                if x["source"] == summary_file
            ]

            try:
                with st.spinner("Generating summary..."):
                    st.session_state.summary = generate_full_pdf_summary(docs)
                    st.session_state.summary_file = summary_file

                st.success("Summary generated.")

            except Exception as e:
                st.error("Summary failed.")
                st.exception(e)

    # ========================================================
    # EXAM GENERATOR
    # ========================================================

    st.divider()
    st.header("📝 Exam Generator")

    if st.session_state.files:

        exam_file = st.selectbox(
            "Study Material",
            st.session_state.files,
            key="exam_select",
        )

        count = st.slider(
            "Questions",
            1,
            20,
            5,
        )

        difficulty = st.selectbox(
            "Difficulty",
            ["Easy", "Medium", "Hard"],
        )

        qtype = st.selectbox(
            "Type",
            ["MCQ", "Short Answer"],
        )

        if st.button(
            "🚀 Generate Exam",
            type="primary",
            use_container_width=True,
        ):

            docs = [
                x
                for x in st.session_state.chunks
                if x["source"] == exam_file
            ]

            context = build_context(docs)

            try:
                with st.spinner("Generating questions..."):

                    questions = generate_questions(
                        context,
                        num_questions=count,
                        difficulty=difficulty,
                        question_type=qtype,
                    )

                st.session_state.questions = questions
                st.session_state.question_file = exam_file
                st.session_state.quiz_result = None

                st.success(
                    f"Generated {len(questions)} questions."
                )

            except Exception as e:
                st.error("Question generation failed.")
                st.exception(e)

    # ========================================================
    # STATUS
    # ========================================================

    st.divider()

    st.write(
        f"📄 PDFs: {len(st.session_state.files)}"
    )

    st.write(
        f"🧩 Chunks: {len(st.session_state.chunks)}"
    )

    if st.button(
        "🗑️ Clear Knowledge Base",
        use_container_width=True,
    ):

        for key in [
            "chunks",
            "files",
            "questions",
            "summary",
        ]:
            st.session_state[key] = []

        st.session_state.index = None
        st.session_state.question_file = None
        st.session_state.summary_file = None

        st.rerun()

    if st.button(
        "💬 Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []
        st.rerun()


# ============================================================
# SUMMARY DISPLAY
# ============================================================

if st.session_state.summary:

    st.divider()

    st.subheader(
        f"📝 Summary — {st.session_state.summary_file}"
    )

    st.markdown(st.session_state.summary)

    if st.button("✕ Close Summary"):
        st.session_state.summary = None
        st.session_state.summary_file = None
        st.rerun()


# ============================================================
# EXAM + DOWNLOAD + QUIZ
# ============================================================

if st.session_state.questions:

    questions = st.session_state.questions

    st.divider()

    st.header(
        f"📝 Exam — {st.session_state.question_file}"
    )

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    st.subheader("📥 Download Questions / Answers")

    questions_text = create_questions_text(questions)
    answers_text = create_answer_key(questions)
    combined_text = create_questions_and_answers(questions)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button(
            "📄 Questions",
            text_to_bytes(questions_text),
            "questions.txt",
            "text/plain",
            use_container_width=True,
        )

    with c2:
        st.download_button(
            "✅ Answers",
            text_to_bytes(answers_text),
            "answers.txt",
            "text/plain",
            use_container_width=True,
        )

    with c3:
        st.download_button(
            "📚 Both",
            text_to_bytes(combined_text),
            "questions_answers.txt",
            "text/plain",
            use_container_width=True,
        )

    # --------------------------------------------------------
    # QUESTIONS
    # --------------------------------------------------------

    st.divider()
    st.subheader("📋 Generated Questions")

    for i, q in enumerate(questions, 1):

        st.markdown(
            f"### {i}. {q.get('question', '')}"
        )

        options = q.get("options", [])

        for n, option in enumerate(options):
            st.write(
                f"**{chr(65 + n)}.** {option}"
            )

        with st.expander("💡 Show Answer"):
            st.write(
                f"**Answer:** {q.get('answer', '')}"
            )

            if q.get("explanation"):
                st.write(
                    f"**Explanation:** "
                    f"{q['explanation']}"
                )

        if q.get("source"):
            st.caption(
                f"📄 {q['source']} "
                f"• Page {q.get('page', '')}"
            )

    # --------------------------------------------------------
    # QUIZ MODE
    # --------------------------------------------------------

    st.divider()
    st.header("🎯 Quiz Mode")

    if st.session_state.quiz_result is None:

        with st.form("quiz_form"):

            user_answers = {}

            for i, q in enumerate(questions):

                st.markdown(
                    f"### Question {i + 1}"
                )

                st.write(
                    q.get("question", "")
                )

                options = q.get("options", [])

                if options:

                    user_answers[i] = st.radio(
                        "Select answer",
                        options,
                        key=f"quiz_{i}",
                        index=None,
                    )

                else:

                    user_answers[i] = st.text_input(
                        "Your answer",
                        key=f"quiz_{i}",
                    )

            submitted = st.form_submit_button(
                "✅ Submit Quiz",
                use_container_width=True,
            )

        if submitted:

            result = calculate_score(
                questions,
                user_answers,
            )

            st.session_state.quiz_result = result
            st.rerun()

    else:

        result = st.session_state.quiz_result

        st.success(
            f"🎉 Score: {result['percentage']}%"
        )

        c1, c2, c3 = st.columns(3)

        c1.metric("Correct", result["correct"])
        c2.metric("Wrong", result["wrong"])
        c3.metric("Total", result["total"])

        st.subheader("📋 Quiz Review")

        for i, item in enumerate(
            result["results"],
            1,
        ):

            if item["is_correct"]:
                st.success(
                    f"Question {i} — Correct ✅"
                )
            else:
                st.error(
                    f"Question {i} — Wrong ❌"
                )

            st.write(item["question"])

            st.write(
                f"Your answer: "
                f"{item['user_answer']}"
            )

            st.write(
                f"Correct answer: "
                f"{item['correct_answer']}"
            )

        if st.button(
            "🔄 Retake Quiz",
            use_container_width=True,
        ):

            st.session_state.quiz_result = None
            st.rerun()


# ============================================================
# PDF VIEWER
# ============================================================

if st.session_state.viewer_file:

    st.divider()

    st.subheader("📖 Source Document")

    st.caption(
        f"{st.session_state.viewer_file} "
        f"• Page {st.session_state.viewer_page}"
    )

    show_pdf(
        st.session_state.viewer_file,
        st.session_state.viewer_page,
    )

    if st.button("✕ Close Document"):
        st.session_state.viewer_file = None
        st.rerun()


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "💬 Ask a question about your PDFs..."
)

if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        if st.session_state.index is None:

            answer = "📄 Please upload a PDF first."
            st.warning(answer)

        else:

            try:

                with st.spinner("🔎 Searching documents..."):

                    query_embedding = (
                        create_query_embedding(question)
                    )

                    results = search_vector_store(
                        st.session_state.index,
                        query_embedding,
                        st.session_state.chunks,
                        top_k=3,
                    )

                if not results:

                    answer = (
                        "I couldn't find relevant "
                        "information in your PDFs."
                    )

                    st.warning(answer)

                else:

                    context = build_context(results)

                    with st.spinner(
                        "🤖 Generating answer..."
                    ):

                        answer = generate_answer(
                            question,
                            context,
                        )

                    st.markdown(answer)

                    st.divider()
                    st.subheader("📚 Sources")

                    seen = set()

                    for i, result in enumerate(results):

                        source = result["source"]
                        page = result["page"]

                        if (source, page) in seen:
                            continue

                        seen.add((source, page))

                        c1, c2 = st.columns([5, 1])

                        with c1:
                            st.write(
                                f"📄 **{source}** "
                                f"— Page **{page}**"
                            )

                        with c2:

                            if st.button(
                                "Open",
                                key=f"source_{i}_{page}",
                            ):
                                open_source(
                                    source,
                                    page,
                                )

            except Exception as e:

                answer = (
                    "❌ Something went wrong "
                    "while answering."
                )

                st.error(answer)
                st.exception(e)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
