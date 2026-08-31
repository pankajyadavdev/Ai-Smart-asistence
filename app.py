import streamlit as st

from utils.pdf_loader import extract_pages_from_pdf
from utils.chunker import split_pages_into_chunks
from utils.embeddings import create_embeddings, create_query_embedding
from utils.vector_store import create_vector_store, search_vector_store
from utils.rag_pipeline import generate_answer
from utils.college_data_loader import load_college_data
from utils.college_query import answer_college_question


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI College Assistant",
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🎓 AI College Assistant")

st.caption(
    "IIT(BHU) Knowledge Base + PDF RAG + Granite 4.1:3B"
)

st.divider()


# ============================================================
# SESSION STATE
# ============================================================

if "college_documents" not in st.session_state:
    st.session_state.college_documents = load_college_data()

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "file_name" not in st.session_state:
    st.session_state.file_name = ""

if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


# ============================================================
# BUILD INITIAL IIT(BHU) KNOWLEDGE BASE
# ============================================================

if st.session_state.index is None:

    college_documents = st.session_state.college_documents

    if college_documents:

        with st.spinner("🏫 Loading IIT(BHU) knowledge base..."):

            college_texts = [
                document["text"]
                for document in college_documents
            ]

            college_embeddings = create_embeddings(
                college_texts
            )

            st.session_state.index = create_vector_store(
                college_embeddings
            )

            st.session_state.chunks = college_documents.copy()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.success("🟢 IIT(BHU) data loaded")

    st.write(
        f"College sections: "
        f"{len(st.session_state.college_documents)}"
    )

    st.divider()

    st.subheader("📄 Add Study Material")

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        if st.button(
            "📖 Add PDF to Knowledge Base",
            use_container_width=True
        ):

            # ------------------------------------------------
            # EXTRACT PDF
            # ------------------------------------------------

            with st.spinner("Reading PDF..."):

                pages = extract_pages_from_pdf(
                    uploaded_file
                )

            if not pages:

                st.error(
                    "Could not extract text from PDF."
                )

                st.stop()

            # ------------------------------------------------
            # CHUNK PDF
            # ------------------------------------------------

            with st.spinner("Creating PDF chunks..."):

                pdf_chunks = split_pages_into_chunks(
                    pages
                )

            # ------------------------------------------------
            # FORMAT PDF DOCUMENTS
            # ------------------------------------------------

            pdf_documents = []

            for chunk in pdf_chunks:

                pdf_documents.append(
                    {
                        "source": uploaded_file.name,
                        "page": chunk["page"],
                        "text": chunk["text"]
                    }
                )

            # ------------------------------------------------
            # COMBINE IIT(BHU) + PDF
            # ------------------------------------------------

            all_documents = (
                st.session_state.college_documents
                + pdf_documents
            )

            # ------------------------------------------------
            # EMBEDDINGS
            # ------------------------------------------------

            with st.spinner(
                "Creating embeddings..."
            ):

                all_texts = [
                    document["text"]
                    for document in all_documents
                ]

                embeddings = create_embeddings(
                    all_texts
                )

            # ------------------------------------------------
            # VECTOR INDEX
            # ------------------------------------------------

            with st.spinner(
                "Updating knowledge base..."
            ):

                index = create_vector_store(
                    embeddings
                )

            st.session_state.chunks = all_documents
            st.session_state.index = index
            st.session_state.file_name = uploaded_file.name
            st.session_state.pdf_processed = True

            st.success(
                "✅ PDF added to knowledge base!"
            )

            st.rerun()

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Status")

    st.write(
        f"🏫 IIT(BHU) sections: "
        f"{len(st.session_state.college_documents)}"
    )

    st.write(
        f"📚 Total chunks: "
        f"{len(st.session_state.chunks)}"
    )

    if st.session_state.pdf_processed:

        st.write(
            f"📄 PDF: "
            f"{st.session_state.file_name}"
        )

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.rerun()


# ============================================================
# WELCOME
# ============================================================

if not st.session_state.messages:

    st.info(
        "Ask a question about IIT(BHU), "
        "or add a PDF for study-material questions."
    )


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.write(message["content"])


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "💬 Ask about IIT(BHU) or your PDF..."
)


# ============================================================
# QUESTION PROCESSING
# ============================================================

if question:

    # --------------------------------------------------------
    # SHOW USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.write(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # --------------------------------------------------------
    # DIRECT IIT(BHU) LOOKUP
    #
    # Website, location, courses, departments, B.Tech
    # duration/admission, history, etc. are answered directly
    # from college_data.json.
    # --------------------------------------------------------

    direct_answer = answer_college_question(
        question
    )

    if direct_answer:

        with st.chat_message("assistant"):

            st.markdown(direct_answer)

            st.divider()

            st.caption(
                "🏫 Source: IIT(BHU) College Data"
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": direct_answer
            }
        )

    else:

        # ----------------------------------------------------
        # RAG SEARCH
        # ----------------------------------------------------

        if st.session_state.index is None:

            st.error(
                "Knowledge base is not ready."
            )

            st.stop()

        with st.chat_message("assistant"):

            with st.spinner(
                "🔎 Searching knowledge base..."
            ):

                query_embedding = (
                    create_query_embedding(
                        question
                    )
                )

                results = search_vector_store(
                    st.session_state.index,
                    query_embedding,
                    st.session_state.chunks,
                    top_k=5
                )

            if not results:

                answer = (
                    "I could not find relevant "
                    "information in the current "
                    "knowledge base."
                )

                st.warning(answer)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            else:

                # --------------------------------------------
                # BUILD CONTEXT
                # --------------------------------------------

                context_parts = []

                for result in results:

                    context_parts.append(
                        f"""
SOURCE:
{result["source"]}

SECTION / PAGE:
{result["page"]}

CONTENT:
{result["text"]}
"""
                    )

                context = "\n\n".join(
                    context_parts
                )

                # --------------------------------------------
                # GENERATE ANSWER
                # --------------------------------------------

                with st.spinner(
                    "🤖 Granite 4.1:3B is thinking..."
                ):

                    answer = generate_answer(
                        question,
                        context
                    )

                st.markdown(answer)

                # --------------------------------------------
                # SOURCES
                # --------------------------------------------

                st.divider()

                st.subheader("📚 Sources")

                displayed = set()

                for result in results:

                    key = (
                        result["source"],
                        result["page"]
                    )

                    if key in displayed:
                        continue

                    displayed.add(key)

                    if "IIT(BHU)" in result["source"]:

                        st.markdown(
                            f"🏫 **{result['source']} — "
                            f"{result['page']}**"
                        )

                    else:

                        st.markdown(
                            f"📄 **{result['source']} — "
                            f"Page {result['page']}**"
                        )

                # --------------------------------------------
                # RETRIEVED INFORMATION
                # --------------------------------------------

                with st.expander(
                    "🔎 View Retrieved Information"
                ):

                    for i, result in enumerate(results):

                        st.markdown(
                            f"### Source {i + 1}"
                        )

                        st.write(
                            f"**Source:** "
                            f"{result['source']}"
                        )

                        st.write(
                            f"**Section/Page:** "
                            f"{result['page']}"
                        )

                        st.write(
                            result["text"]
                        )

                        st.divider()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )
