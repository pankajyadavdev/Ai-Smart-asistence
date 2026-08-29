import streamlit as st
import hashlib
import hmac
import sqlite3
from pathlib import Path
from datetime import datetime


from utils.pdf_loader import extract_pages_from_pdf
from utils.chunker import split_pages_into_chunks
from utils.embeddings import create_embeddings, create_query_embedding
from utils.vector_store import create_vector_store, search_vector_store
from utils.rag_pipeline import generate_answer


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

st.caption("PDF RAG Study Assistant • Student Login & Dashboard")

st.divider()


# ============================================================
# LOGIN / SIGNUP DATABASE
# ============================================================

DB_PATH = Path(__file__).resolve().parent / "users.db"

def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL, email TEXT UNIQUE NOT NULL,
            student_id TEXT NOT NULL, college TEXT NOT NULL,
            course TEXT NOT NULL, semester TEXT NOT NULL,
            password_hash TEXT NOT NULL, created_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS activity (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
            activity_type TEXT NOT NULL, details TEXT, created_at TEXT NOT NULL
        )""")

def hash_password(password):
    salt = hashlib.sha256(password.encode()).hexdigest()[:32]
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
    return f"{salt}${digest}"

def verify_password(password, stored):
    try:
        salt, saved = stored.split("$", 1)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 120000).hex()
        return hmac.compare_digest(digest, saved)
    except Exception:
        return False

def create_user(name, email, student_id, college, course, semester, password):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("""INSERT INTO users
                (full_name,email,student_id,college,course,semester,password_hash,created_at)
                VALUES (?,?,?,?,?,?,?,?)""",
                (name.strip(), email.strip().lower(), student_id.strip(), college.strip(),
                 course.strip(), semester, hash_password(password),
                 datetime.now().isoformat(timespec="seconds")))
            conn.commit()
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except Exception as e:
        return False, str(e)

def authenticate(email, password):
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute("""SELECT id,full_name,email,student_id,college,course,semester,password_hash,created_at
                             FROM users WHERE email=?""", (email.strip().lower(),)).fetchone()
    if not row or not verify_password(password, row[7]):
        return None
    return {"id":row[0],"full_name":row[1],"email":row[2],"student_id":row[3],
            "college":row[4],"course":row[5],"semester":row[6],"created_at":row[8]}

def log_activity(user_id, kind, details=""):
    try:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("INSERT INTO activity(user_id,activity_type,details,created_at) VALUES(?,?,?,?)",
                         (user_id, kind, details[:250], datetime.now().isoformat(timespec="seconds")))
            conn.commit()
    except Exception:
        pass

def activity_count(user_id, kind):
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT COUNT(*) FROM activity WHERE user_id=? AND activity_type=?",
                            (user_id, kind)).fetchone()[0]

def recent_activity(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        return conn.execute("SELECT activity_type,details,created_at FROM activity WHERE user_id=? ORDER BY id DESC LIMIT 8",
                            (user_id,)).fetchall()

init_db()

# ============================================================
# LOGIN PAGE
# ============================================================

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "user" not in st.session_state:
    st.session_state.user = None

def show_auth_page():
    st.markdown("<h1 style='text-align:center'>🎓 AI College Study Assistant</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:gray'>Login or create a student account to continue</p>", unsafe_allow_html=True)
    left, center, right = st.columns([1, 2, 1])
    with center:
        login_tab, signup_tab = st.tabs(["🔐 Login", "📝 Sign Up"])
        with login_tab:
            email = st.text_input("Email", key="auth_login_email")
            password = st.text_input("Password", type="password", key="auth_login_password")
            if st.button("Login", type="primary", use_container_width=True):
                if not email or not password:
                    st.warning("Please enter email and password.")
                else:
                    user = authenticate(email, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        log_activity(user["id"], "login", "User logged in")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")
        with signup_tab:
            name = st.text_input("Full Name", key="auth_name")
            email = st.text_input("Email", key="auth_signup_email")
            student_id = st.text_input("Student ID / Roll Number", key="auth_student_id")
            college = st.text_input("College Name", key="auth_college")
            course = st.text_input("Course / Branch", key="auth_course")
            semester = st.selectbox("Semester", [f"{i}th" for i in range(1,9)], key="auth_semester")
            password = st.text_input("Password", type="password", key="auth_signup_password")
            confirm = st.text_input("Confirm Password", type="password", key="auth_confirm_password")
            if st.button("Create Account", type="primary", use_container_width=True):
                if not all([name,email,student_id,college,course,password,confirm]):
                    st.warning("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must contain at least 6 characters.")
                else:
                    ok,msg=create_user(name,email,student_id,college,course,semester,password)
                    if ok: st.success(msg)
                    else: st.error(msg)

if not st.session_state.authenticated:
    show_auth_page()
    st.stop()

current_user = st.session_state.user

# ============================================================
# SESSION STATE
# ============================================================

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

if "page" not in st.session_state:
    st.session_state.page = "Study Assistant"

# ============================================================
# STUDENT DASHBOARD
# ============================================================

if st.session_state.page == "Dashboard":
    st.header("📊 Student Dashboard")
    st.caption("Your saved student profile and study activity")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions", activity_count(current_user["id"], "question"))
    c2.metric("PDF Uploads", activity_count(current_user["id"], "upload"))
    c3.metric("Exam Questions", activity_count(current_user["id"], "exam_question"))
    c4.metric("Quizzes", activity_count(current_user["id"], "quiz"))

    st.divider()
    left, right = st.columns(2)

    with left:
        st.subheader("👤 Student Details")
        st.write(f"**Name:** {current_user['full_name']}")
        st.write(f"**Email:** {current_user['email']}")
        st.write(f"**Student ID:** {current_user['student_id']}")
        st.write(f"**College:** {current_user['college']}")
        st.write(f"**Course:** {current_user['course']}")
        st.write(f"**Semester:** {current_user['semester']}")
        st.write(f"**Account Created:** {current_user['created_at']}")

    with right:
        st.subheader("🕒 Recent Activity")
        rows = recent_activity(current_user["id"])
        if rows:
            for kind, details, created in rows:
                st.write(f"**{kind.replace('_', ' ').title()}** — {details}")
                st.caption(created)
        else:
            st.info("No activity yet.")

    st.divider()
    st.success("🔐 Your student details are stored locally in users.db.")

    if st.button("💬 Go to Study Assistant", type="primary"):
        st.session_state.page = "Study Assistant"
        st.rerun()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🎓 Student Panel")
    st.write(f"**{current_user['full_name']}**")
    st.caption(f"{current_user['course']} • {current_user['semester']}")

    st.divider()
    st.subheader("🧭 Navigation")
    st.session_state.page = st.radio(
        "Go to",
        ["Dashboard", "Study Assistant"],
        index=0 if st.session_state.page == "Dashboard" else 1,
        label_visibility="collapsed"
    )

    if st.button("🚪 Logout", use_container_width=True):
        log_activity(current_user["id"], "logout", "User logged out")
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()

    st.divider()
    st.subheader("📄 Add Study Material")

    st.divider()

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
            # KEEP ONLY USER PDF KNOWLEDGE
            # ------------------------------------------------

            all_documents = pdf_documents

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
            log_activity(
                current_user["id"],
                "upload",
                uploaded_file.name
            )

            st.rerun()

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    st.divider()

    st.subheader("📊 Status")

    st.write(
        f"📚 Knowledge sections: "
        f"{len(st.session_state.chunks)}"
    )

    st.write(
        f"💬 Questions asked: "
        f"{activity_count(current_user['id'], 'question')}"
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


if st.session_state.page == "Study Assistant":

    # ============================================================
    # WELCOME
    # ============================================================

    if not st.session_state.messages:

        st.info(
            "Upload your study PDFs from the sidebar, then ask questions "
            "from the uploaded material."
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
        "💬 Ask a question about your uploaded PDFs..."
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
        # RAG SEARCH
        # --------------------------------------------------------


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
