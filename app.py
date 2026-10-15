import streamlit as st
import hashlib
import hmac
import os
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
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_updated ON conversations(user_id, updated_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_messages_conversation ON chat_messages(conversation_id, id)")

def hash_password(password):
    salt = os.urandom(16).hex()
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
    # Upgrade accounts created by the earlier deterministic-salt implementation.
    old_salt = hashlib.sha256(password.encode()).hexdigest()[:32]
    if row[7].split("$", 1)[0] == old_salt:
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(password), row[0]))
    return {"id":row[0],"full_name":row[1],"email":row[2],"student_id":row[3],
            "college":row[4],"course":row[5],"semester":row[6],"created_at":row[8]}

def save_chat_exchange(user_id, conversation_id, question, answer):
    now = datetime.now().isoformat(timespec="seconds")
    with sqlite3.connect(DB_PATH) as conn:
        if conversation_id is None:
            title = " ".join(question.split())[:100] or "Study conversation"
            cursor = conn.execute(
                "INSERT INTO conversations (user_id,title,created_at,updated_at) VALUES (?,?,?,?)",
                (user_id, title, now, now)
            )
            conversation_id = cursor.lastrowid
        else:
            owner = conn.execute(
                "SELECT id FROM conversations WHERE id=? AND user_id=?",
                (conversation_id, user_id)
            ).fetchone()
            if owner is None:
                raise ValueError("Conversation does not belong to the signed-in user.")
            conn.execute(
                "UPDATE conversations SET updated_at=? WHERE id=? AND user_id=?",
                (now, conversation_id, user_id)
            )
        conn.executemany(
            "INSERT INTO chat_messages (conversation_id,role,content,created_at) VALUES (?,?,?,?)",
            [(conversation_id, "user", question, now),
             (conversation_id, "assistant", answer, now)]
        )
    return conversation_id

def list_conversations(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT id,title,created_at,updated_at FROM conversations WHERE user_id=? ORDER BY updated_at DESC",
            (user_id,)
        ).fetchall()

def load_conversation(user_id, conversation_id):
    with sqlite3.connect(DB_PATH) as conn:
        owner = conn.execute(
            "SELECT id FROM conversations WHERE id=? AND user_id=?",
            (conversation_id, user_id)
        ).fetchone()
        if owner is None:
            return None
        return conn.execute(
            "SELECT m.role,m.content FROM chat_messages AS m "
            "JOIN conversations AS c ON c.id=m.conversation_id "
            "WHERE c.id=? AND c.user_id=? ORDER BY m.id",
            (conversation_id, user_id)
        ).fetchall()

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

def reset_user_session():
    for key in ("chunks", "index", "messages", "file_name", "pdf_processed", "page", "current_conversation_id"):
        st.session_state.pop(key, None)

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
                        reset_user_session()
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

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

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

    if st.button("🗂️ View Study History", type="primary"):
        st.session_state.page = "History"
        st.rerun()

elif st.session_state.page == "History":
    st.header("🗂️ Your Study History")
    st.caption("Saved questions and answers from your study sessions")
    conversations = list_conversations(current_user["id"])
    if not conversations:
        st.info("Your saved conversations will appear here after you ask a question.")
    else:
        for conversation in conversations:
            with st.container():
                st.subheader(conversation["title"])
                st.caption(f"Last activity: {conversation['updated_at']}")
                if st.button("Open conversation", key=f"open_conversation_{conversation['id']}"):
                    messages = load_conversation(current_user["id"], conversation["id"])
                    if messages is not None:
                        st.session_state.messages = [
                            {"role": row[0], "content": row[1]} for row in messages
                        ]
                        st.session_state.current_conversation_id = conversation["id"]
                        st.session_state.page = "Study Assistant"
                        st.rerun()
                st.divider()

    if st.button("➕ Start a new chat"):
        st.session_state.messages = []
        st.session_state.current_conversation_id = None
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
        ["Dashboard", "Study Assistant", "History"],
        index={"Dashboard": 0, "Study Assistant": 1, "History": 2}.get(st.session_state.page, 1),
        label_visibility="collapsed"
    )

    if st.button("🚪 Logout", use_container_width=True):
        log_activity(current_user["id"], "logout", "User logged out")
        reset_user_session()
        st.session_state.authenticated = False
        st.session_state.user = None
        st.rerun()

    st.divider()
    st.subheader("📄 Add Study Material")

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key=f"pdf_upload_{current_user['id']}"
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
        "➕ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []
        st.session_state.current_conversation_id = None

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

        if st.session_state.index is None:
            st.error("Knowledge base is not ready. Upload a PDF from the sidebar first.")
            st.stop()

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

        st.session_state.current_conversation_id = save_chat_exchange(
            current_user["id"],
            st.session_state.current_conversation_id,
            question,
            answer
        )
        log_activity(current_user["id"], "question", question)
