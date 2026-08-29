import base64
import hashlib
import hmac
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from utils.pdf_loader import extract_pages_from_pdf
from utils.chunker import split_pages_into_chunks
from utils.embeddings import create_embeddings, create_query_embedding
from utils.vector_store import create_vector_store, search_vector_store
from utils.rag_pipeline import generate_answer

try:
    from utils.summarizer import summarize_pdf
except ImportError:
    summarize_pdf = None

st.set_page_config(
    page_title="AI College Study Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
PDF_DIR = APP_DIR / "uploaded_pdfs"
PDF_DIR.mkdir(exist_ok=True)
DB_PATH = APP_DIR / "users.db"


# ---------------- DATABASE ----------------

def db():
    return sqlite3.connect(DB_PATH)


def init_db():
    with db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                student_id TEXT NOT NULL,
                college TEXT NOT NULL,
                course TEXT NOT NULL,
                semester TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activity (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                activity_type TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()


def hash_password(password):
    salt = hashlib.sha256(password.encode()).hexdigest()[:32]
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode(), salt.encode(), 120000
    ).hex()
    return f"{salt}${digest}"


def verify_password(password, stored):
    try:
        salt, saved = stored.split("$", 1)
        digest = hashlib.pbkdf2_hmac(
            "sha256", password.encode(), salt.encode(), 120000
        ).hex()
        return hmac.compare_digest(digest, saved)
    except Exception:
        return False


def register_user(name, email, student_id, college, course, semester, password):
    try:
        with db() as conn:
            conn.execute("""
                INSERT INTO users
                (full_name,email,student_id,college,course,semester,password_hash,created_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                name.strip(), email.strip().lower(), student_id.strip(),
                college.strip(), course.strip(), semester,
                hash_password(password),
                datetime.now().isoformat(timespec="seconds"),
            ))
            conn.commit()
        return True, "Account created successfully."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."
    except Exception as e:
        return False, str(e)


def login_user(email, password):
    with db() as conn:
        row = conn.execute("""
            SELECT id,full_name,email,student_id,college,course,semester,
                   password_hash,created_at
            FROM users WHERE email=?
        """, (email.strip().lower(),)).fetchone()

    if not row or not verify_password(password, row[7]):
        return None

    return {
        "id": row[0],
        "full_name": row[1],
        "email": row[2],
        "student_id": row[3],
        "college": row[4],
        "course": row[5],
        "semester": row[6],
        "created_at": row[8],
    }


def activity(user_id, kind, details=""):
    try:
        with db() as conn:
            conn.execute("""
                INSERT INTO activity(user_id,activity_type,details,created_at)
                VALUES(?,?,?,?)
            """, (
                user_id, kind, details,
                datetime.now().isoformat(timespec="seconds"),
            ))
            conn.commit()
    except Exception:
        pass


def stats(user_id):
    result = {
        "question": 0, "upload": 0, "quiz": 0, "exam_question": 0
    }
    with db() as conn:
        rows = conn.execute("""
            SELECT activity_type, COUNT(*)
            FROM activity WHERE user_id=?
            GROUP BY activity_type
        """, (user_id,)).fetchall()
    for kind, count in rows:
        if kind in result:
            result[kind] = count
    return result


def recent_activity(user_id):
    with db() as conn:
        return conn.execute("""
            SELECT activity_type, details, created_at
            FROM activity WHERE user_id=?
            ORDER BY id DESC LIMIT 8
        """, (user_id,)).fetchall()


init_db()


# ---------------- SESSION ----------------

defaults = {
    "authenticated": False,
    "user": None,
    "page": "Dashboard",
    "chunks": [],
    "index": None,
    "messages": [],
    "files": [],
    "summary": None,
    "summary_file": None,
    "viewer_file": None,
    "viewer_page": 1,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- LOGIN / SIGNUP ----------------

def auth_page():
    st.markdown(
        "<h1 style='text-align:center'>🎓 AI College Study Assistant</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center;color:gray'>Login to access your personal study dashboard</p>",
        unsafe_allow_html=True,
    )

    _, center, _ = st.columns([1, 2, 1])

    with center:
        login_tab, signup_tab = st.tabs(["🔐 Login", "📝 Sign Up"])

        with login_tab:
            email = st.text_input("Email", key="login_email")
            password = st.text_input(
                "Password", type="password", key="login_password"
            )

            if st.button("Login", type="primary", use_container_width=True):
                if not email or not password:
                    st.warning("Enter your email and password.")
                else:
                    user = login_user(email, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.session_state.page = "Dashboard"
                        activity(user["id"], "login", "User logged in")
                        st.rerun()
                    else:
                        st.error("Invalid email or password.")

        with signup_tab:
            name = st.text_input("Full Name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            student_id = st.text_input(
                "Student ID / Roll Number", key="signup_id"
            )
            college = st.text_input("College Name", key="signup_college")
            course = st.text_input(
                "Course / Branch",
                key="signup_course",
                placeholder="Example: B.Tech CSE",
            )
            semester = st.selectbox(
                "Semester",
                [f"{i}{'st' if i==1 else 'nd' if i==2 else 'rd' if i==3 else 'th'}"
                 for i in range(1, 9)],
                key="signup_semester",
            )
            password = st.text_input(
                "Password", type="password", key="signup_password"
            )
            confirm = st.text_input(
                "Confirm Password",
                type="password",
                key="signup_confirm",
            )

            if st.button(
                "Create Account",
                type="primary",
                use_container_width=True,
            ):
                if not all([name, email, student_id, college, course, password, confirm]):
                    st.warning("Please fill in all fields.")
                elif password != confirm:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must contain at least 6 characters.")
                else:
                    ok, msg = register_user(
                        name, email, student_id, college,
                        course, semester, password
                    )
                    if ok:
                        st.success(msg + " You can now log in.")
                    else:
                        st.error(msg)


if not st.session_state.authenticated:
    auth_page()
    st.stop()

user = st.session_state.user


# ---------------- STYLE ----------------

st.markdown("""
<style>
.block-container {max-width:1250px;padding-top:1.5rem;}
.main-title {font-size:2.4rem;font-weight:700;}
.subtitle {color:#777;font-size:1.05rem;}
</style>
""", unsafe_allow_html=True)


# ---------------- PDF / RAG HELPERS ----------------

def save_pdf(uploaded_file):
    path = PDF_DIR / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return path


def process_pdf(uploaded_file):
    pages = extract_pages_from_pdf(uploaded_file)
    if not pages:
        raise ValueError("Could not extract text from this PDF.")

    chunks = split_pages_into_chunks(pages)
    if not chunks:
        raise ValueError("Could not create chunks from this PDF.")

    return [
        {
            "source": uploaded_file.name,
            "page": chunk.get("page", 1),
            "text": chunk.get("text", ""),
        }
        for chunk in chunks
    ]


def rebuild_index(documents):
    texts = [d["text"] for d in documents if d.get("text")]
    if not texts:
        raise ValueError("No text was available for embeddings.")
    return create_vector_store(create_embeddings(texts))


def show_pdf(source, page):
    path = PDF_DIR / source
    if not path.exists():
        st.error(f"PDF file not found: {source}")
        return

    try:
        page = max(1, int(page))
    except Exception:
        page = 1

    encoded = base64.b64encode(path.read_bytes()).decode()
    url = f"data:application/pdf;base64,{encoded}#page={page}"

    components.html(
        f"""<iframe src="{url}" width="100%" height="750"
        style="border:1px solid #ddd;border-radius:10px;"></iframe>""",
        height=770,
        scrolling=True,
    )


def open_source(source, page):
    st.session_state.viewer_file = source
    try:
        st.session_state.viewer_page = int(page)
    except Exception:
        st.session_state.viewer_page = 1
    st.rerun()


# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.markdown("## 🎓 AI College Assistant")
    st.write(f"Welcome, **{user['full_name']}**")
    st.divider()

    options = ["Dashboard", "Study Assistant", "Study Material"]
    st.session_state.page = st.radio(
        "Navigation",
        options,
        index=options.index(st.session_state.page),
    )

    st.divider()
    st.markdown("### 👤 Student")
    st.write(f"**Name:** {user['full_name']}")
    st.write(f"**Student ID:** {user['student_id']}")
    st.write(f"**College:** {user['college']}")
    st.write(f"**Course:** {user['course']}")
    st.write(f"**Semester:** {user['semester']}")

    st.divider()

    if st.button("🚪 Logout", use_container_width=True):
        activity(user["id"], "logout", "User logged out")
        for key in [
            "authenticated", "user", "chunks", "index", "messages",
            "files", "summary", "summary_file", "viewer_file"
        ]:
            st.session_state.pop(key, None)
        st.rerun()


# ---------------- DASHBOARD ----------------

if st.session_state.page == "Dashboard":
    st.markdown(
        '<div class="main-title">📊 Student Dashboard</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Your personal study overview</div>',
        unsafe_allow_html=True,
    )

    s = stats(user["id"])

    st.info(
        f"👋 Welcome **{user['full_name']}**! "
        "Use the sidebar to study from your PDFs."
    )

    a, b, c, d = st.columns(4)
    a.metric("Questions Asked", s["question"])
    b.metric("PDF Uploads", s["upload"])
    c.metric("Quizzes", s["quiz"])
    d.metric("Exam Questions", s["exam_question"])

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("👤 Student Details")
        st.write(f"**Full Name:** {user['full_name']}")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**Student ID:** {user['student_id']}")
        st.write(f"**College:** {user['college']}")
        st.write(f"**Course:** {user['course']}")
        st.write(f"**Semester:** {user['semester']}")

    with right:
        st.subheader("🕒 Recent Activity")
        rows = recent_activity(user["id"])
        labels = {
            "login": "🔐 Login",
            "logout": "🚪 Logout",
            "question": "💬 Question",
            "upload": "📄 PDF Upload",
            "quiz": "🧠 Quiz",
            "exam_question": "📝 Exam Question",
        }

        if not rows:
            st.caption("No activity recorded yet.")
        else:
            for kind, details, created in rows:
                st.write(
                    f"**{labels.get(kind, kind.title())}** "
                    f"— {details or ''}"
                )
                st.caption(created)

    st.divider()
    st.subheader("🚀 Quick Start")

    x, y, z = st.columns(3)
    with x:
        if st.button("💬 Ask Questions", use_container_width=True):
            st.session_state.page = "Study Assistant"
            st.rerun()
    with y:
        if st.button("📄 Add Study Material", use_container_width=True):
            st.session_state.page = "Study Material"
            st.rerun()
    with z:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


# ---------------- STUDY MATERIAL ----------------

elif st.session_state.page == "Study Material":
    st.markdown(
        '<div class="main-title">📚 Study Material</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Upload your college notes and PDFs</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload notes, textbooks, syllabi, or study material.",
    )

    if uploaded_file:
        st.caption(
            f"📄 {uploaded_file.name} • "
            f"{uploaded_file.size / (1024*1024):.2f} MB"
        )

        if st.button(
            "➕ Add PDF to Knowledge Base",
            type="primary",
            use_container_width=True,
        ):
            try:
                progress = st.progress(0, text="Saving PDF...")
                save_pdf(uploaded_file)

                progress.progress(25, text="Extracting PDF text...")
                new_docs = process_pdf(uploaded_file)

                old_docs = [
                    d for d in st.session_state.chunks
                    if d["source"] != uploaded_file.name
                ]
                all_docs = old_docs + new_docs

                progress.progress(50, text="Creating embeddings...")
                new_index = rebuild_index(all_docs)

                progress.progress(100, text="Knowledge base ready!")

                st.session_state.chunks = all_docs
                st.session_state.index = new_index

                if uploaded_file.name not in st.session_state.files:
                    st.session_state.files.append(uploaded_file.name)

                activity(
                    user["id"],
                    "upload",
                    uploaded_file.name,
                )

                st.success("✅ PDF added successfully.")

            except Exception as e:
                st.error("❌ PDF processing failed.")
                st.exception(e)

    st.divider()

    a, b = st.columns(2)
    a.metric("PDFs", len(st.session_state.files))
    b.metric("RAG Chunks", len(st.session_state.chunks))

    if st.session_state.files:
        st.subheader("📁 Uploaded PDFs")
        for filename in st.session_state.files:
            st.write(f"📄 {filename}")

        if summarize_pdf is not None:
            st.divider()
            st.subheader("📝 PDF Summary")
            selected = st.selectbox(
                "Select PDF",
                st.session_state.files,
            )

            if st.button("✨ Generate Summary", use_container_width=True):
                docs = [
                    d for d in st.session_state.chunks
                    if d["source"] == selected
                ]
                try:
                    with st.spinner("🧠 Generating summary..."):
                        st.session_state.summary = summarize_pdf(docs)
                    st.session_state.summary_file = selected
                    st.success("✅ Summary generated.")
                except Exception as e:
                    st.error("❌ Summary failed.")
                    st.exception(e)

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


# ---------------- RAG CHAT ----------------

elif st.session_state.page == "Study Assistant":
    st.markdown(
        '<div class="main-title">💬 Study Assistant</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="subtitle">Ask questions from your uploaded study material</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.index is None:
        st.info(
            "📄 No PDF is loaded yet. Go to **Study Material** and upload a PDF."
        )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input(
        "💬 Ask a question about your PDFs..."
    )

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        activity(
            user["id"],
            "question",
            question[:200],
        )

        with st.chat_message("assistant"):
            if st.session_state.index is None:
                answer = "📄 Please upload a PDF first."
                st.warning(answer)

            else:
                try:
                    with st.spinner("🔎 Searching your documents..."):
                        query_embedding = create_query_embedding(question)
                        results = search_vector_store(
                            st.session_state.index,
                            query_embedding,
                            st.session_state.chunks,
                            top_k=3,
                        )

                    if not results:
                        answer = (
                            "I couldn't find relevant information "
                            "in your PDFs."
                        )
                        st.warning(answer)

                    else:
                        context = "

".join(
                            f"SOURCE: {r['source']}\n"
                            f"PAGE: {r['page']}\n"
                            f"CONTENT:\n{r['text']}"
                            for r in results
                        )

                        with st.spinner("🤖 Generating answer..."):
                            answer = generate_answer(
                                question,
                                context,
                            )

                        st.markdown(answer)
                        st.divider()
                        st.subheader("📚 Sources")

                        displayed = set()

                        for i, result in enumerate(results):
                            source = result["source"]
                            page = result["page"]
                            key = (source, page)

                            if key in displayed:
                                continue
                            displayed.add(key)

                            c1, c2 = st.columns([5, 1])
                            with c1:
                                st.markdown(
                                    f"📄 **{source}** — Page **{page}**"
                                )
                            with c2:
                                if st.button(
                                    "Open",
                                    key=f"open_{i}_{source}_{page}",
                                    use_container_width=True,
                                ):
                                    open_source(source, page)

                        with st.expander("🔎 View Retrieved Information"):
                            for i, result in enumerate(results):
                                st.markdown(f"### Source {i+1}")
                                st.caption(
                                    f"{result['source']} • Page {result['page']}"
                                )
                                st.write(result["text"])
                                if i < len(results) - 1:
                                    st.divider()

                except Exception as e:
                    answer = (
                        "❌ Something went wrong while answering the question."
                    )
                    st.error(answer)
                    st.exception(e)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
        })


# ---------------- SOURCE VIEWER ----------------

if st.session_state.viewer_file:
    st.divider()
    st.subheader("📖 Source Document")
    st.caption(
        f"{st.session_state.viewer_file} • "
        f"Page {st.session_state.viewer_page}"
    )

    show_pdf(
        st.session_state.viewer_file,
        st.session_state.viewer_page,
    )

    if st.button("✕ Close Document"):
        st.session_state.viewer_file = None
        st.rerun()
