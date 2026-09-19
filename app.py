import streamlit as st
import numpy as np
import os
import tempfile
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from ingestion import ingest_text_file, ingest_youtube, ingest_pdf
from vector_store import VectorStore
from hybrid_search import HybridSearch
from reranker import Reranker

load_dotenv()

st.set_page_config(page_title="Ask My Notes", page_icon="🔎", layout="centered")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stApp {
        background-color: #000000;
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 760px;
    }

    .app-header {
        font-size: 2.6rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.3rem;
        background: linear-gradient(90deg, #f5f5f5, #ef4444, #f5f5f5);
        background-size: 200% auto;
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 4s linear infinite;
    }

    @keyframes shimmer {
        0% { background-position: 0% center; }
        100% { background-position: 200% center; }
    }

    .app-subtitle {
        color: #9a9a9a;
        font-size: 0.95rem;
        text-align: center;
        margin-bottom: 2.5rem;
    }

    [data-testid="stChatMessage"] {
        background-color: #1c1c1c;
        border: 1px solid #2a2a2a;
        border-radius: 14px;
        padding: 0.7rem 1rem;
        margin-bottom: 0.7rem;
    }

    a, a:visited {
        color: #ef4444;
    }

    section[data-testid="stSidebar"] {
        background-color: #141414;
        border-right: 1px solid #2a2a2a;
    }

    [data-testid="stSidebarContent"] {
        background-color: #141414;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 500;
        background-color: #dc2626;
        color: white;
        border: none;
    }

    .stButton button:hover {
        background-color: #ef4444;
        color: white;
    }

    [data-testid="stChatInput"] {
        border-radius: 14px;
        background-color: #1c1c1c;
    }

    input, textarea {
        background-color: #1c1c1c !important;
        color: #f5f5f5 !important;
    }

    .stProgress > div > div {
        background-color: #ef4444;
    }

    .line-sidebar {
        padding-left: 70px;
        margin-top: 0.5rem;
    }

    .line-sidebar__item {
        position: relative;
        padding: 10px 0;
        cursor: default;
    }

    .line-sidebar__marker {
        position: absolute;
        top: 50%;
        left: -70px;
        height: 1px;
        width: 60px;
        background-color: #4a4a4a;
        transform: translateY(-50%) scaleX(0.7);
        transition: all 0.25s ease;
    }

    .line-sidebar__item:hover .line-sidebar__marker {
        background-color: #ef4444;
        transform: translateY(-50%) scaleX(1);
    }

    .line-sidebar__label {
        color: #9a9a9a;
        font-size: 0.9rem;
        transition: all 0.25s ease;
        display: inline-block;
    }

    .line-sidebar__item:hover .line-sidebar__label {
        color: #ef4444;
        transform: translateX(10px);
    }

    .line-sidebar__index {
        font-family: monospace;
        margin-right: 8px;
        opacity: 0.5;
        font-size: 0.8em;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


@st.cache_resource
def load_reranker():
    return Reranker()


model = load_model()
client = load_client()
reranker = load_reranker()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()  # loads from disk if present
if "hybrid_search" not in st.session_state:
    st.session_state.hybrid_search = HybridSearch(st.session_state.vector_store)
if "messages" not in st.session_state:
    st.session_state.messages = []


def add_chunks(new_chunks):
    if not new_chunks:
        return
    texts = [c["text"] for c in new_chunks]
    batch_embeddings = model.encode(texts)
    st.session_state.vector_store.add(new_chunks, batch_embeddings)


def extract_youtube_id(url_or_id):
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def build_contextual_query(current_question, history, max_history=2):
    recent_user_msgs = [m["content"] for m in history if m["role"] == "user"][-max_history:]
    return " ".join(recent_user_msgs + [current_question])


def format_history(history, max_turns=4):
    recent = history[-(max_turns * 2):]
    lines = []
    for m in recent:
        role = "User" if m["role"] == "user" else "Assistant"
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def render_line_sidebar():
    sources = list(dict.fromkeys(c["source"] for c in st.session_state.vector_store.chunks))
    if not sources:
        return

    parts = ['<div class="line-sidebar">']
    for i, src in enumerate(sources):
        display_name = src if len(src) < 30 else src[:27] + "..."
        parts.append(
            f'<div class="line-sidebar__item">'
            f'<span class="line-sidebar__marker"></span>'
            f'<span class="line-sidebar__label">'
            f'<span class="line-sidebar__index">{str(i+1).zfill(2)}</span>{display_name}'
            f'</span>'
            f'</div>'
        )
    parts.append('</div>')

    st.markdown("".join(parts), unsafe_allow_html=True)


st.markdown('<div class="app-header">Ask My Notes</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Ask questions across your PDFs, lectures, and articles — grounded, cited answers.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Sources")

    uploaded_files = st.file_uploader(
        "Upload PDF or text file",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if not any(f.name == c["source"] for c in st.session_state.vector_store.chunks)
        ]

        if new_files:
            total_files = len(new_files)
            progress_bar = st.progress(0, text="Starting...")

            for file_idx, uploaded in enumerate(new_files):
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                if suffix == ".pdf":
                    def page_progress(current_page, total_pages, fname=uploaded.name, idx=file_idx):
                        file_fraction = current_page / total_pages
                        overall = (idx + file_fraction) / total_files
                        progress_bar.progress(
                            overall,
                            text=f"Processing {fname} — page {current_page}/{total_pages}"
                        )
                    new_chunks = ingest_pdf(tmp_path, progress_callback=page_progress)
                else:
                    progress_bar.progress(
                        file_idx / total_files,
                        text=f"Processing {uploaded.name}..."
                    )
                    new_chunks = ingest_text_file(tmp_path)

                for c in new_chunks:
                    c["source"] = uploaded.name

                add_chunks(new_chunks)
                progress_bar.progress((file_idx + 1) / total_files, text=f"Added {uploaded.name}")

            progress_bar.progress(1.0, text="Done!")
            st.success(f"Added {total_files} file(s)")
            progress_bar.empty()

    yt_input = st.text_input("YouTube URL or ID", placeholder="Paste a link...")
    if st.button("Add video", use_container_width=True) and yt_input:
        video_id = extract_youtube_id(yt_input)
        already_added = any(video_id in c["source"] for c in st.session_state.vector_store.chunks)
        if not already_added:
            with st.spinner("Fetching transcript..."):
                new_chunks = ingest_youtube(video_id)
                add_chunks(new_chunks)
            st.success(f"Video added — {len(new_chunks)} chunks")

    st.divider()
    st.caption(f"{len(st.session_state.vector_store)} chunks loaded")
    render_line_sidebar()

    if len(st.session_state.vector_store) > 0:
        st.divider()
        if st.button("Clear all sources", use_container_width=True):
            st.session_state.vector_store.clear()
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question...")

if question:
    if len(st.session_state.vector_store) == 0:
        st.warning("Add at least one source first.")
    else:
        history_before = st.session_state.messages.copy()

        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        contextual_query = build_contextual_query(question, history_before)
        query_embedding = model.encode(contextual_query)
        candidates = st.session_state.hybrid_search.search(contextual_query, query_embedding, top_k=15)
        top_chunks = reranker.rerank(contextual_query, candidates, top_k=3)

        combined_context = "\n\n".join(
            f"[Source: {c['source']} @ {c['location']}]\n{c['text']}"
            for score, c in top_chunks
        )

        history_text = format_history(history_before)

        prompt = f"""You are answering questions using ONLY the context provided below.

Conversation so far:
{history_text if history_text else "(no earlier messages)"}

Context from sources:
{combined_context}

Current question: {question}

Instructions: If the current question is a follow-up (like "explain more", "why", "what about that", etc.), use the conversation above to understand what it's referring to, then answer using the context. Write a clear, complete, well-explained answer in plain prose — no source citations, no brackets, no meta-commentary about where the information came from. If the answer genuinely isn't in the context, say "I don't know based on the provided notes."

Answer:"""

        with st.chat_message("assistant"):
            with st.spinner("Searching your sources..."):
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
                answer = response.text
                st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})