import streamlit as st
import numpy as np
import os
import tempfile
import re
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from ingestion import ingest_text_file, ingest_youtube, ingest_pdf
from vector_store import VectorStore
from hybrid_search import HybridSearch
from reranker import Reranker
import telemetry

load_dotenv()

st.set_page_config(page_title="Vantage", page_icon="🔎", layout="centered")


st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stToolbar"] {
        visibility: hidden;
    }

    .sidebar-toggle button {
        border-radius: 8px !important;
        padding: 0.25rem 0.6rem !important;
        font-size: 0.85rem !important;
    }

    [data-testid^="stChatMessageAvatar"] {
        display: none !important;
    }

    .stApp {
        background-color: #0F1512;
    }

    .block-container {
        padding-top: 2.5rem;
        max-width: 760px;
    }

    .app-header {
        font-size: 2.2rem;
        font-weight: 600;
        text-align: left;
        margin-bottom: 0.2rem;
        color: #DCE6DF;
        letter-spacing: -0.02em;
    }

    .app-subtitle {
        color: #6E7D74;
        font-size: 0.9rem;
        text-align: left;
        margin-bottom: 1.2rem;
    }

    .app-header-wrap {
        border-bottom: 1px solid #1E2A23;
        padding-bottom: 1.4rem;
        margin-bottom: 1.8rem;
    }

    [data-testid="stChatMessage"] {
        background-color: #1A2620;
        border: 1px solid #263029;
        border-radius: 12px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.25);
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background-color: transparent;
        border: none;
        box-shadow: none;
        padding: 0.4rem 1.1rem;
    }

    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) p {
        color: #8A968E;
        font-size: 0.92rem;
    }

    a, a:visited {
        color: #E8B34A;
    }

    section[data-testid="stSidebar"] {
        background-color: #0C120F;
        border-right: 1px solid #1E2A23;
        transition: width 0.3s ease, min-width 0.3s ease, opacity 0.2s ease;
        overflow: hidden;
    }

    [data-testid="stSidebarCollapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    button[aria-label="Collapse sidebar"],
    button[aria-label="Open sidebar"],
    button[aria-label="Close sidebar"] {
        display: none !important;
    }

    [data-testid="stSidebarContent"] {
        background-color: #0C120F;
    }

    section[data-testid="stSidebar"] h3 {
        color: #6E7D74;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.5rem;
    }

    .stButton button {
        border-radius: 10px;
        font-weight: 500;
        background-color: #1A2620;
        color: #E8B34A;
        border: 1px solid #263029;
    }

    .stButton button:hover {
        background-color: #263029;
        border-color: #E8B34A;
        color: #E8B34A;
    }

    .stButton button[kind="primary"] {
        background-color: #E8B34A;
        color: #0F1512;
        border: 1px solid #E8B34A;
    }

    .stButton button[kind="primary"]:hover {
        background-color: #F2C368;
        border-color: #F2C368;
        color: #0F1512;
    }

    [data-testid="stChatInput"] {
        border-radius: 999px;
        background-color: #1A2620;
        border: 1px solid #263029;
    }

    [data-testid="stChatInput"]:focus-within {
        border-color: #E8B34A !important;
        box-shadow: none !important;
    }

    input, textarea {
        background-color: #1A2620 !important;
        color: #DCE6DF !important;
        outline: none !important;
        box-shadow: none !important;
    }

    input:focus, textarea:focus {
        outline: none !important;
        box-shadow: none !important;
        border-color: #E8B34A !important;
    }

    [data-baseweb="base-input"]:focus-within {
        border-color: #E8B34A !important;
        box-shadow: none !important;
    }

    .stProgress > div > div {
        background-color: #E8B34A;
    }

    .line-sidebar {
        padding-left: 26px;
        margin-top: 0.5rem;
    }

    .line-sidebar__item {
        position: relative;
        padding: 8px 0;
        cursor: default;
    }

    .line-sidebar__marker {
        position: absolute;
        top: 50%;
        left: -26px;
        height: 1px;
        width: 16px;
        background-color: #3A473F;
        transform: translateY(-50%);
        transition: all 0.2s ease;
    }

    .line-sidebar__item:hover .line-sidebar__marker {
        background-color: #E8B34A;
        width: 20px;
    }

    .line-sidebar__label {
        color: #8A968E;
        font-size: 0.87rem;
        transition: all 0.2s ease;
        display: inline-block;
    }

    .line-sidebar__item:hover .line-sidebar__label {
        color: #DCE6DF;
    }

    .line-sidebar__index {
        font-family: 'JetBrains Mono', monospace;
        margin-right: 8px;
        opacity: 0.5;
        font-size: 0.8em;
    }

    .confidence-meter {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 0.9rem;
        padding-top: 0.7rem;
        border-top: 1px solid #22302A;
    }

    .confidence-meter__track {
        flex: 0 0 90px;
        height: 4px;
        border-radius: 2px;
        background-color: #263029;
        overflow: hidden;
    }

    .confidence-meter__fill {
        height: 100%;
        background-color: #E8B34A;
        border-radius: 2px;
    }

    .confidence-meter__value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: #6E7D74;
    }

    .error-card {
        color: #C97A63;
        font-size: 0.92rem;
        padding: 0.4rem 0;
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

telemetry.init_db()

if "vector_store" not in st.session_state:
    st.session_state.vector_store = VectorStore()  # loads from disk if present
if "hybrid_search" not in st.session_state:
    st.session_state.hybrid_search = HybridSearch(st.session_state.vector_store)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True


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


# --- Sidebar visibility: animate width/opacity instead of toggling
# display:none <-> block (display can't be CSS-transitioned).
if st.session_state.sidebar_visible:
    st.markdown(
        '''<style>
        section[data-testid="stSidebar"] {
            width: 21rem !important;
            min-width: 21rem !important;
            margin-left: 0 !important;
            transform: none !important;
            opacity: 1 !important;
        }
        </style>''',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '''<style>
        section[data-testid="stSidebar"] {
            width: 0rem !important;
            min-width: 0rem !important;
            opacity: 0 !important;
            padding: 0 !important;
            border: none !important;
        }
        </style>''',
        unsafe_allow_html=True
    )

st.markdown('<div class="sidebar-toggle">', unsafe_allow_html=True)
toggle_label = "Hide sources" if st.session_state.sidebar_visible else "Show sources"
if st.button(toggle_label, key="sidebar_toggle_btn"):
    st.session_state.sidebar_visible = not st.session_state.sidebar_visible
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
<div class="app-header-wrap">
    <div class="app-header">Vantage</div>
    <div class="app-subtitle">Ask questions across your PDFs, lectures, and articles — grounded, cited answers.</div>
</div>
""", unsafe_allow_html=True)

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
    if st.button("Add video", use_container_width=True, type="primary") and yt_input:
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

        retrieval_start = time.perf_counter()
        # Separate, cheap dense-only lookup purely for a true 0-1 cosine
        # similarity score to log — the hybrid/reranked results below use
        # RRF and cross-encoder scores, which aren't on a comparable scale.
        dense_top = st.session_state.vector_store.search(query_embedding, top_k=1)
        top_similarity_score = dense_top[0][0] if dense_top else 0.0

        candidates = st.session_state.hybrid_search.search(contextual_query, query_embedding, top_k=15)
        top_chunks = reranker.rerank(contextual_query, candidates, top_k=3)
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

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
                # gemini-3.1-flash-lite: ~3x the rate limit of 3.6-flash and
                # far cheaper, fine for grounded QA over retrieved context
                # (no multi-step reasoning needed here). Retry covers
                # transient 503s, which can happen on any tier.
                GENERATION_MODEL = "gemini-3.1-flash-lite"
                max_retries = 2
                last_error = None
                generation_start = time.perf_counter()
                response = None

                for attempt in range(max_retries + 1):
                    try:
                        response = client.models.generate_content(
                            model=GENERATION_MODEL,
                            contents=prompt
                        )
                        last_error = None
                        break
                    except Exception as e:
                        last_error = e
                        print(f"[Vantage] Generation attempt {attempt + 1} failed: {e}")
                        if attempt < max_retries:
                            time.sleep(1.5 * (attempt + 1))

                generation_latency_ms = (time.perf_counter() - generation_start) * 1000

                if last_error is not None:
                    st.markdown(
                        '<div class="error-card">'
                        "Couldn't reach the AI model just now — this is usually temporary "
                        "(the API may be under heavy load). Try asking again in a moment."
                        '</div>',
                        unsafe_allow_html=True
                    )
                    st.stop()

                answer = response.text

                st.markdown(answer)

                confidence_pct = max(0.0, min(1.0, top_similarity_score)) * 100
                st.markdown(
                    f'<div class="confidence-meter">'
                    f'<div class="confidence-meter__track">'
                    f'<div class="confidence-meter__fill" style="width:{confidence_pct:.0f}%"></div>'
                    f'</div>'
                    f'<span class="confidence-meter__value">{top_similarity_score:.2f} similarity</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0

                query_id = telemetry.log_query(
                    user_query=question,
                    retrieval_latency_ms=retrieval_latency_ms,
                    generation_latency_ms=generation_latency_ms,
                    top_similarity_score=top_similarity_score,
                    retrieved_chunks_count=len(top_chunks),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
                st.session_state.last_query_id = query_id
                st.session_state.feedback_given = False

        st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.get("last_query_id") and not st.session_state.get("feedback_given"):
    fb_col1, fb_col2, _ = st.columns([1, 1, 8])
    with fb_col1:
        if st.button("👍", key=f"fb_up_{st.session_state.last_query_id}"):
            telemetry.update_feedback(st.session_state.last_query_id, 1)
            st.session_state.feedback_given = True
            st.rerun()
    with fb_col2:
        if st.button("👎", key=f"fb_down_{st.session_state.last_query_id}"):
            telemetry.update_feedback(st.session_state.last_query_id, -1)
            st.session_state.feedback_given = True
            st.rerun()
elif st.session_state.get("feedback_given"):
    st.caption("Thanks for the feedback!")