import streamlit as st
import numpy as np
import os
import tempfile
import re
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from google import genai
from ingestion import ingest_text_file, ingest_youtube, ingest_pdf, ingest_web_article

load_dotenv()

st.set_page_config(page_title="Ask My Notes", page_icon="🔎", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer, header {visibility: hidden;}

    .block-container {
        padding-top: 2rem;
        max-width: 780px;
    }

    .app-header {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f5f5f7;
    margin-bottom: 0.2rem;
    }

    .app-subtitle {
        color: #9ca3af;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 0.5rem 0.9rem;
        margin-bottom: 0.6rem;
    }

    .source-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
        color: white;
    }

    .badge-pdf { background-color: #ef4444; }
    .badge-youtube { background-color: #dc2626; }
    .badge-web { background-color: #3b82f6; }
    .badge-text { background-color: #6b7280; }

    section[data-testid="stSidebar"] {
    background-color: #1a1c24;
    border-right: 1px solid #2d2f3a;
}

.stApp {
    background-color: #0e1117;
}

[data-testid="stSidebarContent"] {
    background-color: #1a1c24;
}

    .stButton button {
        border-radius: 8px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


@st.cache_resource
def load_client():
    return genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


model = load_model()
client = load_client()

if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "embeddings_matrix" not in st.session_state:
    st.session_state.embeddings_matrix = None
if "messages" not in st.session_state:
    st.session_state.messages = []


def add_chunks(new_chunks):
    if not new_chunks:
        return
    texts = [c["text"] for c in new_chunks]
    batch_embeddings = model.encode(texts)

    for c, emb in zip(new_chunks, batch_embeddings):
        c["embedding"] = emb

    st.session_state.chunks.extend(new_chunks)

    all_embeddings = np.array([c["embedding"] for c in st.session_state.chunks])
    st.session_state.embeddings_matrix = all_embeddings


def vectorized_search(question_embedding, top_k=5):
    matrix = st.session_state.embeddings_matrix
    q = question_embedding

    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(q)
    scores = np.dot(matrix, q) / norms

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(scores[i], st.session_state.chunks[i]) for i in top_indices]


def extract_youtube_id(url_or_id):
    match = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url_or_id)
    return match.group(1) if match else url_or_id.strip()


def badge_for(source_type):
    labels = {"pdf": "PDF", "youtube": "YouTube", "web": "Web", "text": "Notes"}
    return f'<span class="source-badge badge-{source_type}">{labels.get(source_type, source_type)}</span>'


st.markdown('<div class="app-header">Ask My Notes</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Ask questions across your PDFs, lectures, and articles — grounded, cited answers.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Sources")

    uploaded_files = st.file_uploader("Upload PDF or text file", type=["pdf", "txt"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        for uploaded in uploaded_files:
            already_added = any(uploaded.name == c["source"] for c in st.session_state.chunks)
            if not already_added:
                suffix = os.path.splitext(uploaded.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded.getvalue())
                    tmp_path = tmp.name

                new_chunks = ingest_pdf(tmp_path) if suffix == ".pdf" else ingest_text_file(tmp_path)
                for c in new_chunks:
                    c["source"] = uploaded.name

                add_chunks(new_chunks)
                st.success(f"{uploaded.name} — {len(new_chunks)} chunks")

    yt_input = st.text_input("YouTube URL or ID", placeholder="Paste a link...")
    if st.button("Add video", use_container_width=True) and yt_input:
        video_id = extract_youtube_id(yt_input)
        already_added = any(video_id in c["source"] for c in st.session_state.chunks)
        if not already_added:
            new_chunks = ingest_youtube(video_id)
            add_chunks(new_chunks)
            st.success(f"Video added — {len(new_chunks)} chunks")

    web_input = st.text_input("Article URL", placeholder="Paste a link...")
    if st.button("Add article", use_container_width=True) and web_input:
        already_added = any(web_input == c["source"] for c in st.session_state.chunks)
        if not already_added:
            new_chunks = ingest_web_article(web_input)
            add_chunks(new_chunks)
            st.success(f"Article added — {len(new_chunks)} chunks")

    st.divider()
    st.caption(f"{len(st.session_state.chunks)} chunks loaded")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

question = st.chat_input("Ask a question...")

if question:
    if not st.session_state.chunks:
        st.warning("Add at least one source first.")
    else:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        question_embedding = model.encode(question)
        top_chunks = vectorized_search(question_embedding, top_k=5)

        combined_context = "\n\n".join(
            f"[Source: {c['source']} @ {c['location']}]\n{c['text']}"
            for score, c in top_chunks
        )

        prompt = f"""Answer the question using ONLY the context below.
Multiple sources are provided, each labeled. Cite which source(s) you used in your answer.

After your answer, add a section called "Source Notes" where you:
- Point out if any sources add extra detail not mentioned by the others
- Point out if any sources appear to disagree or conflict with each other
- If all sources agree and no extra context exists, simply say "Sources are consistent."

If the answer isn't in the context, say "I don't know based on the provided notes."

Context:
{combined_context}

Question: {question}

Answer:"""

        with st.chat_message("assistant"):
            with st.spinner("Searching your sources..."):
                response = client.models.generate_content(
                    model="gemini-3.6-flash",
                    contents=prompt
                )
                answer = response.text
                st.markdown(answer)

                with st.expander("Sources used"):
                    for score, c in top_chunks:
                        st.markdown(
                            f"{badge_for(c['source_type'])} **{c['source']}** @ {c['location']} · score {score:.2f}",
                            unsafe_allow_html=True
                        )
                        st.caption(c['text'][:150] + "...")

        st.session_state.messages.append({"role": "assistant", "content": answer})