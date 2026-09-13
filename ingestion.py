import os
import io
import pymupdf as fitz
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from google import genai
from youtube_transcript_api import YouTubeTranscriptApi
import trafilatura

load_dotenv()
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def format_timestamp(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"


def ingest_text_file(filepath, chunk_size=200):
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = []
    for i in range(0, len(text), chunk_size):
        piece = text[i:i + chunk_size]
        if piece.strip():
            chunks.append({
                "text": piece,
                "source": filepath,
                "source_type": "text",
                "location": f"position {i}"
            })
    return chunks


def ingest_youtube(video_id, chunk_seconds=60):
    ytt_api = YouTubeTranscriptApi()
    transcript = ytt_api.fetch(video_id)

    chunks = []
    current_text = ""
    current_start = transcript.snippets[0].start

    for snippet in transcript.snippets:
        if snippet.start - current_start > chunk_seconds and current_text:
            chunks.append({
                "text": current_text.strip(),
                "source": f"youtube:{video_id}",
                "source_type": "youtube",
                "location": format_timestamp(current_start)
            })
            current_text = ""
            current_start = snippet.start
        current_text += " " + snippet.text

    if current_text:
        chunks.append({
            "text": current_text.strip(),
            "source": f"youtube:{video_id}",
            "source_type": "youtube",
            "location": format_timestamp(current_start)
        })
    return chunks


def ingest_pdf(filepath, chunk_size=200, progress_callback=None):
    doc = fitz.open(filepath)
    chunks = []
    total_pages = len(doc)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text()

        if not text.strip():
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)

        for i in range(0, len(text), chunk_size):
            piece = text[i:i + chunk_size]
            if piece.strip():
                chunks.append({
                    "text": piece,
                    "source": filepath,
                    "source_type": "pdf",
                    "location": f"page {page_num + 1}"
                })

        if progress_callback:
            progress_callback(page_num + 1, total_pages)

    return chunks


def ingest_web_article(url, chunk_size=200):
    downloaded = trafilatura.fetch_url(url)
    text = trafilatura.extract(downloaded)

    if not text:
        return []

    chunks = []
    for i in range(0, len(text), chunk_size):
        piece = text[i:i + chunk_size]
        if piece.strip():
            chunks.append({
                "text": piece,
                "source": url,
                "source_type": "web",
                "location": f"position {i}"
            })
    return chunks