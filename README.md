# Ask-My-Notes

A multi-source RAG (Retrieval-Augmented Generation) knowledge base that lets you 
ask questions across your PDFs, YouTube lecture transcripts, and web articles — 
with answers grounded in your actual sources, not hallucinated.

## Features
- Multi-source ingestion: PDFs (including scanned/OCR), YouTube transcripts, web articles, plain text
- Meaning-based retrieval using sentence embeddings and cosine similarity (not keyword matching)
- Grounded AI answers using Google's Gemini API
- Conversational follow-ups — remembers recent context for questions like "explain more"
- Persistent storage — sources stay loaded across sessions
- Clean, custom-themed chat interface built with Streamlit

## How it works
1. Documents are split into chunks and tagged with their source and location (page number, video timestamp, etc.)
2. Each chunk is converted into an embedding — a vector representing its meaning
3. A question is compared against every chunk using cosine similarity
4. The top matching chunks are passed to an AI model, instructed to answer only from that retrieved context

## Stack
Python, Streamlit, sentence-transformers, numpy, Google Gemini API, PyMuPDF, pytesseract, youtube-transcript-api, trafilatura

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
Requires a `.env` file with `GOOGLE_API_KEY=your_key_here`, and Tesseract OCR installed for scanned PDF support.