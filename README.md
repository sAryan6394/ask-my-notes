# Ask-My-Notes

A multi-source RAG (Retrieval-Augmented Generation) knowledge base that lets you 
ask questions across your PDFs, YouTube lecture transcripts, and text notes — 
with answers grounded in your actual sources and cited by location (page number, 
video timestamp, etc.), not hallucinated.

## How it works
- Documents are split into chunks and tagged with their source and location
- Each chunk is converted into an embedding (a vector representing its meaning)
- A question is compared against every chunk using cosine similarity — no keyword 
  matching, purely meaning-based
- The top matching chunks are passed to an AI model, which is instructed to answer 
  only from that retrieved context, citing which source it used

## Stack
Python, sentence-transformers, numpy, Google Gemini API, youtube-transcript-api

## Status
Core retrieval + generation pipeline working. In progress: PDF/OCR support, 
Streamlit interface, deployment.