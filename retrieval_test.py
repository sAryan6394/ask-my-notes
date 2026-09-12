from sentence_transformers import SentenceTransformer
import numpy as np
from google import genai
from ingestion import ingest_text_file, ingest_youtube
import os
from dotenv import load_dotenv

model = SentenceTransformer('all-MiniLM-L6-v2')

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

chunks = []
chunks += ingest_text_file("sample_notes.txt")
chunks += ingest_youtube("dQw4w9WgXcQ") 

for c in chunks:
    c["embedding"] = model.encode(c["text"])

question = "How do cells release energy?"
question_embedding = model.encode(question)

scored_chunks = []
for c in chunks:
    score = cosine_similarity(question_embedding, c["embedding"])
    scored_chunks.append((score, c))

scored_chunks.sort(key=lambda x: x[0], reverse=True)

top_k = 3
top_chunks = scored_chunks[:top_k]

print(f"Question: {question}\n")
print("Top matches:")
for score, chunk in top_chunks:
    print(f"  [{score:.4f}] {chunk['source']} @ {chunk['location']}")
print()

combined_context = "\n\n".join(
    f"[Source: {c['source']} @ {c['location']}]\n{c['text']}"
    for score, c in top_chunks
)

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

prompt = f"""Answer the question using ONLY the context below.
Multiple sources are provided, each labeled. Cite which source(s) you used in your answer.
If the answer isn't in the context, say "I don't know based on the provided notes."

Context:
{combined_context}

question = "What does the song say about never giving up on someone?"

Answer:"""

ai_response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("--- AI Answer ---")
print(ai_response.text)