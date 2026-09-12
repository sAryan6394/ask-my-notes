from sentence_transformers import SentenceTransformer
import numpy as np
import os
from dotenv import load_dotenv
from google import genai
from ingestion import ingest_text_file, ingest_youtube, ingest_pdf, ingest_web_article

load_dotenv()
model = SentenceTransformer('all-MiniLM-L6-v2')


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


chunks = []
chunks += ingest_text_file("sample_notes.txt")
chunks += ingest_text_file("sample_notes2.txt")
chunks += ingest_youtube("dQw4w9WgXcQ")
chunks += ingest_pdf("real_sample.pdf")
chunks += ingest_web_article("https://en.wikipedia.org/wiki/Cellular_respiration")

for c in chunks:
    c["embedding"] = model.encode(c["text"])

question = "Where does cellular respiration happen in the cell?"
question_embedding = model.encode(question)

scored_chunks = []
for c in chunks:
    score = cosine_similarity(question_embedding, c["embedding"])
    scored_chunks.append((score, c))

scored_chunks.sort(key=lambda x: x[0], reverse=True)

top_k = 5
top_chunks = scored_chunks[:top_k]

print(f"Question: {question}\n")
print("Top matches:")
for score, chunk in top_chunks:
    print(f"  [{score:.4f}] {chunk['source_type']}: {chunk['source']} @ {chunk['location']}")
print()

combined_context = "\n\n".join(
    f"[Source: {c['source']} @ {c['location']}]\n{c['text']}"
    for score, c in top_chunks
)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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

ai_response = client.models.generate_content(
    model="gemini-3.6-flash",
    contents=prompt
)

print("--- AI Answer ---")
print(ai_response.text)