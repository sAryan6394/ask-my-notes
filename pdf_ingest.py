def chunk_text(text, source_name, chunk_size=200):
    chunks = []
    for i in range(0, len(text), chunk_size):
        piece = text[i:i + chunk_size]
        chunks.append({"text": piece, "source": source_name, "position": i})
    return chunks

with open("sample_notes.txt", "r", encoding="utf-8") as f:
    content = f.read()

chunks = chunk_text(content, source_name="sample_notes.txt")

print(f"Total chunks: {len(chunks)}")
for c in chunks:
    print(f"[{c['source']} @ {c['position']}] {c['text'][:60]}...")