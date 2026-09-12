def chunk_text(text, source_name, chunk_size=200):
    """
    Splits text into chunks of roughly chunk_size characters,
    and tags each chunk with where it came from.
    """
    chunks = []
    for i in range(0, len(text), chunk_size):
        piece = text[i:i + chunk_size]
        chunks.append({
            "text": piece,
            "source": source_name,
            "position": i  # character offset — stand-in for page/timestamp later
        })
    return chunks


sample_text = """
Photosynthesis is the process by which plants convert light energy into chemical energy.
It occurs in the chloroplasts of plant cells. The process uses carbon dioxide and water,
producing glucose and oxygen as byproducts. This is essential for life on Earth because
it produces the oxygen we breathe and forms the base of most food chains.
"""

chunks = chunk_text(sample_text, source_name="biology_notes.pdf")

for c in chunks:
    print(f"[{c['source']} @ {c['position']}] {c['text'][:50]}...")