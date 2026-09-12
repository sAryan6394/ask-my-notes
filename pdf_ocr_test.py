from ingestion import ingest_pdf

chunks = ingest_pdf("real_sample.pdf")
print(f"Total chunks: {len(chunks)}")
for c in chunks[:3]:
    print(f"[{c['source']} @ {c['location']}] {c['text'][:80]}...")