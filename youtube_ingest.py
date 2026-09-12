from youtube_transcript_api import YouTubeTranscriptApi

def format_timestamp(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}:{s:02d}"

def get_youtube_chunks(video_id, chunk_seconds=60):
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
                "timestamp": format_timestamp(current_start)
            })
            current_text = ""
            current_start = snippet.start
        current_text += " " + snippet.text

    if current_text:
        chunks.append({
            "text": current_text.strip(),
            "source": f"youtube:{video_id}",
            "timestamp": format_timestamp(current_start)
        })

    return chunks


video_id = "dQw4w9WgXcQ"
chunks = get_youtube_chunks(video_id)

print(f"Total chunks: {len(chunks)}")
for c in chunks[:3]:
    print(f"[{c['source']} @ {c['timestamp']}] {c['text'][:80]}...")