from youtube_transcript_api import YouTubeTranscriptApi

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