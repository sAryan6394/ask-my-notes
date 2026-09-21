import sqlite3
import csv
import os
from datetime import datetime, timedelta

DB_PATH = os.getenv("TELEMETRY_DB_PATH", "rag_telemetry.db")
OUTPUT_DIR = "powerbi_export"

LOW_CONFIDENCE_THRESHOLD = 0.45


def export_fact_table(conn, out_dir):
    rows = conn.execute("""
        SELECT query_id, timestamp, user_query, retrieval_latency_ms,
               generation_latency_ms, top_similarity_score, retrieved_chunks_count,
               prompt_tokens, completion_tokens, estimated_cost_usd, user_feedback_rating
        FROM rag_telemetry_logs
        ORDER BY timestamp
    """).fetchall()

    path = os.path.join(out_dir, "fact_telemetry.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "date", "timestamp", "user_query", "retrieval_latency_ms",
            "generation_latency_ms", "top_similarity_score", "confidence_category",
            "retrieved_chunks_count", "prompt_tokens", "completion_tokens",
            "estimated_cost_usd", "user_feedback_rating"
        ])
        for row in rows:
            (query_id, timestamp, user_query, retrieval_ms, generation_ms,
             similarity, chunks, prompt_tok, completion_tok, cost, feedback) = row

            date_part = timestamp[:10]
            category = "Low Confidence" if similarity < LOW_CONFIDENCE_THRESHOLD else "High Confidence"

            writer.writerow([
                query_id, date_part, timestamp, user_query, retrieval_ms,
                generation_ms, similarity, category, chunks, prompt_tok,
                completion_tok, cost, feedback
            ])

    return path, len(rows)


def export_dim_date(conn, out_dir):
    dates = conn.execute("""
        SELECT DISTINCT DATE(timestamp) AS d FROM rag_telemetry_logs ORDER BY d
    """).fetchall()

    path = os.path.join(out_dir, "dim_date.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "date", "year", "quarter", "month", "month_name",
            "day", "day_of_week", "day_name", "is_weekend"
        ])
        for (date_str,) in dates:
            d = datetime.strptime(date_str, "%Y-%m-%d")
            writer.writerow([
                date_str, d.year, (d.month - 1) // 3 + 1, d.month, d.strftime("%B"),
                d.day, d.isoweekday(), d.strftime("%A"), d.isoweekday() >= 6
            ])

    return path, len(dates)


def export_dim_query_status(out_dir):
    path = os.path.join(out_dir, "dim_query_status.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["confidence_category", "description", "sort_order"])
        writer.writerow([
            "Low Confidence",
            f"Top similarity score below {LOW_CONFIDENCE_THRESHOLD} — flags a likely knowledge gap",
            1
        ])
        writer.writerow([
            "High Confidence",
            f"Top similarity score at or above {LOW_CONFIDENCE_THRESHOLD}",
            2
        ])
    return path


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        fact_path, fact_count = export_fact_table(conn, OUTPUT_DIR)
        date_path, date_count = export_dim_date(conn, OUTPUT_DIR)
        status_path = export_dim_query_status(OUTPUT_DIR)
    finally:
        conn.close()

    print(f"Exported {fact_count} query rows -> {fact_path}")
    print(f"Exported {date_count} date rows -> {date_path}")
    print(f"Exported dim_query_status -> {status_path}")
    print(f"\nOpen Power BI Desktop -> Get Data -> Text/CSV -> load all three files from ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()