"""
metrics_summary.py — reads logs/app.jsonl and computes aggregate statistics.

This is the bridge from LOGGING (individual events) to MONITORING (trends
across many events, answering "is the system healthy right now").
"""

import json
from collections import Counter
from pathlib import Path

LOG_FILE_PATH = Path("logs/app.jsonl")


def load_log_events(path: Path) -> list[dict]:
    if not path.exists():
        print(f"No log file found at {path}. Run the app first to generate logs.")
        return []

    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def summarize(events: list[dict]) -> None:
    if not events:
        return

    successes = [e for e in events if e.get("event") == "pipeline_answered_successfully"]
    no_context = [e for e in events if e.get("event") == "no_candidates_found"]
    below_threshold = [e for e in events if e.get("event") == "reranking_below_relevance_threshold"]
    llm_failures = [e for e in events if e.get("event") == "llm_generation_failed"]

    total_answered = len(successes) + len(no_context) + len(below_threshold) + len(llm_failures)

    print(f"\n{'='*60}")
    print("METRICS SUMMARY")
    print(f"{'='*60}\n")

    if total_answered == 0:
        print("No question-answering activity found in the logs yet.")
        return

    success_rate = len(successes) / total_answered
    print(f"Total questions processed: {total_answered}")
    print(f"Successful answers:        {len(successes)} ({success_rate:.0%})")
    print(f"No relevant context found: {len(no_context) + len(below_threshold)}")
    print(f"LLM failures:              {len(llm_failures)}")

    source_events = [e for e in events if e.get("event") in ("file_loaded", "blob_loaded", "object_loaded")]
    if source_events:
        print(f"\nDocuments loaded this run: {len(source_events)}")

    rerank_events = [e for e in events if e.get("event") == "reranking_completed" and e.get("top_score") is not None]
    if rerank_events:
        scores = [e["top_score"] for e in rerank_events]
        avg_score = sum(scores) / len(scores)
        print(f"\nReranking calls: {len(rerank_events)}")
        print(f"Average top relevance score: {avg_score:.2f}")
        print(f"Best score seen:  {max(scores):.2f}")
        print(f"Worst score seen: {min(scores):.2f}")

    skip_events = Counter(
        e["event"] for e in events if e.get("event", "").endswith("_skipped_not_configured")
    )
    if skip_events:
        print(f"\nOptional sources skipped (not configured):")
        for event_name, count in skip_events.items():
            print(f"  - {event_name}: {count} time(s)")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    events = load_log_events(LOG_FILE_PATH)
    summarize(events)