import json
from pathlib import Path
from statistics import mean, median

# Adjust if needed
CHUNKS_PATH = Path("../data/processed/chunks.json").resolve()


def percentile(values, p):
    if not values:
        return 0
    values = sorted(values)
    index = int((len(values) - 1) * p / 100)
    return values[index]


def main():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("=" * 70)
    print("RAG CHUNK ANALYSIS")
    print("=" * 70)

    total_chunks = len(chunks)

    token_counts = [c["token_count"] for c in chunks]

    lesson_counts = {}

    for chunk in chunks:
        lesson = chunk["lesson_id"]
        lesson_counts.setdefault(lesson, 0)
        lesson_counts[lesson] += 1

    print(f"Total chunks            : {total_chunks}")
    print(f"Total lessons           : {len(lesson_counts)}")
    print()

    print("Token statistics")
    print("-" * 70)
    print(f"Average                 : {mean(token_counts):.2f}")
    print(f"Median                  : {median(token_counts):.2f}")
    print(f"Minimum                 : {min(token_counts)}")
    print(f"Maximum                 : {max(token_counts)}")
    print(f"5th percentile          : {percentile(token_counts,5)}")
    print(f"95th percentile         : {percentile(token_counts,95)}")

    print()

    smallest = min(lesson_counts.items(), key=lambda x: x[1])
    largest = max(lesson_counts.items(), key=lambda x: x[1])

    print("Chunks per lesson")
    print("-" * 70)
    print(f"Average                 : {total_chunks / len(lesson_counts):.2f}")
    print(f"Smallest lesson         : {smallest[0]} ({smallest[1]} chunks)")
    print(f"Largest lesson          : {largest[0]} ({largest[1]} chunks)")

    print()

    reasons = {}

    for chunk in chunks:
        reason = chunk.get("boundary_reason", "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1

    print("Boundary reasons")
    print("-" * 70)
    for reason, count in sorted(reasons.items()):
        print(f"{reason:<20} {count}")

    print()

    print("Very small chunks (<100 tokens)")
    print("-" * 70)

    small_chunks = [c for c in chunks if c["token_count"] < 100]

    print(f"Count: {len(small_chunks)}")

    for chunk in small_chunks[:15]:
        print(
            f"Lesson {chunk['lesson_id']:>3} | "
            f"Chunk {chunk['chunk_index']:>2} | "
            f"{chunk['token_count']:>3} tokens"
        )

    print()

    print("Very large chunks (>450 tokens)")
    print("-" * 70)

    large_chunks = [c for c in chunks if c["token_count"] > 450]

    print(f"Count: {len(large_chunks)}")

    for chunk in large_chunks[:15]:
        print(
            f"Lesson {chunk['lesson_id']:>3} | "
            f"Chunk {chunk['chunk_index']:>2} | "
            f"{chunk['token_count']:>3} tokens"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()