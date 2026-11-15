"""Print candidate cultivation passages for manual review."""
import re
from pathlib import Path

text = Path("data/corpus.txt").read_text(encoding="utf-8")

topics = {
    "watering": r"\bwater(?:ing)?\b",
    "soil": r"\bsoil\b",
    "transplanting": r"\btransplant(?:ing|ed)?\b",
    "spacing": r"\b(?:spacing|distance|apart)\b",
}

paragraphs = [
    re.sub(r"\s+", " ", paragraph).strip()
    for paragraph in text.split("\n\n")
]

for topic, pattern in topics.items():
    print(f"\n=== {topic.upper()} ===")
    selected = 0

    for paragraph in paragraphs:
        words = paragraph.split()

        if not 50 <= len(words) <= 160:
            continue
        if not re.search(pattern, paragraph, flags=re.IGNORECASE):
            continue

        print(paragraph)
        print()
        selected += 1

        if selected == 2:
            break