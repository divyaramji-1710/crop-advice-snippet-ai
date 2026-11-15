"""Project implementation, built one milestone at a time."""
import re
from pathlib import Path


def milestone_1(
    source="data/raw/tomato-culture.txt",
    output="data/corpus.txt",
):
    text = Path(source).read_text(encoding="utf-8-sig")

    # Remove the Gutenberg footer.
    end = re.search(
        r"^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*$",
        text,
        flags=re.MULTILINE,
    )
    if end:
        text = text[:end.start()]

    # Skip the title, preface, and table of contents.
    chapters = list(re.finditer(
        r"^[ \t]*CHAPTER I[ \t]*\r?$",
        text,
        flags=re.MULTILINE,
    ))
    if not chapters:
        raise ValueError("Could not find the first chapter")
    text = text[chapters[-1].start():]

    # Normalize typography and remove illustration/page markers.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = re.sub(
        r"\[Illustration:.*?\]", "", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"\[Pg[^\]]*\]", "", text)

    # Remove Gutenberg's plain-text emphasis markers.
    text = text.replace("=", "").replace("_", "")
    
    # Join wrapped lines while preserving paragraph boundaries.
    paragraphs = re.split(r"\n[ \t]*\n+", text)
    paragraphs = [
        re.sub(r"\s+", " ", paragraph).strip()
        for paragraph in paragraphs
    ]
    cleaned = "\n\n".join(p for p in paragraphs if p) + "\n"

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(cleaned, encoding="utf-8")
    return cleaned


def milestone_2():
    raise NotImplementedError("TODO(milestone 2)")


def milestone_3():
    raise NotImplementedError("TODO(milestone 3)")


def milestone_4():
    raise NotImplementedError("TODO(milestone 4)")


def milestone_5():
    raise NotImplementedError("TODO(milestone 5)")


def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")