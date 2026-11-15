"""Project implementation, built one milestone at a time."""
import re
from pathlib import Path
import torch
import torch.nn as nn

from app.tokenizer import BytePairTokenizer
from app.model import Block


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

        # Exclude the back-of-book index and transcriber's notes.
    back_matter = re.search(
        r"^[ \t]*(?:INDEX|Transcriber's Notes)[ \t]*\r?$",
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )
    if back_matter:
        text = text[:back_matter.start()]
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


def milestone_2(
    text, num_merges=64, n_embd=64, block_size=32,
):
    tokenizer = BytePairTokenizer().fit(text, num_merges)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    token_embedding = nn.Embedding(tokenizer.vocab_size, n_embd)
    position_embedding = nn.Embedding(block_size, n_embd)

    return tokenizer, data, token_embedding, position_embedding


def milestone_3(n_embd=64, n_heads=4, block_size=32):
    return Block(n_embd, n_heads, block_size)


def milestone_4(**kwargs):
    from app.training import train_model
    return train_model(**kwargs)


def milestone_5(
    prompt,
    checkpoint="artifacts/gpt.pt",
    max_new_tokens=60,
    temperature=0.8,
    top_k=20,
    seed=0,
):
    from app.generation import SnippetGenerator

    generator = SnippetGenerator(checkpoint)
    return generator.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
    )


def milestone_6():
    raise NotImplementedError("TODO(milestone 6)")