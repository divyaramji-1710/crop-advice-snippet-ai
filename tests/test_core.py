"""Tests for each project milestone."""
import pytest
import torch

from app.core import milestone_1, milestone_2


def test_milestone_1(tmp_path):
    source = tmp_path / "raw.txt"
    output = tmp_path / "prepared" / "corpus.txt"

    source.write_text(
        "Gutenberg header\n"
        "Contents\n"
        "  CHAPTER I\n"
        "  Botany 1\n\n"
        "CHAPTER I\n\n"
        "Botany of the Tomato\n\n"
        "The tomato’s roots\n"
        "need space.\n\n"
        "[Illustration: A tomato plant.]\n"
        "[Pg 2]\n"
        "Keep paragraphs separate.\n\n"
        "*** END OF THE PROJECT GUTENBERG EBOOK TEST ***\n"
        "License footer",
        encoding="utf-8",
    )

    text = milestone_1(source, output)

    assert text.startswith("CHAPTER I\n\nBotany")
    assert "The tomato's roots need space." in text
    assert "\n\nKeep paragraphs separate." in text
    assert "Contents" not in text
    assert "Gutenberg header" not in text
    assert "License footer" not in text
    assert "[Illustration:" not in text
    assert "[Pg" not in text
    assert output.read_text(encoding="utf-8") == text


def test_milestone_2():
    text = "tomato roots need space. " * 20
    tokenizer, data, tok_emb, pos_emb = milestone_2(
        text, num_merges=8, n_embd=16, block_size=8,
    )

    assert tokenizer.decode(data.tolist()) == text
    assert tokenizer.vocab_size > 256
    assert len(data) < len(text.encode("utf-8"))
    assert data.dtype == torch.long
    assert int(data.max()) < tokenizer.vocab_size

    # The first merge must be the most frequent byte pair.
    from collections import Counter
    raw = list(text.encode("utf-8"))
    expected_pair = Counter(zip(raw, raw[1:])).most_common(1)[0][0]
    assert tokenizer.merges[0][0] == expected_pair

    # Byte tokens also support text absent from training.
    unseen = "New crop: café 🌱"
    assert tokenizer.decode(tokenizer.encode(unseen)) == unseen

    batch = data[:8].unsqueeze(0)
    embedded = tok_emb(batch) + pos_emb(torch.arange(8))
    assert embedded.shape == (1, 8, 16)

    # Both embedding tables receive gradients.
    embedded.sum().backward()
    assert tok_emb.weight.grad is not None
    assert pos_emb.weight.grad is not None


@pytest.mark.skip(reason="start milestone 3")
def test_milestone_3():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 4")
def test_milestone_4():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 5")
def test_milestone_5():
    raise NotImplementedError


@pytest.mark.skip(reason="start milestone 6")
def test_milestone_6():
    raise NotImplementedError