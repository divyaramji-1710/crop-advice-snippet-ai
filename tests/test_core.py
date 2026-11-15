"""Tests for each project milestone."""
import pytest
import torch

from app.core import milestone_1, milestone_2, milestone_3


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


def test_milestone_3():
    torch.manual_seed(1337)
    block = milestone_3(
        n_embd=16, n_heads=4, block_size=8
    )
    block.eval()

    x = torch.randn(2, 8, 16)
    result = block(x)

    assert result.shape == x.shape
    assert torch.isfinite(result).all()

    # Changing future inputs must not affect earlier outputs.
    changed = x.clone()
    changed[:, 4:, :] = torch.randn_like(changed[:, 4:, :])
    changed_result = block(changed)

    torch.testing.assert_close(
        result[:, :4, :],
        changed_result[:, :4, :],
    )

    # Verify gradients reach every trainable parameter.
    result.square().mean().backward()
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()

    with pytest.raises(ValueError):
        milestone_3(n_embd=15, n_heads=4, block_size=8)


def test_milestone_4(tmp_path):
    from app.core import milestone_4
    from app.model import GPT

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Tomato roots need space. Keep soil evenly moist.\n" * 30,
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"

    report = milestone_4(
        corpus=corpus,
        output_dir=output,
        steps=2,
        eval_interval=2,
        eval_batches=1,
        batch_size=2,
        block_size=8,
        n_embd=16,
        n_heads=2,
        n_layers=1,
        num_merges=4,
    )

    assert [row["step"] for row in report["history"]] == [0, 2]
    assert all(
        torch.isfinite(torch.tensor(row[key]))
        for row in report["history"]
        for key in ("train", "val")
    )
    assert (output / "training.json").exists()

    checkpoint = torch.load(
        output / "gpt.pt",
        map_location="cpu",
        weights_only=True,
    )
    restored = GPT(**checkpoint["config"])
    restored.load_state_dict(checkpoint["model_state"])

    inputs = torch.zeros((1, 4), dtype=torch.long)
    logits, _ = restored(inputs)
    assert logits.shape == (
        1, 4, checkpoint["config"]["vocab_size"]
    )


def test_milestone_5(tmp_path):
    from app.core import milestone_4, milestone_5
    from app.generation import SnippetGenerator

    corpus = tmp_path / "corpus.txt"
    corpus.write_text(
        "Tomato roots need space. Keep soil evenly moist.\n" * 30,
        encoding="utf-8",
    )
    output = tmp_path / "artifacts"

    milestone_4(
        corpus=corpus,
        output_dir=output,
        steps=1,
        eval_interval=1,
        eval_batches=1,
        batch_size=2,
        block_size=8,
        n_embd=16,
        n_heads=2,
        n_layers=1,
        num_merges=4,
    )

    checkpoint = output / "gpt.pt"
    options = {
        "checkpoint": checkpoint,
        "max_new_tokens": 5,
        "seed": 3,
    }

    first = milestone_5("Tomato", **options)
    second = milestone_5("Tomato", **options)
    assert isinstance(first, str)
    assert first == second

    # Top-k=1 chooses the highest-scoring token.
    generator = SnippetGenerator(checkpoint)
    prompt = torch.tensor(
        [generator.tokenizer.encode("Tomato")],
        dtype=torch.long,
    )
    with torch.no_grad():
        logits, _ = generator.model(
            prompt[:, -generator.model.block_size:]
        )
    expected_id = logits[0, -1].argmax().item()

    result = generator.generate(
        "Tomato", max_new_tokens=1, top_k=1
    )
    assert result == generator.tokenizer.decode([expected_id])

    with pytest.raises(ValueError):
        generator.generate("")

    with pytest.raises(ValueError):
        generator.generate("Tomato", temperature=0)


@pytest.mark.skip(reason="start milestone 6")
def test_milestone_6():
    raise NotImplementedError
def test_gpt_forward_and_causality():
    from app.model import GPT

    torch.manual_seed(1337)
    model = GPT(
        vocab_size=32,
        block_size=8,
        n_embd=16,
        n_heads=4,
        n_layers=2,
    )

    inputs = torch.randint(32, (2, 8))
    targets = torch.randint(32, (2, 8))
    logits, loss = model(inputs, targets)

    assert logits.shape == (2, 8, 32)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    # Future tokens must not affect earlier predictions.
    changed = inputs.clone()
    changed[:, 4:] = (changed[:, 4:] + 1) % 32
    changed_logits, _ = model(changed)

    torch.testing.assert_close(
        logits[:, :4],
        changed_logits[:, :4],
    )

    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()