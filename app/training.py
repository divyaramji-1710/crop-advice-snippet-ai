"""Train the GPT and save reproducible evaluation results."""
import json
from pathlib import Path

import torch

from app.model import GPT
from app.tokenizer import BytePairTokenizer


def train_model(
    corpus="data/corpus.txt",
    output_dir="artifacts",
    steps=400,
    eval_interval=100,
    eval_batches=10,
    batch_size=16,
    block_size=32,
    n_embd=64,
    n_heads=4,
    n_layers=3,
    num_merges=64,
    learning_rate=0.003,
    seed=1337,
):
    if min(steps, eval_interval, eval_batches, batch_size) < 1:
        raise ValueError("Training counts must be positive")

    torch.manual_seed(seed)
    text = Path(corpus).read_text(encoding="utf-8")

    # Hold out the final 10% before learning the tokenizer.
    cut = int(0.9 * len(text))
    tokenizer = BytePairTokenizer().fit(text[:cut], num_merges)
    train_data = torch.tensor(
        tokenizer.encode(text[:cut]), dtype=torch.long
    )
    val_data = torch.tensor(
        tokenizer.encode(text[cut:]), dtype=torch.long
    )

    if min(len(train_data), len(val_data)) <= block_size:
        raise ValueError("Corpus splits are too short for block_size")

    config = {
        "vocab_size": tokenizer.vocab_size,
        "block_size": block_size,
        "n_embd": n_embd,
        "n_heads": n_heads,
        "n_layers": n_layers,
    }
    model = GPT(**config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )

    # Evaluation sampling does not change the training sampler's state.
    train_rng = torch.Generator().manual_seed(seed)
    eval_rng = torch.Generator().manual_seed(seed + 1)

    def get_batch(source, generator):
        positions = torch.randint(
            len(source) - block_size,
            (batch_size,),
            generator=generator,
        )
        x = torch.stack([
            source[i:i + block_size] for i in positions
        ])
        y = torch.stack([
            source[i + 1:i + block_size + 1] for i in positions
        ])
        return x, y

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = {}
        for name, source in [
            ("train", train_data), ("val", val_data)
        ]:
            values = []
            for _ in range(eval_batches):
                _, loss = model(*get_batch(source, eval_rng))
                values.append(loss.item())
            losses[name] = sum(values) / len(values)
        model.train()
        return losses

    history = []

    def record(step):
        losses = evaluate()
        history.append({"step": step, **losses})
        print(
            f"step {step:4d}: train {losses['train']:.3f} "
            f"val {losses['val']:.3f}",
            flush=True,
        )

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
        flush=True,
    )
    record(0)

    for step in range(1, steps + 1):
        _, loss = model(*get_batch(train_data, train_rng))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == steps:
            record(step)

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    # Store weights, architecture, and tokenizer together.
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "merges": [
                [list(pair), token_id]
                for pair, token_id in tokenizer.merges
            ],
        },
        destination / "gpt.pt",
    )

    report = {
        "seed": seed,
        "steps": steps,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "train_tokens": len(train_data),
        "val_tokens": len(val_data),
        "config": config,
        "history": history,
    }
    (destination / "training.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(f"Saved checkpoint and results in {destination}", flush=True)
    return report