"""Train the GPT and preserve latest and best checkpoints."""
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
    resume=False,
):
    if min(steps, eval_interval, eval_batches, batch_size) < 1:
        raise ValueError("Training counts must be positive")

    torch.manual_seed(seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "gpt.pt"
    report_path = destination / "training.json"
    best_path = destination / "best.pt"

    text = Path(corpus).read_text(encoding="utf-8")
    cut = int(0.9 * len(text))
    history = []
    start_step = 0
    best_val = float("inf")
    best_step = None
    tokenizer = BytePairTokenizer()

    if resume:
        saved = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
        config = saved["config"]
        block_size = config["block_size"]

        for pair, token_id in saved["merges"]:
            pair = tuple(pair)
            tokenizer.merges.append((pair, token_id))
            tokenizer.vocab[token_id] = (
                tokenizer.vocab[pair[0]]
                + tokenizer.vocab[pair[1]]
            )

        model = GPT(**config)
        model.load_state_dict(saved["model_state"])
        start_step = saved.get("step", 0)

        if report_path.exists():
            previous = json.loads(
                report_path.read_text(encoding="utf-8")
            )
            history = previous["history"]
            start_step = previous["steps"]

        if best_path.exists():
            best_saved = torch.load(
                best_path,
                map_location="cpu",
                weights_only=True,
            )
            best_val = best_saved.get("val_loss", float("inf"))
            best_step = best_saved.get("step")
    else:
        tokenizer.fit(text[:cut], num_merges)
        config = {
            "vocab_size": tokenizer.vocab_size,
            "block_size": block_size,
            "n_embd": n_embd,
            "n_heads": n_heads,
            "n_layers": n_layers,
        }
        model = GPT(**config)

    train_data = torch.tensor(
        tokenizer.encode(text[:cut]), dtype=torch.long
    )
    val_data = torch.tensor(
        tokenizer.encode(text[cut:]), dtype=torch.long
    )
    if min(len(train_data), len(val_data)) <= block_size:
        raise ValueError("Corpus splits are too short")

    # Resuming continues weights with a fresh optimizer.
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate
    )
    train_rng = torch.Generator().manual_seed(seed + start_step)
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
            source[i + 1:i + block_size + 1]
            for i in positions
        ])
        return x, y

    @torch.no_grad()
    def evaluate():
        model.eval()
        losses = {}

        for name, source in [
            ("train", train_data),
            ("val", val_data),
        ]:
            values = []
            for _ in range(eval_batches):
                _, loss = model(*get_batch(source, eval_rng))
                values.append(loss.item())
            losses[name] = sum(values) / len(values)

        model.train()
        return losses

    def checkpoint(step, val_loss):
        return {
            "model_state": model.state_dict(),
            "config": config,
            "step": step,
            "val_loss": val_loss,
            "merges": [
                [list(pair), token_id]
                for pair, token_id in tokenizer.merges
            ],
        }

    def save_report(step):
        report = {
            "seed": seed,
            "steps": step,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "train_tokens": len(train_data),
            "val_tokens": len(val_data),
            "config": config,
            "best_step": best_step,
            "best_val": best_val,
            "history": history,
        }
        report_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )
        return report

    def record(step):
        nonlocal best_val, best_step

        losses = evaluate()
        history.append({"step": step, **losses})
        print(
            f"step {step:4d}: train {losses['train']:.3f} "
            f"val {losses['val']:.3f}",
            flush=True,
        )

        state = checkpoint(step, losses["val"])
        torch.save(state, checkpoint_path)

        if losses["val"] < best_val:
            best_val = losses["val"]
            best_step = step
            torch.save(state, best_path)
            print(f"Saved best.pt at step {step}", flush=True)

        save_report(step)

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
        flush=True,
    )
    record(start_step)

    for additional_step in range(1, steps + 1):
        _, loss = model(*get_batch(train_data, train_rng))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        step = start_step + additional_step
        if (
            additional_step % eval_interval == 0
            or additional_step == steps
        ):
            record(step)

    print(
        f"Saved checkpoint and results in {destination}",
        flush=True,
    )
    return save_report(start_step + steps)