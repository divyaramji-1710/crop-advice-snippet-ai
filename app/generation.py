"""Load a trained GPT and sample a continuation."""
import torch
from torch.nn import functional as F

from app.model import GPT
from app.tokenizer import BytePairTokenizer


class SnippetGenerator:
    def __init__(self, checkpoint="artifacts/gpt.pt"):
        saved = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )

        self.tokenizer = BytePairTokenizer()
        for pair, token_id in saved["merges"]:
            pair = tuple(pair)
            self.tokenizer.merges.append((pair, token_id))
            self.tokenizer.vocab[token_id] = (
                self.tokenizer.vocab[pair[0]]
                + self.tokenizer.vocab[pair[1]]
            )

        self.model = GPT(**saved["config"])
        self.model.load_state_dict(saved["model_state"])
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=60,
        temperature=0.8,
        top_k=20,
        seed=0,
    ):
        if not prompt.strip():
            raise ValueError("Prompt must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 1 <= top_k <= self.tokenizer.vocab_size:
            raise ValueError("top_k is outside the vocabulary")

        rng = torch.Generator().manual_seed(seed)
        idx = torch.tensor(
            [self.tokenizer.encode(prompt)],
            dtype=torch.long,
        )
        prompt_length = idx.shape[1]

        for _ in range(max_new_tokens):
            logits, _ = self.model(
                idx[:, -self.model.block_size:]
            )
            logits = logits[:, -1, :] / temperature

            threshold = torch.topk(logits, top_k)[0][:, [-1]]
            logits = logits.masked_fill(
                logits < threshold,
                float("-inf"),
            )
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(
                probabilities, 1, generator=rng
            )
            idx = torch.cat([idx, next_token], dim=1)

        # Return the generated continuation, excluding the prompt.
        return self.tokenizer.decode(
            idx[0, prompt_length:].tolist()
        )