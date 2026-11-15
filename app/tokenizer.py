"""A small byte-pair tokenizer implemented from scratch."""
from collections import Counter


def merge_pair(ids, pair, new_id):
    result = []
    i = 0
    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            result.append(new_id)
            i += 2
        else:
            result.append(ids[i])
            i += 1
    return result


class BytePairTokenizer:
    def __init__(self):
        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def fit(self, text, num_merges=64):
        if not text:
            raise ValueError("Training text must not be empty")
        if num_merges < 0:
            raise ValueError("num_merges must be non-negative")

        self.merges = []
        self.vocab = {i: bytes([i]) for i in range(256)}
        ids = list(text.encode("utf-8"))

        for _ in range(num_merges):
            counts = Counter(zip(ids, ids[1:]))
            if not counts:
                break

            pair, frequency = counts.most_common(1)[0]
            if frequency < 2:
                break

            new_id = len(self.vocab)
            self.vocab[new_id] = (
                self.vocab[pair[0]] + self.vocab[pair[1]]
            )
            self.merges.append((pair, new_id))
            ids = merge_pair(ids, pair, new_id)

        return self

    def encode(self, text):
        ids = list(text.encode("utf-8"))
        for pair, new_id in self.merges:
            ids = merge_pair(ids, pair, new_id)
        return ids

    def decode(self, ids):
        raw = b"".join(self.vocab[int(i)] for i in ids)
        return raw.decode("utf-8", errors="replace")