# Crop Advice Snippet AI

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gru3s8ae9t/badge.svg)](https://ask.thili.ai/b/gru3s8ae9t)

An educational tomato-text generator built with a GPT trained from
scratch in PyTorch. It implements byte-pair encoding, causal attention,
transformer blocks, training, and command-line generation.

**Current limitation:** generated text remains grammatically unreliable.
The project does not yet achieve its goal of coherent agricultural advice.
Do not use its output for farming decisions.

## Implementation status

- [x] Collect and clean a tomato-growing corpus.
- [x] Implement byte-level BPE and learned token/position embeddings.
- [x] Implement causal self-attention and transformer blocks.
- [x] Train a GPT and record training and validation losses.
- [x] Implement generation with temperature and top-k sampling.
- [x] Implement a CLI accepting a crop or problem prompt.

These checkboxes describe implemented components. Coherent, relevant,
factually reliable advice remains an unmet output-quality goal.

## Setup

Requires Python 3.12. Run from the project root.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Linux or macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The direct dependencies are pinned in requirements.txt.
No API key, external language-model service, or GPU is required.

## Generate text

Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m app.cli generate "Tomato plants need"
```

Interactive prompt:

```powershell
.\.venv\Scripts\python.exe -m app.cli generate
```

Change sampling settings:

```powershell
.\.venv\Scripts\python.exe -m app.cli generate "The soil should be" --temperature 0.6 --top-k 10 --max-new-tokens 60 --seed 0
```

On Linux or macOS, use `python` instead of
`.\.venv\Scripts\python.exe`.

The default checkpoint is `artifacts/best-model/best.pt`.
Use `--checkpoint` to select another compatible checkpoint.

## Example output

Command:

```powershell
.\.venv\Scripts\python.exe -m app.cli generate "Tomato plants need"
```

Observed continuation:

```text
s are taken they makes nearly matures, and soil was been prevented flavor by sold for gath
```

The CLI successfully loads the model and samples tokens. This example
also demonstrates its current grammatical and semantic limitations.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest
```

The suite contains 10 tests covering corpus preparation, BPE round trips,
embeddings, causal behavior, gradient flow, checkpoint loading,
sampling, CLI argument handling, and smoke checks.

Passing tests verifies software behavior, not the correctness or
usefulness of generated agricultural advice.

## Training data

Source: W. W. Tracy, *Tomato Culture: A Practical Treatise on the Tomato*.

https://www.gutenberg.org/ebooks/28011

Original download: `data/raw/tomato-culture.txt`

Cleaned corpus: `data/corpus.txt`

Cleaning removes front matter, the index, transcriber's notes,
Gutenberg footer, illustration/page markers, and emphasis markers.
Whitespace and apostrophes are normalized while case and punctuation
are preserved.

The final cleaned corpus contains 178,501 characters.
This historical text is not a source of verified modern recommendations.

The separate `data/examples.jsonl` contains 12 AI-assisted,
source-attributed prompt-response drafts. Their sources are recorded in
`data/SOURCES.md`. These drafts were not used to train the submitted model.

## Architecture

- Byte-level BPE, starting with 256 byte tokens and learning up to 64 merges.
- Learned token and position embeddings.
- Scaled dot-product attention implemented directly in PyTorch.
- A causal mask preventing attention to future positions.
- Multiple attention heads, feed-forward layers, residual connections,
  and pre-layer normalization.
- Next-token cross-entropy training with AdamW.
- Autoregressive generation with temperature and top-k filtering.

No pretrained model weights are used.

## Experiments

### Initial tokenizer experiment

Before the index was removed, the corpus contained 183,328 bytes.
Learning 64 merges reduced it to 111,486 tokens with a vocabulary
of 320 tokens. Decoding reproduced the input exactly.

These counts describe the initial tokenizer experiment, not the final
clean-corpus training split.

### Initial small model

The first model had 192,832 parameters. After 400 steps:

| Step | Training loss | Validation loss |
| --- | --- | --- |
| 0 | 5.940 | 5.940 |
| 100 | 3.543 | 3.838 |
| 200 | 3.283 | 3.611 |
| 300 | 3.099 | 3.425 |
| 400 | 2.932 | 3.306 |

The original corpus still included an index in this experiment.
Its validation numbers are not directly comparable with later runs
using the revised corpus.

### Clean-corpus small model

A fresh 192,832-parameter model trained for 3,000 steps reached
training loss 2.074 and validation loss 2.741.
The lowest sampled validation loss was 2.705 at step 2400.
Generation remained incoherent.

### Larger model

The larger model has 684,352 parameters, 128-dimensional embeddings,
four attention heads, three blocks, and a 64-token context window.

| Step | Training loss | Validation loss |
| --- | --- | --- |
| 0 | 5.931 | 5.918 |
| 300 | 3.069 | 3.302 |
| 600 | 2.511 | 2.927 |
| 900 | 2.188 | 2.767 |
| 1200 | 1.984 | 2.653 |
| 1500 | 1.800 | 2.640 |
| 1800 | 1.746 | 2.713 |
| 2100 | 1.591 | 2.783 |
| 2400 | 1.512 | 2.836 |
| 2700 | 1.411 | 2.887 |
| 3000 | 1.275 | 3.021 |

Training loss continued falling while validation loss worsened after
step 1500, indicating overfitting.

A subsequent fresh run used best-checkpoint saving and stopped at
1500 steps. Its checkpoint and recorded metrics are in
`artifacts/best-model/`.

Validation loss is estimated from sampled batches rather than every
token in the validation split. Small fluctuations should not be
overinterpreted.

## Reproduce training

The final 10% of the corpus is held out before fitting BPE.
The tokenizer is learned only from the training portion.

Regenerate the cleaned corpus:

```powershell
.\.venv\Scripts\python.exe -c "from app.core import milestone_1; milestone_1()"
```

Train the larger configuration into a new output folder:

```powershell
.\.venv\Scripts\python.exe -u -c "from app.core import milestone_4; milestone_4(output_dir='artifacts/reproduction', steps=1500, eval_interval=300, learning_rate=0.001, block_size=64, n_embd=128, n_heads=4, n_layers=3)"
```

Training writes:

- `gpt.pt`: latest evaluated checkpoint.
- `best.pt`: checkpoint with the lowest sampled validation loss.
- `training.json`: configuration and loss history.

Checkpoints contain model weights, architecture settings, and BPE merges.

Resuming continues model weights with a fresh optimizer; it does not
restore the original optimizer state. Exact numerical reproducibility
can vary across platforms and PyTorch builds.

## What the experiments showed

Lower next-token loss did not guarantee coherent advice.
Increasing model capacity improved some validation measurements but
eventually increased overfitting. Lower temperature and top-k filtering
did not resolve the underlying language-quality problem.

The historical book also differs from the desired short prompt-response
format. The focused example file is an initial dataset-design experiment,
not evidence that this limitation has been solved.

## AI assistance

An AI coding assistant provided substantial help with implementation,
tests, debugging, experiment planning, and documentation.
Training and the example outputs were run locally.