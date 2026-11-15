# Crop Advice Snippet AI

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/gru3s8ae9t/badge.svg)](https://ask.thili.ai/b/gru3s8ae9t)

This project develops an AI-powered CLI tool designed for farmers. It generates concise, context-aware agricultural advice snippets for specific crops, leveraging a custom-trained GPT model on farming best practices. The goal is to provide quick, relevant guidance to aid agricultural decision-making.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that generates concise, context-aware agricultural advice snippets for specific crops, trained on a dataset of farming best practices.

**Done when.** The project is done when you have successfully applied the byte-pair tokenizer and embeddings to prepare raw text, implemented causal self-attention, assembled the transformer block, and trained a GPT model from scratch to generate coherent agricultural advice snippets via a functional CLI.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Collect a small corpus of agricultural advice, crop guides, or farming articles.**  
      Hint: Gathering and cleaning your raw text data.
- [x] **2. Implement the byte-pair tokenizer and generate embeddings from the agricultural text.**  
      Hint: Apply Byte-Pair Encoding and create token embeddings.
- [x] **3. Build the causal self-attention mechanism and assemble the transformer block.**  
      Hint: Construct the Causal Self-Attention layer and stack Transformer Blocks.
- [x] **4. Train the GPT model using the agricultural text data.**  
      Hint: Implement the training loop for your custom GPT model.
- [ ] **5. Implement a text generation function to provide concise advice snippets.**  
      Hint: Develop a sampling strategy for coherent text generation.
- [ ] **6. Create a CLI where users can input a crop or problem and get advice.**  
      Hint: Design a simple command-line interface for user interaction.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Experiment with different model sizes or hyperparameters to improve the relevance and quality of the generated advice.
- Implement a simple mechanism to filter or re-rank generated advice snippets to ensure conciseness and direct relevance.
- Explore methods to make the advice more actionable, perhaps by including specific steps or recommendations.

## Training corpus

Source: W. W. Tracy, *Tomato Culture: A Practical Treatise on the Tomato*.
https://www.gutenberg.org/ebooks/28011

The original download is preserved in data/raw/tomato-culture.txt.
The cleaned training text is saved in data/corpus.txt.

This historical corpus is used for an educational text-generation
experiment. Generated text is not verified modern farming advice.

## Tokenization results

The byte-level BPE tokenizer learns 64 merges from the corpus.
It reduces 183,328 bytes to 111,486 tokens with a vocabulary of
320 tokens. Encoding and decoding reproduce the input exactly.

Token and position embeddings are initialized here and learned
during model training.

## Training results

The GPT has 192,832 parameters and was trained for 400 steps.
The final 10% of the text was held out before fitting the BPE tokenizer.

| Step | Training loss | Validation loss |
| --- | --- | --- |
| 0 | 5.940 | 5.940 |
| 100 | 3.543 | 3.838 |
| 200 | 3.283 | 3.611 |
| 300 | 3.099 | 3.425 |
| 400 | 2.932 | 3.306 |

Both losses decreased. Held-out performance improved, although these
losses alone do not establish coherent or factually correct advice.

The checkpoint is saved in artifacts/gpt.pt.
Training settings and losses are saved in artifacts/training.json.

## Current results and limitations

The project implements byte-level BPE, learned token and position
embeddings, causal self-attention, transformer blocks, GPT training,
checkpoint loading, and a command-line generation interface.

The larger model has 684,352 parameters. Its validation loss reached
2.640 at step 1500, then rose to 3.021 at step 3000 while training loss
continued falling, indicating overfitting.

Example CLI command:

```powershell
.\.venv\Scripts\python.exe -m app.cli generate "Tomato plants need"
```

Observed continuation:

```text
s are taken they makes nearly matures, and soil was been prevented flavor by sold for gath
```

Generation works mechanically, but output remains grammatically
unreliable and does not yet meet the coherent-advice goal.
This is an educational prototype, not verified farming guidance.

The default CLI checkpoint is artifacts/best-model/best.pt.
The 12 examples in data/examples.jsonl are an experimental dataset
draft and were not used to train this checkpoint.

## Data provenance

The trained model uses Tomato Culture: A Practical Treatise on the
Tomato by W. W. Tracy, downloaded from Project Gutenberg:
https://www.gutenberg.org/ebooks/28011

Original file: data/raw/tomato-culture.txt
Cleaned file: data/corpus.txt

Cleaning removes front matter, the index, transcriber's notes,
Gutenberg footer, illustration markers, and emphasis markers.
It normalizes whitespace and apostrophes.

The separate data/examples.jsonl file contains AI-assisted,
source-attributed example drafts. It was not used to train the
submitted checkpoint.