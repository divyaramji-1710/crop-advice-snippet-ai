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
- [ ] **4. Train the GPT model using the agricultural text data.**  
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