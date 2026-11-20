# Creative Writing Prompter

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/dwhr2kumwu/badge.svg)](https://ask.thili.ai/b/dwhr2kumwu)

This CLI tool helps aspiring writers overcome writer's block by offering context-aware next-word suggestions. It uses a next-word prediction engine, built from scratch, to analyze a corpus of short stories and provide relevant word choices based on n-gram probabilities.

## What you are building

**Goal.** You will build a command-line interface (CLI) tool that provides context-aware next-word suggestions for aspiring writers, leveraging a next-word prediction engine to sample from a genre corpus.

**Done when.** The CLI tool successfully takes user input, processes it using n-gram probabilities with add-k smoothing, and suggests a next word by sampling from the model.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Prepare a text corpus of short stories.**
      Hint: Text preprocessing, tokenization.
- [x] **2. Implement n-gram counting and maximum-likelihood probabilities.**
      Hint: N-gram generation, frequency counting.
- [x] **3. Add add-k smoothing to handle unseen n-grams.**
      Hint: Add-k smoothing formula.
- [ ] **4. Develop a function to generate text by sampling from the model.**  
      Hint: Probability distribution sampling.
- [ ] **5. Create a CLI interface for users to input text and get suggestions.**  
      Hint: Command-line arguments, user input.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

The bundled sample corpus is in `app/corpus.txt`. The CLI uses it by default, and model commands can also load a
custom UTF-8 corpus file. Text preprocessing lowercases and normalizes words while preserving internal apostrophes.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Allow users to specify the desired length for generated text sequences.
- Implement a feature to display the top N most probable next words instead of just one.
- Enable users to configure the smoothing parameter 'k' through the CLI.
