# Calibration Guide

TriAttention uses pre-computed statistics (Q/K centers and norms) for each model. Pre-computed stats for supported models are included in `triattention/calibration/`.

## Generating Stats for a Custom Model

```bash
python scripts/calibrate.py \
    --model <your-model-id-or-path> \
    --input <calibration_text.txt> \
    --output triattention/calibration/model_stats.pt
```

### What is `calibration_text.txt`?

Any plain-text file containing natural, coherent text. The script tokenizes it and runs a single forward pass to collect query-state frequency statistics.

- **Domain-agnostic**: calibration and inference do not need to be in the same domain — e.g., calibrating on code and running math reasoning works fine.
- **Robust**: we have tested across various text types (code, math, general prose) with consistent results.
- **Longer is better**: provide enough text to approach `--max-length` (default 32768 tokens) for best coverage.
- **Avoid degenerate inputs**: do not use garbled/corrupted text or long repetitive loops (e.g., the same sentence copied thousands of times).

A Wikipedia article, a book chapter, or a source code file are all good choices.

The calibration script runs a forward pass on plain text input, captures query states from every attention layer, inverts RoPE, and computes per-head frequency statistics. The resulting `.pt` file is loaded at inference time to score keys via the trigonometric series.

The script also runs a separate direct calibration pass via `main2()`,
which captures raw `q_proj` outputs before Q normalization and RoPE. For
`--output stats.pt`, the original method saves `stats.pt` and the direct method
saves `stats.direct.pt`. Afterward, it compares all heads and prints maximum
absolute error, relative L2 error, and `allclose` for each statistic
(rtol=0.01, atol=0.0001). The original calibration code is unchanged.
For CPU runs, add `--device cpu --attn-implementation eager`.

## Pre-computed Stats

Stats are organised by experiment target. Each sub-directory contains per-model `.pt` files:

**For AIME-24 experiments** (`triattention/calibration/for_aime24_experiment/`):

| Model | Stats Path |
|-------|-----------|
| Qwen3-8B | `triattention/calibration/for_aime24_experiment/qwen3_8b.pt` |
| DeepSeek-R1-Distill-Llama-8B | `triattention/calibration/for_aime24_experiment/ds_llama8b.pt` |
| DeepSeek-R1-Distill-Qwen-7B | `triattention/calibration/for_aime24_experiment/ds_qwen7b.pt` |

**For AIME-25 experiments** (`triattention/calibration/for_aime25_experiment/`):

| Model | Stats Path |
|-------|-----------|
| Qwen3-8B | `triattention/calibration/for_aime25_experiment/qwen3_8b.pt` |
| DeepSeek-R1-Distill-Llama-8B | `triattention/calibration/for_aime25_experiment/ds_llama8b.pt` |
| DeepSeek-R1-Distill-Qwen-7B | `triattention/calibration/for_aime25_experiment/ds_qwen7b.pt` |
