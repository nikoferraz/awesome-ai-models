# Performance Benchmarks

[← Back to Main List](README.md)

All benchmarks cited with source and retrieval date. Data marked as **Measured**, **Reported**, or **Estimated**.

## Standardized Academic Benchmarks

### Text LLM Benchmarks

| Benchmark | Tests | Best Open | Best Closed |
|-----------|-------|-----------|-------------|
| **MMLU** (5-shot) | General knowledge (57 subjects) | DeepSeek-V3: ~86% | GPT-5: ~87% |
| **GPQA Diamond** | Graduate-level STEM | Llama 3.1 405B: ~83% | Grok 4: **87.5%** |
| **HumanEval** | Python code synthesis | DeepSeek-V3: ~70% | GPT-4.5: ~70% |
| **SWE-bench** | Real GitHub issues | Llama 3.1 70B: ~55% | Claude 4 Opus: **72.5%** |
| **GSM8K** | Math word problems | Qwen 2.5: ~92% | DeepSeek-V3: ~93% |
| **MATH** | Competition math | Qwen 2.5 72B: **83.1%** | o1: 83.5% |
| **BFCL** | Function calling | Llama 3.1 405B: **81.1%** | GPT-4: ~80% |

**Key Insights**:
- Open models now competitive on most benchmarks
- Claude dominates **real-world coding** (SWE-bench)
- Llama 3.1 405B leads in **tool use** (BFCL)
- Math remains challenging; Qwen and DeepSeek excel

**Sources**: [Analytics Vidhya LLM Benchmarks](https://www.analyticsvidhya.com/blog/2025/04/what-are-llm-benchmarks/), [Arize 40 Benchmarks](https://arize.com/blog/llm-benchmarks-mmlu-codexglue-gsm8k) (Retrieved: 2025-10-28)

---

## Multimodal Benchmarks

| Benchmark | Tests | Top Models |
|-----------|-------|------------|
| **MMMU** | Multi-discipline multimodal | InternVL3-78B, Gemini 2.5 Pro |
| **MM-Vet** | Multimodal reasoning | Qwen3-VL, GPT-4V |
| **VQAv2** | Visual question answering | LLaVA-NeXT-34B |
| **TextVQA** | Text in images | Qwen3-VL (29 languages) |

---

## Human Preference Leaderboards

### LMArena (Chatbot Arena)

Elo ratings from blind, randomized head-to-head battles.

| Rank | Model | Elo | Provider |
|------|-------|-----|----------|
| 1 | **Gemini 2.5 Pro** | 1315 | Google |
| 2 | GPT-5 | ~1300 | OpenAI |
| 3 | Claude 4 Opus | ~1280 | Anthropic |
| 4 | DeepSeek-V3 | ~1260 | DeepSeek AI |
| 5 | Llama 3.1 405B | ~1250 | Meta |

**Insight**: Gemini 2.5 Pro tops **real-world user preference** despite not leading all academic benchmarks—disconnect between metrics and perceived utility.

---

## Vellum AI Leaderboard

Focuses on non-saturated, practical tests.

| Category | Test | Leader | Score |
|----------|------|--------|-------|
| Reasoning | GPQA Diamond | Grok 4 | 87.5% |
| Coding | SWE-bench | Claude 4 Sonnet | 72.7% |
| Tool Use | BFCL | Llama 3.1 405B | 81.1% |
| Math | AIME 2025 | GPT-5 | 100% |

---

## Domain-Specific Capabilities

### Long-Context Processing

| Model | Context Window | Needle-in-Haystack |
|-------|----------------|-------------------|
| Gemini 2.5 Pro | **1M-2M tokens** | ✅ Strong |
| GPT-4.1 | 1.05M tokens | ✅ Strong |
| Claude 4 | 200K tokens | ✅ Strong |
| Llama 3.1 | 128K tokens | ✅ Good |
| DeepSeek-V3 | 128K tokens | ✅ Good |

**Note**: Theoretical window ≠ effective recall. Test with your data.

---

## Multilingual Support

| Model | Languages | OCR Languages |
|-------|-----------|---------------|
| Llama 4 | 200+ (pre-training) | N/A |
| Qwen 2.5 VL | N/A | 29 |
| Gemini 2.5 | 100+ | N/A |

---

## Performance Summary Matrix (Q4 2025)

| Model | Provider | Type | GPQA | SWE-bench | HumanEval | LMArena Elo |
|-------|----------|------|------|-----------|-----------|-------------|
| GPT-5 | OpenAI | Closed | 87.3% | 74.9% | ~70% | ~1300 |
| Claude 4 Opus | Anthropic | Closed | 67.9% | **72.5%** | ~68% | ~1280 |
| Gemini 2.5 Pro | Google | Closed | 86.4% | N/R | N/R | **1315** |
| Llama 3.1 405B | Meta | Open | N/R | N/R | ~65% | ~1250 |
| DeepSeek-V3 | DeepSeek | Open | N/R | N/R | **~70%** | ~1260 |
| Mixtral 8x7B | Mistral | Open | N/R | N/R | ~60% | ~1200 |

**Legend**: N/R = Not Reported
