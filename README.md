# Awesome AI Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of AI models covering language, vision, audio, and multimodal models. Includes both commercial APIs and open-source models with licensing, performance, and deployment information.

**📚 Detailed Guides**: [Strategic Guide](GUIDE.md) | [Benchmarks](BENCHMARKS.md) | [Cost Analysis](COST_ANALYSIS.md) | [Deployment](DEPLOYMENT.md) | [Case Studies](CASE_STUDIES.md)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Contents

- [Commercial Language Models](#commercial-language-models)
- [Open-Source Language Models](#open-source-language-models)
- [Vision Models](#vision-models)
- [Multimodal Models](#multimodal-models)
- [Embedding Models](#embedding-models)
- [Audio Models](#audio-models)
- [Deployment Tools](#deployment-tools)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [Learning Resources](#learning-resources)
- [Contributing](#contributing)

---

## Commercial Language Models

*Closed-source models available via API with enterprise support.*

### OpenAI

* [GPT-5](https://openai.com/gpt-5) - Latest flagship with 400K context, intelligent task routing between fast and deep reasoning modes. GPQA: 87.3%, SWE-bench: 74.9%.
* [GPT-4.1](https://openai.com/gpt-4-1) - Long-context model with 1.05M token window, optimized for document analysis and coding.
* [GPT-4.1 Mini](https://openai.com/gpt-4-1-mini) - Cost-effective long-context model. $0.40/$1.60 per 1M tokens.
* [o3](https://openai.com/o3) - Specialized reasoning model for complex multi-step logical problems.
* [GPT-OSS 120B](https://github.com/openai/gpt-oss) - First open-weight release from OpenAI, Apache 2.0 license, 120B parameters with 5.1B active.

### Anthropic

* [Claude 4 Opus](https://anthropic.com/claude) - Enterprise-focused model with Constitutional AI training. SWE-bench leader: 72.5%. $15/$75 per 1M tokens.
* [Claude 4 Sonnet](https://anthropic.com/claude) - Balanced model for enterprise use, exceptional coding performance. $3/$15 per 1M tokens.
* [Claude 3 Haiku](https://anthropic.com/claude) - Fast, cost-effective model for high-volume tasks. $0.25/$1.25 per 1M tokens.

### Google

* [Gemini 2.5 Pro](https://cloud.google.com/vertex-ai/generative-ai/docs/models) - Native multimodal model with 1M-2M token context. LMArena top rank: 1315 Elo. $1.25/$10 per 1M tokens.
* [Gemini 2.5 Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models) - Price/performance optimized multimodal model. $0.30/$2.50 per 1M tokens.
* [Gemma 3 27B](https://ai.google.dev/gemma) - Open-weight model with 128K context, Gemma Terms of Use license.

### Other Providers

* [Cohere Command R+](https://cohere.com/) - Enterprise RAG-optimized model with strong retrieval capabilities. $3/$15 per 1M tokens.
* [Cohere Embed v3](https://cohere.com/embed) - Multilingual embedding model for semantic search. $0.10 per 1M tokens.
* [xAI Grok 4](https://x.ai/) - Highest GPQA score (87.5%), strong reasoning capabilities.

---

## Open-Source Language Models

*Self-hostable models with varying license terms.*

### Meta Llama Series

* [Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B) - Frontier open-weight model, tool use leader (81.1% BFCL). Llama Community License (700M user restriction).
* [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B) - Strong general-purpose model, 128K context. Runs on 2x A100 (4-bit). Community License.
* [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B) - Efficient small model, ~16GB VRAM (FP16), ~4.7GB (4-bit). Community License.
* [Llama 4 Scout](https://ai.meta.com/llama) - Emerging multimodal model with 200+ language support.
* [Llama 4 Maverick](https://ai.meta.com/llama) - Multimodal model variant with enhanced capabilities.

### Mistral AI

* [Mixtral 8x7B](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1) - MoE model (47B total, 13B active), runs on single RTX 4090 (4-bit). Apache 2.0.
* [Mistral Large 2](https://huggingface.co/mistralai/Mistral-Large-2) - 123B parameter model, 32K context. Research License (non-commercial).
* [Mistral NeMo 12B](https://huggingface.co/mistralai/Mistral-Nemo-12B) - 12B general-purpose model, Apache 2.0 license.
* [Pixtral 12B](https://huggingface.co/mistralai/Pixtral-12B) - 12B vision-language model, Apache 2.0 license.

### DeepSeek AI

* [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) - 671B MoE (37B active), HumanEval ~70%, MIT license for code. Advanced MLA architecture.
* [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) - Reasoning-focused model with MIT license.

### Alibaba Qwen

* [Qwen 2.5 72B](https://huggingface.co/Qwen/Qwen2.5-72B) - 72B general-purpose model, 128K context, Apache 2.0.
* [Qwen3-VL 32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) - 32B vision-language model with 256K context, Apache 2.0.
* [Qwen 2.5 VL](https://huggingface.co/Qwen/Qwen2.5-VL) - Multimodal model with 29-language OCR support, Apache 2.0.

### Microsoft Phi

* [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) - 3.8B small language model, 68.8% MMLU, device-suitable, MIT license.
* [Phi-3 Small](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) - 7B model with ~71% MMLU, MIT license.
* [Phi-3 Medium](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) - 14B model with ~75% MMLU, MIT license.

---

## Vision Models

*Models specialized for image understanding and generation.*

### Vision-Language Models

* [InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B) - State-of-the-art vision-language model, 78B parameters, MIT license.
* [LLaVA-NeXT-34B](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) - 34B visual reasoning model, Apache 2.0 license.
* [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) - 19B vision understanding model, Apache 2.0 license.

### Image Generation

* [Stable Diffusion XL](https://github.com/Stability-AI/generative-models) - High-quality text-to-image diffusion model, OpenRAIL license.
* [Stable Diffusion 3](https://stability.ai/stable-diffusion-3) - Latest SD version with improved quality, OpenRAIL license.
* [Flux.1 Schnell](https://github.com/black-forest-labs/flux) - Fast high-quality image generation, Apache 2.0 license.

### Commercial Vision APIs

* [DALL-E 3](https://openai.com/dall-e-3) - OpenAI's image generation API. $0.04-$0.12 per image.
* [Midjourney](https://midjourney.com/) - High-quality image generation service. $10-$120/month subscription.
* [GPT-4V](https://openai.com/gpt-4) - Vision-enabled GPT-4 for image understanding.

---

## Multimodal Models

*Models handling multiple modalities (text, image, audio, video).*

* [Gemini 2.5 Pro](https://cloud.google.com/vertex-ai/generative-ai/docs/models) - Native multimodal: text, image, audio, video. 1M-2M context.
* [GPT-4V](https://openai.com/gpt-4) - Vision-capable GPT-4 for image understanding and analysis.
* [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) - Open multimodal with strong vision capabilities, Apache 2.0.
* [Llama 4 Scout](https://ai.meta.com/llama) - Upcoming multimodal model with 200+ language support.

---

## Embedding Models

*Vector embedding models for semantic search and retrieval.*

### Open Embeddings

* [Nomic Embed v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) - 768-dim embeddings with binary support, 8K context, Apache 2.0.
* [BGE-M3](https://huggingface.co/BAAI/bge-m3) - 1024-dim multilingual embeddings with binary support, MIT license.
* [E5 Mistral-7B](https://huggingface.co/intfloat/e5-mistral-7b-instruct) - 4096-dim high-quality embeddings, 32K context, MIT license.
* [GTE-large](https://huggingface.co/thenlper/gte-large) - 1024-dim general-purpose embeddings, Apache 2.0.

### Commercial Embeddings

* [OpenAI text-embedding-3-large](https://openai.com/blog/new-embedding-models) - 3072-dim embeddings. $0.13 per 1M tokens.
* [Voyage-3-large](https://www.voyageai.com/) - State-of-the-art embeddings, +9.74% improvement vs OpenAI.
* [Cohere Embed v3](https://cohere.com/embed) - Multilingual embeddings. $0.10 per 1M tokens.

---

## Audio Models

*Speech-to-text and text-to-speech models.*

### Speech-to-Text

* [Whisper Large-v3](https://github.com/openai/whisper) - State-of-the-art ASR, 1.5-2.0% WER, MIT license.
* [Wav2Vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) - Self-supervised speech model, ~2.5% WER, MIT license.
* [HuBERT](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert) - Hidden-unit BERT for speech, ~2.8% WER, MIT license.

### Text-to-Speech

* [Coqui XTTS v2](https://github.com/coqui-ai/TTS) - High-quality TTS with voice cloning, MPL 2.0 license.
* [Bark](https://github.com/suno-ai/bark) - Multilingual TTS model, MIT license.
* [Piper](https://github.com/rhasspy/piper) - Fast lightweight TTS for production, MIT license.

### Commercial Audio APIs

* [AssemblyAI Universal-2](https://www.assemblyai.com/) - Commercial ASR with 6.68% WER.
* [ElevenLabs](https://elevenlabs.io/) - High-quality voice synthesis and cloning.

---

## Deployment Tools

*Tools for running and serving AI models.*

### Local Development

* [Ollama](https://ollama.com/) - Easiest local LLM deployment with one-command setup, OpenAI-compatible API, MIT license.
* [LM Studio](https://lmstudio.ai/) - GUI-based local model management with drag-and-drop, cross-platform, free.
* [GPT4All](https://gpt4all.io/) - Desktop app for running local LLMs, MIT license.

### Production Inference Servers

* [vLLM](https://docs.vllm.ai/) - Industry-leading inference server with PagedAttention, 24x faster than HF, Apache 2.0.
* [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - HuggingFace's production server with monitoring, Apache 2.0.
* [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ inference library for CPU and diverse hardware, MIT license.

### Frameworks & Orchestration

* [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications and agents, MIT license.
* [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for RAG applications, MIT license.
* [DSPy](https://github.com/stanfordnlp/dspy) - Framework for prompt optimization, MIT license.
* [AutoGen](https://github.com/microsoft/autogen) - Multi-agent framework by Microsoft, MIT license.
* [CrewAI](https://github.com/joaomdmoura/crewAI) - Role-based multi-agent framework, MIT license.
* [Outlines](https://github.com/outlines-dev/outlines) - Structured output generation for LLMs, Apache 2.0.

### Model Serving

* [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) - Distributed model serving framework, Apache 2.0.
* [BentoML](https://github.com/bentoml/BentoML) - ML deployment platform with packaging and serving, Apache 2.0.
* [Triton Inference Server](https://github.com/triton-inference-server/server) - NVIDIA-optimized serving, BSD-3 license.

---

## Evaluation & Benchmarking

*Tools and resources for evaluating model performance.*

### Benchmarking Platforms

* [LMSYS Chatbot Arena](https://lmarena.ai/) - Human preference rankings through blind A/B testing.
* [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - Standardized benchmark rankings for open models.
* [Vellum AI Leaderboard](https://www.vellum.ai/llm-leaderboard) - Practical, non-saturated benchmark tests.
* [PapersWithCode](https://paperswithcode.com/task/language-modelling) - Academic benchmark tracking and SOTA results.
* [Artificial Analysis](https://artificialanalysis.ai/) - Comprehensive LLM performance comparison.

### Evaluation Frameworks

* [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - Standardized evaluation framework for language models.
* [MTEB](https://github.com/embeddings-benchmark/mteb) - Massive Text Embedding Benchmark for embedding models.
* [OpenAI Evals](https://github.com/openai/evals) - Framework for evaluating LLM performance.

---

## Learning Resources

*Guides, tutorials, and educational materials.*

### Official Documentation

* [OpenAI Platform Docs](https://platform.openai.com/docs) - Comprehensive API documentation and guides.
* [Anthropic Claude Docs](https://docs.anthropic.com/) - Claude API documentation and best practices.
* [Google Cloud AI](https://cloud.google.com/ai) - Vertex AI and Gemini documentation.
* [Meta Llama Resources](https://ai.meta.com/llama/) - Official Llama model documentation.

### Guides & Tutorials

* [Hugging Face Course](https://huggingface.co/learn) - Free comprehensive course on transformers and NLP.
* [LangChain Documentation](https://python.langchain.com/) - Tutorials for building LLM applications.
* [vLLM Documentation](https://docs.vllm.ai/) - Production deployment guides.

### Community Resources

* [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA) - Community for running local language models.
* [Hugging Face Forums](https://discuss.huggingface.co/) - Technical discussions and support.

---

## Contributing

Contributions welcome! Please read the [contribution guidelines](CONTRIBUTING.md) first.

**Requirements for new entries:**
- Official repository or documentation link
- Clear license information
- At least 2 verifiable benchmark scores or performance metrics
- Active maintenance (updated within last year)

**See detailed guides:**
- [Complete Model Guide](GUIDE.md) - In-depth analysis, case studies, decision frameworks
- [Cost Analysis](COST.md) - TCO calculations, break-even analysis
- [Deployment Guide](DEPLOYMENT.md) - Hardware requirements, setup instructions
- [Benchmarks](BENCHMARKS.md) - Detailed performance comparisons

---

## License

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

---

**Last Updated**: October 28, 2025
**Maintained by**: AI Research Community
**Star this repo** if you find it useful! ⭐
