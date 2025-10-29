# Awesome AI Models [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)

> A curated list of AI models covering language, vision, audio, and multimodal models. Includes both commercial APIs and open-source models with licensing, performance, and deployment information.

**📚 Detailed Guides**: [Strategic Guide](GUIDE.md) | [Benchmarks](BENCHMARKS.md) | [Cost Analysis](COST_ANALYSIS.md) | [Deployment](DEPLOYMENT.md) | [Case Studies](CASE_STUDIES.md)

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Contents

- [Commercial Language Models](#commercial-language-models)
- [Open-Source Language Models](#open-source-language-models)
- [Physical AI & Robotics Models](#physical-ai--robotics-models)
- [Biomedical AI Models](#biomedical-ai-models)
- [Vision Models](#vision-models)
- [Multimodal Models](#multimodal-models)
- [Embedding Models](#embedding-models)
- [Audio Models](#audio-models)
- [Deployment Tools](#deployment-tools)
- [Evaluation & Benchmarking](#evaluation--benchmarking)
- [AI Research Networks](#ai-research-networks)
- [Learning Resources](#learning-resources)
- [Contributing](#contributing)

---

## Commercial Language Models

*Closed-source models available via API with enterprise support.*

### OpenAI

* [o1-pro](https://openai.com/o1) - Advanced reasoning model with inference-time compute, 89th percentile on Codeforces, 83% on AIME 2024.
* [o1](https://openai.com/o1) - Production reasoning model with chain-of-thought, excels at math and code challenges.
* [o1-preview](https://openai.com/o1) - Early version of reasoning models with extended thinking time.
* [o1-mini](https://openai.com/o1) - Efficient reasoning model optimized for speed and cost.
* [GPT-5](https://openai.com/gpt-5) - Latest flagship with 400K context, intelligent task routing between fast and deep reasoning modes. GPQA: 87.3%, SWE-bench: 74.9%.
* [GPT-4.1](https://openai.com/gpt-4-1) - Long-context model with 1.05M token window, optimized for document analysis and coding.
* [GPT-4.1 Mini](https://openai.com/gpt-4-1-mini) - Cost-effective long-context model. $0.40/$1.60 per 1M tokens.
* [GPT-OSS 120B](https://github.com/openai/gpt-oss) - First open-weight release from OpenAI, Apache 2.0 license, 120B parameters with 5.1B active.

### Anthropic

* [Claude 4 Opus](https://anthropic.com/claude) - Enterprise-focused model with Constitutional AI training. SWE-bench leader: 72.5%. $15/$75 per 1M tokens.
* [Claude 4 Sonnet](https://anthropic.com/claude) - Balanced model for enterprise use, exceptional coding performance. $3/$15 per 1M tokens.
* [Claude 3 Haiku](https://anthropic.com/claude) - Fast, cost-effective model for high-volume tasks. $0.25/$1.25 per 1M tokens.

### Google

* [Gemini 2.5 Pro](https://cloud.google.com/vertex-ai/generative-ai/docs/models) - Native multimodal model with 1M-2M token context. LMArena top rank: 1315 Elo. $1.25/$10 per 1M tokens.
* [Gemini 2.5 Flash](https://cloud.google.com/vertex-ai/generative-ai/docs/models) - Price/performance optimized multimodal model. $0.30/$2.50 per 1M tokens.
* [Gemma 3 27B](https://ai.google.dev/gemma) - Open-weight model with 128K context, Gemma Terms of Use license.
* [Gemma 2 9B/27B](https://huggingface.co/google/gemma-2-9b) - Lightweight open models with strong performance, Gemma Terms of Use license.

### Other Providers

* [Cohere Command R+](https://cohere.com/) - Enterprise RAG-optimized model with strong retrieval capabilities. $3/$15 per 1M tokens.
* [Cohere Embed v3](https://cohere.com/embed) - Multilingual embedding model for semantic search. $0.10 per 1M tokens.
* [xAI Grok 4](https://x.ai/) - Highest GPQA score (87.5%), strong reasoning capabilities.
* [AI21 Jamba 1.7](https://www.ai21.com/jamba) - Hybrid Mamba-Transformer with 256K context window, longest available commercially.
* [Reka Core](https://www.reka.ai/) - 67B multimodal model handling text, images, videos, audio across 32 languages.
* [Perplexity Sonar](https://www.perplexity.ai/) - Real-time web-grounded model based on Llama 3.3 70B with 95% accuracy rates.
* [Inflection 3.0](https://inflection.ai/) - Achieves 94% of GPT-4 performance with only 40% of the compute.

---

## Open-Source Language Models

*Self-hostable models with varying license terms.*

### NVIDIA

* [Nemotron Nano 3](https://build.nvidia.com/nvidia/nemotron-nano-3) - Hybrid MoE architecture for reasoning tasks in software development and IT support, Apache 2.0.
* [Nemotron Safety Guard](https://build.nvidia.com/nvidia/nemotron-safety-guard) - Multilingual content moderation across 23 safety categories, Apache 2.0.

### IBM Granite

* [Granite 4.0](https://huggingface.co/ibm-granite/granite-4.0-8b-instruct) - Hybrid Mamba/transformer architecture with 70% memory reduction, optimized for enterprise, Apache 2.0.
* [Granite 3.0 MoE](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct) - Mixture-of-experts models for low-latency applications, Apache 2.0 with IP indemnification.
* [Granite Guardian](https://huggingface.co/ibm-granite/granite-guardian-3.0-8b) - Safety and harm detection model for enterprise AI systems, Apache 2.0.

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

* [DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) - 671B reasoning model matching o1 performance, pure RL training, MIT license. AIME 2024: 79.8%.
* [DeepSeek-R1-Distill](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B) - Distilled variants (1.5B to 70B) maintaining reasoning capabilities, MIT license.
* [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3) - 671B MoE (37B active), HumanEval ~70%, MIT license. Advanced MLA architecture.
* [DeepSeek-Coder-V2](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2) - 236B total (21B active), 90.2% on HumanEval, supports 338 programming languages, MIT license.

### Alibaba Qwen

* [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B-Preview) - Reasoning model achieving 50% on AIME 2024, beating o1-preview on some benchmarks, Apache 2.0.
* [Qwen 2.5 series](https://huggingface.co/Qwen) - 0.5B to 72B models with unprecedented range, Apache 2.0.
* [Qwen2.5-Coder 32B](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) - Matches GPT-4o on code tasks, 128K context, Apache 2.0.
* [Qwen 2.5 72B](https://huggingface.co/Qwen/Qwen2.5-72B) - 72B general-purpose model, 128K context, Apache 2.0.
* [Qwen3-VL 32B](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct) - 32B vision-language handling 20+ minute videos, 256K context, Apache 2.0.
* [Qwen 2.5 VL](https://huggingface.co/Qwen/Qwen2.5-VL) - Multimodal model with 29-language OCR support, Apache 2.0.

### Microsoft Phi

* [Phi-4](https://huggingface.co/microsoft/phi-4) - 14B model outperforming Llama 3.3-70B on math (80.4% MATH benchmark), trained on synthetic data, MIT license.
* [Phi-3.5-mini](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) - 3.8B with 128K context, 86.2% on GSM8K, 2.4GB quantized, supports 20+ languages, MIT license.
* [Phi-3 Mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) - 3.8B small language model, 68.8% MMLU, device-suitable, MIT license.
* [Phi-3 Small](https://huggingface.co/microsoft/Phi-3-small-8k-instruct) - 7B model with ~71% MMLU, MIT license.
* [Phi-3 Medium](https://huggingface.co/microsoft/Phi-3-medium-4k-instruct) - 14B model with ~75% MMLU, MIT license.

### EleutherAI

* [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6b) - 6B parameter open model for smaller-scale deployments, Apache 2.0 license.

### Technology Innovation Institute

* [Falcon 180B](https://huggingface.co/tiiuae/falcon-180B) - 180B parameter model trained on 3.5 trillion tokens, Apache 2.0 license.

### BigScience

* [BLOOM 176B](https://huggingface.co/bigscience/bloom) - 176B multilingual model supporting 46 languages and 13 programming languages, RAIL v1.0 license.

### G42 / Cerebras

* [Jais 30B](https://huggingface.co/core42/jais-30b-chat-v1) - 30B Arabic-English bilingual model optimized for Arabic language tasks, Apache 2.0 license.

### Tsinghua University

* [GLM-130B](https://github.com/THUDM/GLM-130B) - 130B bilingual Chinese-English model with open weights, Apache 2.0 license.

### Aleph Alpha

* [Pharia-1-LLM 7B](https://huggingface.co/Aleph-Alpha/Pharia-1-LLM-7B-control) - European sovereign AI optimized for German, French, Spanish, with explainable AI features, Open Aleph License (non-commercial).
* [TFree-HAT 7B](https://huggingface.co/Aleph-Alpha/llama-tfree-hat-pretrained-7b-dpo) - Tokenizer-free hierarchical autoregressive transformer for multilingual fairness, Open Aleph License.

### SenseTime

* [SenseNova V6.5](https://www.sensetime.com/en) - 600B+ parameter multimodal model for text, image, and video processing, proprietary API access.

### Hugging Face

* [SmolLM3-3B](https://huggingface.co/HuggingFaceTB/SmolLM3-3B) - 3B model with dual reasoning modes trained on curated synthetic data, Apache 2.0.
* [SmolLM2 series](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) - 135M to 1.7B parameter models for efficient edge deployment, Apache 2.0.

### Databricks

* [DBRX](https://huggingface.co/databricks/dbrx-base) - 132B total parameters (36B active) using fine-grained MoE, 2x faster inference than Llama2-70B, Apache 2.0.

### Allen AI

* [OLMo 2](https://huggingface.co/allenai/OLMo-2-1124-32B) - 32B fully-open model outperforming GPT-3.5-Turbo with all training data and 500+ checkpoints, Apache 2.0.
* [OLMo 2-1B](https://huggingface.co/allenai/OLMo-2-1B) - 1B model surpassing Gemma 3-1B and Llama 3.2-1B, Apache 2.0.

### 01.AI

* [Yi-Coder 9B](https://huggingface.co/01-ai/Yi-Coder-9B) - 9B code model supporting 52 programming languages with 100% accuracy on code needle tests, Apache 2.0.
* [Yi-Coder 1.5B](https://huggingface.co/01-ai/Yi-Coder-1.5B) - Efficient 1.5B code model for edge deployment, Apache 2.0.

---

## Physical AI & Robotics Models

*Models for autonomous systems, robotics, and simulation.*

### NVIDIA Cosmos

* [Cosmos Predict 2.5](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/models/cosmos-predict-2-5) - Generative model creating 30-second video simulations from single frames, NVIDIA Open Model License.
* [Cosmos Transfer 2.5](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/models/cosmos-transfer-2-5) - High-quality training data generation from 3D scenes for physical AI, NVIDIA Open Model License.
* [Cosmos Reason](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/cosmos/models/cosmos-reason) - Vision-language model for multimodal understanding in robotics, Apache 2.0.

### NVIDIA Isaac

* [Isaac GR00T N1.6](https://developer.nvidia.com/isaac/gr00t) - Foundation model for humanoid robotics with whole-body control, Apache 2.0.

---

## Biomedical AI Models

*Specialized models for healthcare, life sciences, and drug discovery.*

### NVIDIA Clara

* [Clara CodonFM](https://developer.nvidia.com/clara) - RNA modeling for therapy design, NVIDIA Clara SDK License.
* [Clara La-Proteina](https://developer.nvidia.com/clara) - 3D protein structure generation for drug discovery, NVIDIA Clara SDK License.
* [Clara Reason](https://developer.nvidia.com/clara) - Vision-language model for explainable medical imaging AI, NVIDIA Clara SDK License.

---

## Vision Models

*Models specialized for image understanding and generation.*

### Vision-Language Models

* [InternVL3-78B](https://huggingface.co/OpenGVLab/InternVL3-78B) - State-of-the-art vision-language model, 78B parameters, MIT license.
* [LLaVA-NeXT-34B](https://huggingface.co/llava-hf/llava-v1.6-34b-hf) - 34B visual reasoning model, Apache 2.0 license.
* [CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) - 19B vision understanding model, Apache 2.0 license.

### Image Generation

* [Flux.1 Pro](https://blackforestlabs.ai/) - Highest Elo ratings (~1060) for image quality with 6x faster Flux 1.1 Pro variant.
* [SD 3.5 Large](https://huggingface.co/stabilityai/stable-diffusion-3.5-large) - 8.1B parameters, leads in prompt adherence, includes Turbo (4-step) and Medium (2.5B) variants, OpenRAIL license.
* [Playground v3](https://playground.com/) - Integrates Llama3-8B for unprecedented text rendering with 82% text-synthesis scores.
* [Ideogram v2.0](https://ideogram.ai/) - 2x better text accuracy than DALL-E 3, multiple styles (Realistic, Design, 3D, Anime), handles 3:1 aspect ratios.
* [Imagen 3](https://deepmind.google/technologies/imagen-3/) - Up to 2K resolution with SynthID watermarking for AI detection, Google Cloud integration.
* [Adobe Firefly Image 3](https://www.adobe.com/products/firefly.html) - Commercially-safe generation trained on Adobe Stock, 4K output with C2PA transparency.
* [SDXL Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) - High-quality 1024px images in 1-4 steps (seconds vs minutes), Apache 2.0.
* [Stable Diffusion XL](https://github.com/Stability-AI/generative-models) - High-quality text-to-image diffusion model, OpenRAIL license.
* [PixArt Sigma](https://huggingface.co/PixArt-alpha/PixArt-Sigma) - Efficient 4K image generation at 0.6B parameters, Apache 2.0.

### Commercial Vision APIs

* [DALL-E 3](https://openai.com/dall-e-3) - OpenAI's image generation API. $0.04-$0.12 per image.
* [Midjourney](https://midjourney.com/) - High-quality image generation service. $10-$120/month subscription.
* [GPT-4V](https://openai.com/gpt-4) - Vision-enabled GPT-4 for image understanding.

### Video Generation

**Commercial:**
* [Sora 2](https://openai.com/sora) - Up to 60-second 4K videos with native audio and synchronized dialogue, supports multi-scene narratives.
* [Veo 3.1](https://deepmind.google/technologies/veo/) - 4K resolution with native audio, lip-sync, camera controls (pan/zoom/tilt), character consistency.
* [Runway Gen-3 Alpha](https://runwayml.com/) - 1080p up to 16 seconds with Motion Brush, Advanced Camera Controls, 30+ visual style presets.
* [Pika 2.2](https://pika.art/) - 10-second clips with Pikaframes (keyframe transitions), Pikaswaps (inpainting), Pikadditions (object insertion).

**Open-Source:**
* [CogVideoX-5B](https://huggingface.co/THUDM/CogVideoX-5b) - 10-second videos at 720x480, enhanced 1.5 version available, Apache 2.0.
* [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo) - 13B parameters outperforming commercial models in evaluations, Tencent license.
* [Stable Video Diffusion](https://stability.ai/stable-video) - 2-4 second video generation from images, OpenRAIL license.
* [AnimateDiff-Lightning](https://huggingface.co/ByteDance/AnimateDiff-Lightning) - 10x faster generation with ControlNet integration, Apache 2.0.

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

*Speech-to-text, text-to-speech, and music generation models.*

### Speech-to-Text

* [Whisper Large-v3-Turbo](https://huggingface.co/openai/whisper-large-v3-turbo) - 4-layer decoder for 6x faster processing, maintains Large-v2 performance, 98+ languages, MIT license.
* [Whisper Large-v3](https://github.com/openai/whisper) - State-of-the-art ASR, 1.5-2.0% WER, MIT license.
* [Distil-Whisper](https://huggingface.co/distil-whisper/distil-large-v3) - Within 1% WER of Large-v3 using 2-layer decoder, MIT license.
* [Moonshine](https://huggingface.co/usefulsensors/moonshine) - 27M-62M parameters, runs 5x faster than Whisper on edge devices, Apache 2.0.
* [Seamless M4T v2](https://huggingface.co/facebook/seamless-m4t-v2-large) - 100+ languages for speech/text translation, 20% BLEU improvement, MIT license.
* [Wav2Vec 2.0](https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec) - Self-supervised speech model, ~2.5% WER, MIT license.

### Text-to-Speech

* [Kokoro TTS](https://huggingface.co/hexgrad/Kokoro-82M) - 82M parameters, ranks #1 on TTS Arena, 210× real-time speed on RTX 4090, Apache 2.0.
* [Fish Speech OpenAudio S1](https://huggingface.co/fishaudio/fish-speech-1.4) - 4B parameters trained on 1M+ hours, WER 0.008, Apache 2.0.
* [StyleTTS 2](https://github.com/yl4579/StyleTTS2) - Surpasses human recordings on single-speaker datasets, MIT license.
* [MetaVoice-1B](https://huggingface.co/metavoiceio/metavoice-1B-v0.1) - Zero-shot cloning with 30-second reference audio, Apache 2.0.
* [E2 TTS](https://github.com/lucidrains/e2-tts-pytorch) - Simplified architecture achieving SOTA intelligibility, MIT license.
* [Coqui XTTS v2](https://github.com/coqui-ai/TTS) - High-quality TTS with voice cloning, MPL 2.0 license.
* [Piper](https://github.com/rhasspy/piper) - Fast lightweight TTS for production, MIT license.

### Music Generation

* [MusicGen](https://huggingface.co/facebook/musicgen-large) - 300M to 3.3B parameters, trained on 20K hours licensed music, 32kHz stereo, MIT license.
* [Stable Audio Open](https://huggingface.co/stabilityai/stable-audio-open-1.0) - 1.1B parameters, up to 47-second generation, trained on royalty-free content, OpenRAIL license.
* [Stable Audio 2.5](https://stability.ai/stable-audio) - Commercial version up to 3 minutes with audio-to-audio transformation.
* [AudioLDM 2](https://huggingface.co/cvssp/audioldm2) - Unified framework for audio, music, and speech generation, MIT license.

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

* [SGLang](https://github.com/sgl-project/sglang) - 5x faster inference with RadixAttention prefix caching, 5,000-10,000 tokens/sec, Apache 2.0.
* [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) - 8x throughput increase with FP8/FP4 quantization, day-one support for latest models, Apache 2.0.
* [vLLM](https://docs.vllm.ai/) - Industry-leading inference server with PagedAttention, 24x faster than HF, Apache 2.0.
* [Text Generation Inference](https://github.com/huggingface/text-generation-inference) - HuggingFace's production server with monitoring, Apache 2.0.
* [LiteLLM](https://github.com/BerriAI/litellm) - Unified access to 100+ LLM providers, handles 1,500+ req/sec with automatic retry/fallback, MIT license.
* [llama.cpp](https://github.com/ggerganov/llama.cpp) - C++ inference library for CPU and diverse hardware, MIT license.

### Frameworks & Orchestration

* [LangChain](https://github.com/langchain-ai/langchain) - Framework for LLM applications and agents, MIT license.
* [LangGraph](https://github.com/langchain-ai/langgraph) - Graph-based stateful workflows with 20,300+ GitHub stars, used by Uber, LinkedIn, Klarna, MIT license.
* [LlamaIndex](https://github.com/run-llama/llama_index) - Data framework for RAG applications, MIT license.
* [PydanticAI](https://github.com/pydantic/pydantic-ai) - Type-safe framework with Pydantic validation and FastAPI-inspired design, MIT license.
* [Haystack 2.0](https://github.com/deepset-ai/haystack) - Composable pipeline architecture with 50+ integrations, K8s-native deployment, Apache 2.0.
* [Burr](https://github.com/dagworks-inc/burr) - State machine applications with real-time telemetry UI and pluggable persistence, BSD-3 license.
* [DSPy](https://github.com/stanfordnlp/dspy) - Framework for prompt optimization, MIT license.
* [AutoGen](https://github.com/microsoft/autogen) - Multi-agent framework by Microsoft with AutoGen Studio for no-code prototyping, MIT license.
* [CrewAI](https://github.com/joaomdmoura/crewAI) - Role-based multi-agent framework with intuitive team metaphor, MIT license.
* [Outlines](https://github.com/outlines-dev/outlines) - Structured output generation for LLMs, Apache 2.0.

### Model Serving & MLOps

* [Docker AI](https://www.docker.com/solutions/docker-ai/) - Containerization platform with Model Runner for OCI-compliant LLMs and Docker Compose for AI apps.
* [Kubeflow](https://www.kubeflow.org/) - Open-source MLOps platform with KServe for inference and KFP for pipelines, Apache 2.0.
* [Ray Serve](https://docs.ray.io/en/latest/serve/index.html) - Distributed model serving framework, Apache 2.0.
* [BentoML](https://github.com/bentoml/BentoML) - ML deployment platform with standardized Bento packaging, Apache 2.0.
* [Triton Inference Server](https://github.com/triton-inference-server/server) - NVIDIA-optimized serving, BSD-3 license.

### Cloud AI Platforms

* [Modal](https://modal.com/) - Serverless GPU compute with sub-second container launches and automatic scaling, 8-figure revenue.
* [Groq](https://groq.com/) - Custom LPU silicon delivering 10x+ speed with 241+ tokens/second, $2.5B valuation.
* [Replicate](https://replicate.com/) - Run 1000s of models via single API with pay-per-compute, Cog tool for custom packaging.
* [Fireworks AI](https://fireworks.ai/) - Processing 140B+ tokens daily with FireAttention optimization, $4B valuation.
* [Together AI](https://together.ai/) - Together Inference Engine 4x faster than vLLM with FlashAttention-3, ATLAS speculator provides 400% speedup.
* [Baseten](https://baseten.co/) - TensorRT-LLM integration with Baseten Chains for 6x better GPU usage, $825M valuation.

---

## Evaluation & Benchmarking

*Tools and resources for evaluating model performance.*

### Benchmarking Platforms

* [LMSYS Chatbot Arena](https://lmarena.ai/) - Human preference rankings through blind A/B testing with Elo ratings.
* [Hugging Face Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) - Standardized benchmark rankings using Eleuther AI harness.
* [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) - Massive Text Embedding Benchmark evaluating embedding models across 58 datasets.
* [Vellum AI Leaderboard](https://www.vellum.ai/llm-leaderboard) - Practical, non-saturated benchmark tests.
* [PapersWithCode](https://paperswithcode.com/task/language-modelling) - Academic benchmark tracking and SOTA results.
* [Artificial Analysis](https://artificialanalysis.ai/) - Comprehensive LLM performance comparison.

### Evaluation Frameworks

* [EleutherAI LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) - Standardized evaluation framework for language models.
* [MTEB](https://github.com/embeddings-benchmark/mteb) - Massive Text Embedding Benchmark for embedding models with 8 task categories.
* [OpenAI Evals](https://github.com/openai/evals) - Framework for evaluating LLM performance.
* [PromptLayer](https://promptlayer.com/) - Prompt engineering toolkit with automated regression tests and historical backtests.
* [Galileo](https://www.rungalileo.io/) - AI observability platform with automated accuracy, safety, and performance metrics.
* [EvalAI](https://eval.ai/) - Open-source platform for hosting AI challenges and maintaining benchmarking leaderboards.

---

## AI Research Networks

*Global research collaborations and innovation hubs.*

### European Networks

* [ELLIS](https://ellis.eu/) - European Laboratory for Learning and Intelligent Systems, network of 1,800+ researchers across 44 sites in 17 countries.
* [CAIRNE](https://cairne.eu/) - Confederation of Laboratories for AI Research in Europe, 500+ research groups focused on trustworthy AI.

### Asian Networks

* [National AI Research Lab](https://nairl.kr/) - South Korea's collaborative AI research center with KAIST, POSTECH, Naver, and LG.
* [Taiwan AI Labs](https://www.ailabs.tw/) - Premier non-governmental AI organization in Asia focused on sustainability and local ecosystem.
* [AIRC](https://www.airc.aist.go.jp/en/intro/) - Japan's Artificial Intelligence Research Center promoting practical AI implementation.

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
