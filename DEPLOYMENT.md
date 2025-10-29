# Deployment Tools and Hardware Requirements

[← Back to Main List](README.md)

## Deployment Tools

The viability of self-hosting depends on sophisticated software tooling. This ecosystem has matured dramatically, lowering barriers to entry.

### User-Friendly Local Runners

#### Ollama

**Best For**: Easiest local LLM deployment, development, personal use

| Feature | Details |
|---------|---------|
| **Type** | Local runner + server |
| **Ease of Use** | ⭐⭐⭐⭐⭐ (5/5) |
| **Performance** | Fast quantized inference via llama.cpp |
| **Formats** | GGUF (native), HF PyTorch (via conversion) |
| **Hardware** | CPU, NVIDIA GPU, AMD GPU, Apple Silicon |
| **API** | OpenAI-compatible (built-in) |
| **License** | MIT |

**Installation**:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**Usage**:
```bash
# Pull and run model
ollama pull llama3.2
ollama run llama3.2 "Explain quantum computing"

# Start API server (automatic)
curl http://localhost:11434/v1/chat/completions \
  -d '{"model": "llama3.2", "messages": [{"role": "user", "content": "Hello"}]}'
```

**Key Features**:
- Zero-config setup
- Auto-quantization
- HuggingFace integration
- Model library with one-command pull

**GitHub Stars**: 155,000+

**Source**: [Ollama Docs](https://ollama.readthedocs.io/) (Retrieved: 2024-12-10)

---

#### LM Studio

**Best For**: Non-technical users, GUI-based local deployment

| Feature | Details |
|---------|---------|
| **Type** | Desktop app with GUI |
| **Ease of Use** | ⭐⭐⭐⭐⭐ (5/5) |
| **Performance** | llama.cpp + MLX backends |
| **Formats** | GGUF, MLX |
| **Hardware** | CPU, NVIDIA, AMD, Apple Silicon |
| **API** | OpenAI-compatible (built-in) |
| **License** | Proprietary (free) |

**Key Features**:
- Drag-and-drop model management
- Built-in model search and download
- Chat interface
- Local API server
- Cross-platform (Windows, Mac, Linux)

**Source**: [LM Studio Docs](https://lmstudio.ai/docs/app) (Retrieved: 2024-12-10)

---

### Production-Grade Inference Servers

#### vLLM

**Best For**: High-throughput production serving, highest performance

| Feature | Details |
|---------|---------|
| **Type** | Production inference server |
| **Ease of Use** | ⭐⭐⭐ (3/5) |
| **Performance** | ⭐⭐⭐⭐⭐ (5/5) Industry-leading |
| **Formats** | HF Transformers, GPTQ, AWQ, FP8 |
| **Hardware** | NVIDIA, AMD ROCm, Intel GPU, TPU, AWS Neuron |
| **API** | OpenAI-compatible (built-in) |
| **License** | Apache 2.0 |

**Performance**:
- **24x faster** than HF Transformers
- **2-4x faster** than competitors
- **2,300-2,500 tokens/sec** typical throughput

**Key Innovation**: **PagedAttention**
- Inspired by virtual memory in OS
- Non-contiguous KV cache storage
- ~4% memory waste (vs 20%+ traditional)
- Enables much higher batch sizes

**Installation**:
```bash
pip install vllm
```

**Usage**:
```bash
# Start server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.1-8B \
  --dtype float16

# Use with OpenAI SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**GitHub Stars**: 25,000+

**Source**: [vLLM Docs](https://docs.vllm.ai/en/latest/) (Retrieved: 2024-12-10)

---

#### Text Generation Inference (TGI)

**Best For**: HuggingFace ecosystem, production with observability

| Feature | Details |
|---------|---------|
| **Type** | Production inference server |
| **Ease of Use** | ⭐⭐⭐ (3/5) |
| **Performance** | ⭐⭐⭐⭐ (4/5) Competitive with vLLM |
| **Formats** | HF Transformers, Safetensors, GPTQ, AWQ |
| **Hardware** | NVIDIA, AMD ROCm, Intel, AWS Inferentia, Google TPU |
| **API** | OpenAI-compatible (built-in) |
| **License** | Apache 2.0 |

**Performance**:
- **2,300-2,500 tokens/sec** typical
- Flash/Paged Attention
- Continuous batching

**Key Features**:
- First-class HuggingFace integration
- Prometheus metrics (built-in)
- OpenTelemetry support
- Rust + Python (performance + flexibility)
- Production-grade monitoring

**Installation**:
```bash
docker run --gpus all -p 8080:80 \
  ghcr.io/huggingface/text-generation-inference:3.3.5 \
  --model-id meta-llama/Llama-3.1-8B
```

**GitHub Stars**: 8,000+

**Source**: [TGI Docs](https://huggingface.co/docs/text-generation-inference) (Retrieved: 2024-12-10)

---

### Core Inference Libraries

#### llama.cpp

**Best For**: CPU inference, portability, foundation for other tools

| Feature | Details |
|---------|---------|
| **Type** | C++ inference library |
| **Ease of Use** | ⭐⭐ (2/5 as library), ⭐⭐⭐⭐ (4/5 via frontends) |
| **Performance** | Excellent on CPU and diverse hardware |
| **Formats** | GGUF (native), HF PyTorch (via conversion) |
| **Hardware** | CPU (primary), NVIDIA CUDA, AMD ROCm, Apple Metal, Vulkan |
| **License** | MIT |

**Key Features**:
- 1.5-8 bit quantization
- No dependencies (pure C++)
- Matryoshka quantization
- Powers Ollama, LM Studio, GPT4All

**Usage**:
```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Run inference
./llama-cli -m model-q4.gguf -p "prompt" -n 128 -ngl 32
```

**Performance** (Reported):
- 100-160 tokens/sec on RTX 4090 (7B Q4)

**GitHub Stars**: 65,000+

**Source**: [llama.cpp GitHub](https://github.com/ggml-org/llama.cpp) (Retrieved: 2024-12-10)

---

### Tool Comparison Matrix

| Tool | Use Case | Ease of Use | Performance | OpenAI API | Formats | GUI |
|------|----------|-------------|-------------|------------|---------|-----|
| **Ollama** | Local dev | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | GGUF | ❌ |
| **LM Studio** | Local dev | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ✅ | GGUF, MLX | ✅ |
| **vLLM** | Production | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ✅ | HF, GPTQ, AWQ | ❌ |
| **TGI** | Production | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ | HF, GPTQ, AWQ | ❌ |
| **llama.cpp** | Core library | ⭐⭐ | ⭐⭐⭐⭐ | ❌ | GGUF | ❌ |

---

### Format Compatibility Matrix

| Tool | HF PyTorch | GGUF | GPTQ | AWQ | ONNX | TensorRT | EXL2 |
|------|------------|------|------|-----|------|----------|------|
| **Ollama** | Via convert | ✅ Native | ❌ | ❌ | ❌ | ❌ | ❌ |
| **vLLM** | ✅ Native | Limited | ✅ Native | ✅ Native | ❌ | ❌ | ❌ |
| **TGI** | ✅ Native | ❌ | ✅ Native | ✅ Native | ❌ | ❌ | ❌ |
| **llama.cpp** | Via convert | ✅ Native | ❌ | ❌ | ❌ | ❌ | ❌ |

---

### Additional Tools

#### Model Serving Frameworks

| Tool | Specialty | License |
|------|-----------|---------|
| **Ray Serve** | Distributed serving | Apache 2.0 |
| **BentoML** | ML deployment platform | Apache 2.0 |
| **Triton Inference Server** | NVIDIA-optimized | BSD-3 |

#### Agent Frameworks

| Framework | Specialty | License | Model Support |
|-----------|-----------|---------|---------------|
| **LangChain** | Orchestration | MIT | Open + Closed |
| **LlamaIndex** | RAG-focused | MIT | Open + Closed |
| **DSPy** | Prompt optimization | MIT | Open + Closed |
| **AutoGen** | Multi-agent | MIT | Open + Closed |
| **CrewAI** | Role-based agents | MIT | Open + Closed |
| **Outlines** | Structured output | Apache 2.0 | Open (HF) |

---

## Hardware Requirements

### VRAM Estimation Formula

```
VRAM_required ≈ Parameters × Bytes_per_Parameter
```

**Precision Levels**:
- **FP32** (Full): 4 bytes/param (rarely used)
- **FP16/BF16** (Half): 2 bytes/param (training/high-quality)
- **INT8** (8-bit): 1 byte/param (2x reduction, minimal loss)
- **INT4** (4-bit): 0.5 bytes/param (4x reduction, noticeable loss)

---

### VRAM Requirements by Model

#### Small Models (Consumer GPU Friendly)

| Model | Parameters | FP16 VRAM | INT8 VRAM | INT4 VRAM | Example Hardware |
|-------|------------|-----------|-----------|-----------|------------------|
| **Phi-3 Mini** | 3.8B | ~7.6 GB | ~4 GB | ~2 GB | RTX 3060 (12GB) |
| **Mistral NeMo** | 12B | ~24 GB | ~12 GB | ~6 GB | RTX 3090/4090 (24GB) |
| **Llama 3.1 8B** | 8B | ~16 GB | ~8.5 GB | ~4.7 GB | RTX 3060 (12GB) |

**Deployment**: Single consumer GPU

---

#### Medium Models (Prosumer/Workstation)

| Model | Parameters | FP16 VRAM | INT8 VRAM | INT4 VRAM | Example Hardware |
|-------|------------|-----------|-----------|-----------|------------------|
| **Mixtral 8x7B** | 47B (13B active) | ~94 GB | ~46 GB | **~27 GB** | **1x RTX 4090 (4-bit)** |
| **Qwen3-VL 32B** | 32B | ~64 GB | ~32 GB | ~16 GB | 1x RTX 4090 |
| **LLaVA-NeXT-34B** | 34B | ~68 GB | ~34 GB | ~17 GB | 1x RTX 4090 |

**Key Insight**: Mixtral 8x7B is the most powerful model runnable on a single consumer GPU with 4-bit quantization.

---

#### Large Models (Data Center GPU)

| Model | Parameters | FP16 VRAM | INT8 VRAM | INT4 VRAM | Example Hardware |
|-------|------------|-----------|-----------|-----------|------------------|
| **Llama 3.1 70B** | 71B | ~141 GB | ~75 GB | ~40 GB | 2x A100 (80GB) |
| **Qwen 2.5 72B** | 72B | ~144 GB | ~77 GB | ~41 GB | 2x A100 (80GB) |
| **Mistral Large 2** | 123B | ~246 GB | ~131 GB | ~65 GB | 4x A100 (80GB) |

**Deployment**: Multi-GPU servers

---

#### Frontier Models (Cluster Required)

| Model | Parameters | FP16 VRAM | INT8 VRAM | INT4 VRAM | Example Hardware |
|-------|------------|-----------|-----------|-----------|------------------|
| **Llama 3.1 405B** | 406B | ~812 GB | ~431 GB | ~214 GB | 3x H100 (80GB) 4-bit |
| **DeepSeek-V3** | 671B (37B active) | ~1,543 GB | N/A | ~386 GB | 5-6x H100 (80GB) 4-bit |

**Deployment**: GPU clusters with high-speed interconnect

---

### GPU Selection Guide

#### Consumer GPUs

| GPU | VRAM | Price | Best For | Example Models |
|-----|------|-------|----------|----------------|
| **RTX 3060** | 12 GB | $300 | Small models | Llama 3.1 8B (4-bit), Phi-3 |
| **RTX 3090** | 24 GB | $1,200 | Medium models | Mixtral 8x7B (4-bit) |
| **RTX 4090** | 24 GB | $1,800 | Medium models | Mixtral 8x7B (4-bit), Llama 3.1 8B (FP16) |
| **RTX 5090** | 32 GB | $2,400 | Large quantized | Llama 3.1 70B (heavily quantized) |

---

#### Data Center GPUs

| GPU | VRAM | Cloud Cost/Hr | Purchase | Best For |
|-----|------|---------------|----------|----------|
| **NVIDIA L4** | 24 GB | ~$0.75 | ~$6,000 | Inference-optimized |
| **NVIDIA A40** | 48 GB | ~$1.50 | ~$8,000 | Professional workloads |
| **NVIDIA A100** | 80 GB | $0.50-$4.10 | ~$15,000 | Large models |
| **NVIDIA H100** | 80 GB | $1.49-$4.00 | ~$35,000 | Frontier models |
| **NVIDIA B200** | 192 GB | N/A | ~$70,000 | Future largest models |

---

### Quantization Trade-offs

| Method | Size Reduction | Quality Loss | Speed | Compatibility |
|--------|----------------|--------------|-------|---------------|
| **FP16** | Baseline | None | Baseline | Universal |
| **INT8** | 2x | Minimal (<1% degradation) | 1.5-2x faster | Wide |
| **INT4 (GGUF)** | 4x | Noticeable (2-5% degradation) | 2-3x faster | GGUF tools |
| **GPTQ** | 3-4x | Low (1-3% degradation) | 2-3x faster | vLLM, TGI |
| **AWQ** | 4x | Very low (<2% degradation) | 2-3x faster | vLLM, TGI |

**Recommendation**:
- **INT8**: Best quality/size balance
- **INT4 (GGUF)**: Best for consumer hardware
- **AWQ**: Best for production (quality + speed)
