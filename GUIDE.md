# Guide

[← Back to Main List](README.md)

## TL;DR Executive Summary

### Key Findings (Q4 2025)

**Performance Gap Has Closed**: Open-source models like DeepSeek-V3, Llama 3.1 405B, and Mixtral now rival top commercial offerings on standardized benchmarks. The era of a single "best model" is over—specialization has emerged.

**Cost Crossover Point**: Self-hosting becomes economically superior at **~50-100M tokens/month** ($5K-$15K/month in API costs). High-volume applications should architect for migration to self-hosted solutions.

**Specialization Matters**:
- **Enterprise Coding**: Claude 4 Opus/Sonnet (72.5% SWE-bench)
- **Multimodal Analysis**: Gemini 2.5 Pro (1M+ token context)
- **Tool-Using Agents**: Llama 3.1 405B (81.1% BFCL)
- **Cost Efficiency**: Mixtral 8x7B (MoE architecture, single GPU)
- **Mathematical Reasoning**: DeepSeek-V3, GPT-5
- **On-Device**: Phi-3 Mini (3.8B, 68.8% MMLU)

**Licensing Spectrum**: "Open source" ranges from permissive (Apache 2.0, MIT) to restrictive "open-weight" (Llama Community License with 700M user restriction). Legal due diligence is mandatory.

**Maturation of Tooling**: vLLM, Text Generation Inference, and Ollama have dramatically lowered the barrier to self-hosting, making it viable for broader audiences.

### Strategic Recommendations

1. **API First, Self-Host Second**: Prototype with commercial APIs (speed to market), architect for future migration to self-hosted (cost optimization at scale)

2. **Build Internal Evals**: Public benchmarks are insufficient. Invest in use-case-specific evaluation pipelines benchmarked against your unique data

3. **Licensing as Core Feature**: Treat legal terms as non-negotiable requirements. Prioritize permissive licenses (Apache 2.0, MIT) for maximum flexibility

**Sources**: [Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/04/what-are-llm-benchmarks/), [Arize AI](https://arize.com/blog/llm-benchmarks-mmlu-codexglue-gsm8k), [Evidently AI](https://www.evidentlyai.com/llm-guide/llm-benchmarks) (Retrieved: 2025-10-28)

---

## Decision Framework

### Unified Decision Matrix

Scored 1-5 (5 = best) across five decision vectors.

| Model | Performance | Cost-Effectiveness | Control | Operational Ease | Licensing Freedom |
|-------|-------------|-------------------|---------|------------------|-------------------|
| **GPT-5** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Claude 4 Opus** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Gemini 2.5 Pro** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐⭐ | ⭐ |
| **Llama 3.1 405B** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐ |
| **DeepSeek-V3** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Mixtral 8x7B** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

**Score Justification**:
- **Performance**: Commercial models slight edge, open models highly competitive
- **Cost-Effectiveness**: Self-hosted at scale unbeatable; Mixtral best ROI
- **Control**: Open models full control; commercial zero control
- **Operational Ease**: Commercial turnkey; self-hosting requires engineering
- **Licensing Freedom**: Apache 2.0/MIT maximum freedom; proprietary most restrictive

---

### Selection by Use Case

#### High-Stakes Enterprise Coding

**Requirements**: Maximum accuracy, reliability, security

**Recommendation**: **Claude 4 Opus** or **Sonnet**

**Why**:
- Leading SWE-bench scores (72.5-72.7%)
- Constitutional AI for reliability
- Proven enterprise adoption

**Alternative**: Self-hosted **Llama 3.1 70B** or **DeepSeek-V3** (if data privacy critical)

---

#### Multimodal Analysis (Documents, Images, Video)

**Requirements**: Native multimodality, massive context

**Recommendation**: **Gemini 2.5 Pro**

**Why**:
- 1M-2M token context
- Native multimodal processing
- Top LMArena ranking (human preference)

**Alternative**: **Qwen3-VL** (open, 256K context, 29-language OCR)

---

#### Tool-Using Agents / Autonomous Workflows

**Requirements**: Function calling, multi-step reasoning, customization

**Recommendation**: **Llama 3.1 405B** or **70B**

**Why**:
- **Leading BFCL score** (81.1%)
- Full control over prompts/tools
- Cost-effective at scale

**Alternative**: **GPT-5** (if API acceptable)

---

#### Cost-Sensitive High-Volume Applications

**Requirements**: Minimize cost per query, high throughput

**Recommendation**: **Mixtral 8x7B** (self-hosted)

**Why**:
- Runs on single RTX 4090 (4-bit)
- Apache 2.0 license
- Strong performance/cost ratio

**Alternative**: **Phi-3 Mini** (if on-device/edge)

---

#### On-Device / Edge / Mobile

**Requirements**: Low VRAM, fast inference, offline

**Recommendation**: **Phi-3 Mini** (3.8B)

**Why**:
- 68.8% MMLU (exceptional for 3.8B)
- ~2 GB VRAM (4-bit)
- MIT license

**Alternative**: **Llama 3.1 8B** (larger but more capable)

---

#### Rapid Prototyping / MVP

**Requirements**: Speed to market, minimal engineering

**Recommendation**: **GPT-4.1 Mini** or **Claude Sonnet**

**Why**:
- Zero infrastructure setup
- Good performance/cost balance
- Mature ecosystems

**Migration Path**: Architect for model-agnostic switch to Mixtral/Llama at scale

---

### Selection by Token Volume

| Monthly Volume | Recommendation | Rationale |
|----------------|----------------|-----------|
| **< 10M tokens** | Commercial API (GPT-4.1 Mini, Gemini Flash) | No-brainer: Minimal cost, zero ops |
| **10-50M tokens** | Commercial API + monitoring | Watch for crossover point |
| **50-100M tokens** | **Transition zone** — test self-hosting | Break-even approaching |
| **100M-1B tokens** | Self-hosted (Mixtral, Llama 3.1 70B) | Economics favor self-hosting |
| **> 1B tokens** | Self-hosted clusters (Llama 3.1 405B, DeepSeek-V3) | Massive savings, control essential |

---

### Token Volume Decision Tree

```
                    ┌─────────────────┐
                    │ Token Volume?   │
                    └────────┬────────┘
                             │
                 ┌───────────┼──────────┐
                 │           │          │
           < 50M tokens   50-100M    > 100M tokens
                 │           │          │
        ┌────────▼─────┐     │     ┌────▼─────────┐
        │ Commercial   │     │     │ Self-Hosted  │
        │ API          │     │     │ Open Models  │
        │              │     │     │              │
        │ • GPT-4.1 Mini│    │     │ • Mixtral    │
        │ • Claude Sonnet│   │     │ • Llama 3.1  │
        │ • Gemini Flash│    │     │ • DeepSeek   │
        └──────────────┘     │     └──────────────┘
                             │
                    ┌────────▼────────┐
                    │ **Evaluate:**   │
                    │                 │
                    │ • Growth rate   │
                    │ • Eng capacity  │
                    │ • Privacy needs │
                    │ • Customization │
                    │                 │
                    │ **Action:**     │
                    │ Test self-host  │
                    │ Plan migration  │
                    └─────────────────┘
```

---

## Licensing Guide

Licensing is as critical as performance. Legal terms can introduce significant business risk or limit strategic options.

### License Spectrum

```
Permissive FOSS ←→ Restrictive Open-Weight ←→ Proprietary
  (Maximum freedom)     (Conditional freedom)      (No freedom)
```

---

### Permissive FOSS Licenses

✅ **Apache 2.0** & **MIT**: True open source (OSI-approved)

**Grants**:
- ✅ Commercial use (unlimited)
- ✅ Modification and derivatives
- ✅ Distribution
- ✅ Private use

**Restrictions**:
- ⚠️ Must include license notice
- ⚠️ Must state changes (Apache 2.0)

**Models**:
- DeepSeek-V3 (MIT)
- Mixtral 8x7B (Apache 2.0)
- Mistral NeMo (Apache 2.0)
- Qwen3-VL (Apache 2.0)
- Phi-3 (MIT)
- Gemma 3 (Gemma Terms—similar to Apache 2.0)

**Risk Level**: ✅ **Low** — Maximum legal freedom

---

### Restrictive "Open-Weight" Licenses

⚠️ **Llama 3.1 Community License**: Powerful but with restrictions

**Grants**:
- ✅ Commercial use (for most users)
- ✅ Modification and derivatives
- ✅ Distribution

**Restrictions**:
- ⚠️ **Critical**: Entities with >700M monthly active users need separate license
- ⚠️ Extensive "Acceptable Use Policy" (restricts certain applications)
- ⚠️ Must include license and disclaimers

**Models**:
- Llama 3.1 series (8B, 70B, 405B)
- Llama 4 series

**FSF & OSI Position**: ❌ **Not true open source**

**Risk Level**: ⚠️ **Medium** — Good for most, blocks large competitors

**Source**: [HuggingFace Llama License](https://huggingface.co/meta-llama/Llama-3.1-405B) (Retrieved: 2024-12-10)

---

### Non-Commercial / Research Licenses

❌ **Mistral Research License**: No commercial use without agreement

**Grants**:
- ✅ Research use
- ✅ Evaluation
- ✅ Non-commercial applications

**Restrictions**:
- ❌ **Commercial use prohibited** without separate license
- ⚠️ Must contact Mistral AI for commercial terms

**Models**:
- Mistral Large 2

**Risk Level**: ⚠️ **High** — Requires negotiation for commercial use

---

### Proprietary Terms of Service

❌ **OpenAI, Anthropic, Google**: Standard API Terms

**Key Terms**:
- ❌ No model weights access
- ⚠️ Data usage for training (varies by provider)
- ⚠️ Rate limits and quotas
- ⚠️ Acceptable Use Policies
- ⚠️ No service level guarantees (standard tier)

**Risk Level**: ⚠️ **High** — Complete vendor lock-in

---

### Licensing Decision Matrix

| License Type | Commercial Use | Modifications | Distribution | Large Entity Risk |
|--------------|----------------|---------------|--------------|-------------------|
| **Apache 2.0 / MIT** | ✅ Unlimited | ✅ Yes | ✅ Yes | ✅ None |
| **Llama Community** | ⚠️ <700M users | ✅ Yes | ✅ Yes | ⚠️ High (>700M) |
| **Research License** | ❌ No (needs agreement) | ✅ Yes | ⚠️ Limited | ❌ Must negotiate |
| **Proprietary API** | ⚠️ Per ToS | ❌ No | ❌ No | ⚠️ Vendor terms |

---

### Licensing Recommendations

**For Startups**:
1. **Phase 1 (MVP)**: Use commercial APIs (OpenAI, Anthropic, Google) — speed to market
2. **Phase 2 (Growth)**: Migrate to Apache 2.0/MIT models (Mixtral, DeepSeek) — cost + control
3. **Fallback**: Llama 3.1 (if <700M users) — best performance/hardware balance

**For Enterprises**:
1. **Regulated Industries**: Prioritize Apache 2.0/MIT for compliance
2. **Internal Tools**: Llama 3.1 safe (unlikely to hit 700M internal users)
3. **Customer-Facing**: Apache 2.0/MIT only (avoid license ambiguity)

**For Researchers**:
1. **Academic**: Any license acceptable (publish source/weights)
2. **Spin-out Risk**: Use Apache 2.0/MIT to avoid commercialization blockers
