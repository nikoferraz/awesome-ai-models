# Cost Analysis

[← Back to Main List](README.md)

## Commercial API Pricing (Q4 2025)

### Comparison Table

| Model | Input ($/1M) | Output ($/1M) | Context | $/Query (256 out) |
|-------|--------------|---------------|---------|-------------------|
| **GPT-5** | $1.25 | $10.00 | 400K | $0.00256 |
| **GPT-4.1** | $2.00 | $8.00 | 1.05M | $0.00205 |
| **GPT-4.1 Mini** | $0.40 | $1.60 | 1.05M | $0.00041 |
| **GPT-4.1 Nano** | $0.10 | $0.40 | 1.05M | $0.00010 |
| **Claude 4 Opus** | $15.00 | $75.00 | 200K | $0.01920 |
| **Claude 4 Sonnet** | $3.00 | $15.00 | 200K | $0.00384 |
| **Claude 3 Haiku** | $0.25 | $1.25 | 200K | $0.00032 |
| **Gemini 2.5 Pro** | $1.25 | $10.00 | 1M+ | $0.00256 |
| **Gemini 2.5 Flash** | $0.30 | $2.50 | 1M+ | $0.00064 |

**Assumptions**: 50/50 input/output mix for $/Query calculation

---

## Monthly Cost Scenarios

| Volume | GPT-4.1 Mini | Claude Sonnet | Gemini Flash | GPT-5 |
|--------|--------------|---------------|--------------|-------|
| **10M tokens/mo** | $10 | $90 | $14 | $56 |
| **100M tokens/mo** | $100 | $900 | $140 | $563 |
| **1B tokens/mo** | $1,000 | $9,000 | $1,400 | $5,625 |
| **10B tokens/mo** | $10,000 | $90,000 | $14,000 | $56,250 |

---

## Cost Optimization Strategies

### Prompt Caching

Reduces cost of repeated input tokens by up to **90%**.

| Provider | Cache Discount | Use Case |
|----------|----------------|----------|
| OpenAI | 50% discount | Static system prompts |
| Anthropic | 90% discount | Large context reuse |
| Google | 75% discount | Repeated documents |

**Example**: 100K token prompt sent 1,000 times
- Without cache: $200
- With cache (90%): $20
- **Savings: $180 (90%)**

### Batch Processing

| Provider | Batch Discount | Latency |
|----------|----------------|---------|
| Azure OpenAI | 50% | 24 hours |
| Anthropic | 20% | 12 hours |

### Dynamic Model Routing

Route queries by complexity to optimize cost/quality.

**Example Architecture**:
```
Simple Query → Haiku ($0.00032/query)
Medium Query → Sonnet ($0.00384/query)
Complex Query → Opus ($0.01920/query)
```

**Potential Savings**: 40-60% with 80/15/5 distribution

---

## Self-Hosted Total Cost of Ownership (TCO)

### TCO Formula

```
TCO = Hardware Costs (CapEx or Rental) + OpEx (Engineering, Power, Maintenance)
```

### Hardware Options

#### Option 1: Purchase GPUs

| Hardware | Cost | VRAM | Model Fit | Amortization (3yr) |
|----------|------|------|-----------|-------------------|
| RTX 4090 | $1,800 | 24 GB | 7B-13B (quantized) | $50/month |
| RTX 5090 | $2,400 | 32 GB | 13B-34B (quantized) | $67/month |
| A100 (80GB) | $15,000 | 80 GB | 70B (quantized) | $417/month |
| H100 (80GB) | $35,000 | 80 GB | 405B (multi-GPU) | $972/month |

**Operating Costs**:
- Power: $50-$150/month (depends on utilization)
- Cooling: $20-$50/month (for on-prem)
- Internet: $50-$200/month
- **Engineering**: $5,000-$15,000/month (often largest cost)

#### Option 2: Cloud GPU Rental

**Major Hyperscalers** (On-Demand):

| Provider | GPU | VRAM | Cost/Hour | Cost/Month (24/7) |
|----------|-----|------|-----------|-------------------|
| AWS | A100 | 80 GB | $4.10 | $2,952 |
| GCP | A100 | 80 GB | $3.67 | $2,642 |
| Azure | A100 | 80 GB | $3.02 | $2,174 |
| AWS | H100 | 80 GB | ~$4.00 | ~$2,880 |

**Budget GPU Clouds**:

| Provider | GPU | VRAM | Cost/Hour | Cost/Month (24/7) |
|----------|-----|------|-----------|-------------------|
| **RunPod** | A100 | 80 GB | $0.79 | $569 |
| **Vast.ai** | A100 | 80 GB | $0.50-$1.29 | $360-$929 |
| **Lambda Labs** | A100 | 80 GB | $1.10 | $792 |
| **Lambda Labs** | H100 | 80 GB | $1.99 | $1,433 |

**Note**: Budget providers offer **50-80% savings** vs hyperscalers

---

## Break-Even Analysis

**Crossover Point**: When API costs exceed self-hosted TCO

### Example: Llama 3.1 70B (4-bit) on RunPod A100

**Self-Hosted TCO**:
- GPU rental: $569/month (A100 on RunPod)
- Engineering (part-time): $2,000/month (conservative)
- Tools/monitoring: $100/month
- **Total: $2,669/month**

**Equivalent API Usage** (GPT-4.1 Mini @ $0.00041/query):
- Break-even: 6.5M queries/month
- Or: ~50-100M tokens/month (depends on query size)

**Conclusion**:
- **Below 50M tokens/month**: APIs cheaper
- **50-100M tokens/month**: Transition zone
- **Above 100M tokens/month**: Self-hosting strongly favored

---

## 100M Tokens/Month Cost Comparison

| Deployment | Monthly Cost | $/1M Tokens |
|------------|--------------|-------------|
| **Self-hosted (RTX 4090)** | $179 (fixed) | $1.79 |
| **Self-hosted (A100 rental)** | $569 (RunPod) + $500 ops | $10.69 |
| **GPT-4o mini API** | $375 | $3.75 |
| **Claude Sonnet API** | $900 | $9.00 |
| **GPT-5 API** | $563 | $5.63 |

**Winner at 100M tokens**: Self-hosted RTX 4090 (if hardware already owned)

**Winner without hardware**: GPT-4.1 Mini API ($100/month at 100M tokens)

---

## Cost Decision Framework

```
                ┌─────────────────────────────┐
                │ Token Volume?               │
                └─────────────┬───────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
          < 10M tokens    10-100M       > 100M
                │             │             │
                │             │             │
        ┌───────▼───────┐     │     ┌───────▼────────┐
        │ Use API       │     │     │ Self-Host      │
        │ (No-brainer)  │     │     │ (Economics win)│
        └───────────────┘     │     └────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │ Transition Zone    │
                    │                    │
                    │ Consider:          │
                    │ • Growth trajectory│
                    │ • Engineering cost │
                    │ • Privacy needs    │
                    │ • Customization    │
                    └────────────────────┘
```
