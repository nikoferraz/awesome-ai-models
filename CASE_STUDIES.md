# Case Studies

[← Back to Main List](README.md)

## Case Study 1: AI-Native SaaS Startup (RAG-Based Legal Research)

**Company Profile**:
- Stage: Seed ($2M raised)
- Product: Legal document Q&A platform
- Volume: Starting 5M tokens/month, projecting 500M/month in 12 months
- Team: 2 engineers (no dedicated MLOps)

**Requirements**:
- Speed to market (3-month runway to launch)
- High accuracy on legal documents
- Cost management as priority #2
- Potential investor data security concerns

**Decision Process**:

**Phase 1 (Months 0-3): MVP Launch**

**Choice**: **Claude 4 Sonnet**

**Rationale**:
- ✅ Zero infrastructure overhead (engineers focus 100% on product)
- ✅ Strong document understanding
- ✅ Fast iteration (no model management)
- ✅ Predictable costs at low volume ($135/month at 5M tokens)

**Architecture**:
```python
# Simple, model-agnostic wrapper
class LLMProvider:
    def complete(self, prompt):
        # Initially: Claude API
        # Future: Swap to self-hosted
        pass
```

**Phase 2 (Months 4-12): Growth**

**Volume Growth**: 5M → 150M tokens/month

**API Costs**: $135/month → $4,050/month

**Action**: Monitor and build evaluation suite

**Key Metrics**:
- Track cost per customer
- Benchmark Claude vs open alternatives (Llama 3.1 70B, Mixtral) on legal docs
- Evaluate quality delta

**Phase 3 (Month 12+): Migration**

**Volume**: 500M tokens/month projected

**API Cost Projection**: $13,500/month (Claude Sonnet)

**Self-Hosted TCO** (Llama 3.1 70B on RunPod A100):
- GPU rental: $569/month
- Part-time MLOps: $3,000/month
- Monitoring/tools: $200/month
- **Total**: $3,769/month

**Savings**: $13,500 - $3,769 = **$9,731/month** (72% reduction)

**Migration Plan**:
1. Deploy Llama 3.1 70B (4-bit) on RunPod A100
2. A/B test: 20% traffic to self-hosted
3. Monitor quality metrics (pass/fail on legal eval suite)
4. Gradual ramp: 20% → 50% → 100%
5. Keep Claude as fallback for complex queries (dynamic routing)

**Outcome**:
- ✅ Launched in 3 months (API speed)
- ✅ Migrated at optimal time (cost crossover)
- ✅ Maintained quality (internal evals)
- ✅ Gross margin improved from 60% to 75%

---

## Case Study 2: Enterprise Internal Platform (Financial Services Code Assistant)

**Company Profile**:
- Organization: Large investment bank
- Project: Internal developer assistant for proprietary codebase
- Users: 5,000 developers
- Data: Highly sensitive proprietary code

**Requirements**:
- ❌ **No external APIs** (regulatory constraints)
- ✅ Maximum code understanding
- ✅ On-premises deployment
- ✅ High availability (99.9%)
- Budget: $500K CapEx approved

**Decision Process**:

**Option A: Cloud Deployment (Disqualified)**

All commercial APIs rejected due to data privacy.

**Option B: Self-Hosted Open Model**

**Evaluation Criteria**:
1. Code benchmark performance
2. Hardware feasibility
3. License compatibility (internal use)
4. Vendor support availability

**Model Evaluation**:

| Model | HumanEval | License | Hardware | Verdict |
|-------|-----------|---------|----------|---------|
| DeepSeek-V3 | ~70% | MIT | 6x H100 ($210K) | ✅ Best performance |
| Llama 3.1 405B | ~65% | Community | 3x H100 ($105K) | ✅ Acceptable |
| Llama 3.1 70B | ~62% | Community | 2x A100 ($30K) | ⚠️ Budget option |

**Final Choice**: **DeepSeek-V3**

**Rationale**:
- ✅ Top open-source code performance
- ✅ MIT license (maximum freedom for internal use)
- ✅ Budget allows best solution
- ✅ 6x H100 server within $500K CapEx ($210K hardware + $290K contingency)

**Architecture**:

```
┌─────────────────────────────────────────┐
│ Developer IDE Plugins                   │
│ (VSCode, IntelliJ)                      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ Internal API Gateway                    │
│ (Auth, rate limiting, logging)          │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│ vLLM Server (Load Balanced)             │
│ - Model: DeepSeek-V3 (4-bit quantized)  │
│ - Hardware: 6x H100 (80GB)              │
│ - Deployment: On-premises data center   │
└─────────────────────────────────────────┘
```

**Implementation**:

```bash
# Deploy with vLLM
python -m vllm.entrypoints.openai.api_server \
  --model deepseek-ai/DeepSeek-V3 \
  --dtype float16 \
  --tensor-parallel-size 6 \
  --max-model-len 32768
```

**Monitoring**:
- Prometheus + Grafana
- Custom code quality metrics
- Developer satisfaction surveys

**Results** (6 months post-deployment):
- ✅ 99.95% uptime
- ✅ Average response time: 1.2 seconds
- ✅ Developer satisfaction: 4.3/5
- ✅ Estimated productivity gain: 15% (measured by PR velocity)
- ✅ **ROI**: Payback period ~9 months (vs developer time cost)

**Outcome**:
- Complete data privacy maintained
- Best-in-class code performance
- Strong ROI justifies continued investment
- Considering Llama 4 evaluation for next upgrade

---

## Case Study 3: Academic Research Lab (Multi-Agent Reasoning Experiments)

**Organization Profile**:
- Type: University AI research lab
- Focus: Multi-agent coordination and reasoning
- Budget: $10K/year (grant funded)
- Team: 1 professor, 3 PhD students

**Requirements**:
- Experimentation flexibility
- Ability to inspect model internals
- Minimal costs (limited grant budget)
- Publication and derivative work freedom

**Decision Process**:

**Phase 1: Model Selection**

**Criteria**:
1. Reasoning capability
2. Tool use (function calling)
3. License permits research publication
4. Runnable on lab hardware (2x RTX 3090)

**Options Evaluated**:

| Model | Reasoning | Tool Use | License | Hardware Fit |
|-------|-----------|----------|---------|--------------|
| GPT-5 (API) | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ Proprietary | ✅ API |
| Claude 4 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ Proprietary | ✅ API |
| Llama 3.1 70B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⚠️ Community | ❌ Too large |
| Mixtral 8x7B | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ Apache 2.0 | ✅ 2x RTX 3090 |
| DeepSeek-V3 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ MIT | ❌ Too large |

**Final Choice**: **Mixtral 8x7B** (self-hosted)

**Rationale**:
- ✅ Fits on 2x RTX 3090 (4-bit quantized)
- ✅ Apache 2.0: Can publish modifications, derivatives
- ✅ Strong reasoning and tool use
- ✅ Zero ongoing API costs (critical for $10K budget)

**Phase 2: Infrastructure**

**Hardware** (Already owned by lab):
- 2x RTX 3090 (24GB each)
- Workstation with 128GB RAM

**Software Stack**:
```bash
# Install Ollama (easiest for research iteration)
curl -fsSL https://ollama.com/install.sh | sh

# Pull model
ollama pull mixtral:8x7b-instruct-v0.1-q4_K_M

# Run with API server
ollama serve
```

**Agent Framework**: **LangChain** (MIT license, research-friendly)

```python
from langchain_ollama import ChatOllama
from langchain.agents import AgentExecutor, create_react_agent

llm = ChatOllama(model="mixtral:8x7b-instruct-v0.1-q4_K_M")

# Define tools for multi-agent coordination
tools = [search_tool, calculator_tool, memory_tool]

agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools)

# Run experiments
result = agent_executor.invoke({"input": "Coordinate with other agents to solve..."})
```

**Phase 3: Research Experiments**

**Experiments Conducted**:
1. Multi-agent debate (3 agents reasoning together)
2. Tool use in sequential decision-making
3. Fine-tuning on custom reasoning datasets

**Total API Cost** (if used GPT-5): ~$8,500 over 12 months

**Actual Cost**: $0 (self-hosted)

**Budget Remaining**: $10,000 for other research expenses

**Results**:
- ✅ Published 2 papers (Apache 2.0 license enabled sharing fine-tuned models)
- ✅ Open-sourced custom agent framework (600 GitHub stars)
- ✅ Budget fully allocated to conferences and student stipends
- ✅ Reproducible research (other labs can run exact setup)

**Outcome**:
- Open-source approach enabled high-quality research within budget
- Licensing freedom critical for academic publication
- Local hosting enabled rapid iteration
- Considering Llama 3.1 70B with grant for new GPU cluster
