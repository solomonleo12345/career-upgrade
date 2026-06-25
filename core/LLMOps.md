# LLMOps: Complete Interview Guide

## What is LLMOps?

**LLMOps (Large Language Model Operations)** is the practice of managing the full lifecycle of LLM-powered applications in production — similar to MLOps but tailored for the unique challenges of LLMs (non-determinism, hallucinations, prompt management, token costs, latency, safety).

**Why it exists:** Traditional ML pipelines assume deterministic models with clear metrics (accuracy, F1). LLMs are probabilistic, expensive, can hallucinate, and produce free-form text — so you need new tooling for evaluation, monitoring, safety, and cost control.

**Key pillars:**
1. **Evaluation** — how good is the output?
2. **Guardrails** — is the output safe?
3. **Observability** — what's happening in production?
4. **Prompt management, versioning, cost tracking, A/B testing**

---

## 1. RAGAS (Retrieval-Augmented Generation Assessment)

### What is RAG first?
**RAG** = Retrieval-Augmented Generation. Instead of relying purely on the LLM's training data, you:
1. **Retrieve** relevant documents from a vector database (e.g., chunks from PDFs)
2. **Augment** the user's prompt with those documents
3. **Generate** an answer grounded in the retrieved context

### What is RAGAS?
**RAGAS** is an open-source framework (Python library) for evaluating RAG pipelines **without needing human-labeled ground truth** for many of its metrics. It uses an LLM as a judge.

### The 4 Core RAGAS Metrics

Imagine a RAG pipeline with: **Question → Retrieved Contexts → Generated Answer**

| Metric | What it measures | Simple explanation |
|---|---|---|
| **Faithfulness** | Is the answer grounded in the retrieved context? | Did the LLM make stuff up vs. stick to what was retrieved? Detects hallucination. |
| **Answer Relevancy** | Does the answer actually address the question? | If asked "What is the capital of France?" and answer is "France is a country in Europe" — that's irrelevant even if true. |
| **Context Precision** | Are the retrieved chunks relevant and ranked well? | Out of N retrieved chunks, how many were actually useful? Are useful ones ranked first? |
| **Context Recall** | Did retrieval get all the info needed to answer? | Did we miss any important documents that contain the answer? (This one needs a ground-truth answer.) |

### How RAGAS works under the hood
- For **Faithfulness**: It breaks the generated answer into individual claims, then asks an LLM "can each claim be verified from the context?" Score = verifiable claims / total claims.
- For **Answer Relevancy**: It uses the answer to generate possible questions, then measures cosine similarity between those generated questions and the original question.

### Interview-worthy points
- RAGAS is **reference-free** for most metrics (huge advantage — no need to label thousands of examples).
- It's **LLM-as-a-judge** based, so it inherits the judge's biases.
- Combine RAGAS with **TruLens**, **DeepEval**, or **Phoenix (Arize)** for broader coverage.
- Common follow-up: "How do you improve a low faithfulness score?" → Better chunking, re-ranking, prompt engineering ("answer only from context"), lower temperature.

---

## 2. Evals (LLM Evaluation)

### Why evals are hard for LLMs
- Outputs are free-form text (not classes or numbers)
- The same input can produce different outputs (non-deterministic)
- "Correctness" is often subjective
- Traditional metrics (BLEU, ROUGE) measure word overlap, not meaning

### Types of Evals

#### a) **Reference-based evals** (you have a ground truth)
- **Exact Match / F1**: Good for QA with known answers
- **BLEU / ROUGE**: N-gram overlap (used in translation/summarization, but weak for LLMs)
- **BERTScore**: Semantic similarity using embeddings — better than BLEU
- **Semantic similarity**: Cosine similarity of embeddings

#### b) **Reference-free evals** (no ground truth)
- **LLM-as-a-judge**: Use GPT-4 (or similar) to rate the output on criteria like helpfulness, correctness, tone, safety
- **Self-consistency**: Ask the LLM the same question multiple times — do answers agree?
- **Heuristics**: Length, format compliance (valid JSON?), keyword presence

#### c) **Human evals**
- Gold standard but slow and expensive
- Used for benchmarking and to validate LLM-as-a-judge

### The Eval Workflow (Interview Story)
1. **Build a golden dataset** — 50–500 representative examples covering edge cases
2. **Define metrics** — what does "good" mean for your use case? (Accuracy, tone, safety, latency, cost)
3. **Run evals on every prompt/model change** — like unit tests
4. **Track results over time** — catch regressions when you swap models or tweak prompts

### Key concepts to know
- **Offline evals**: Run before deployment, on a fixed dataset
- **Online evals**: Run on live production traffic (sample 1% of requests)
- **A/B testing**: Compare two prompts/models on real users
- **LLM-as-a-judge pitfalls**: Position bias (prefers first answer shown), verbosity bias (prefers longer answers), self-preference bias (GPT-4 prefers GPT-4 outputs)

### Popular eval frameworks
- **OpenAI Evals**, **DeepEval**, **RAGAS**, **TruLens**, **LangSmith**, **Phoenix (Arize)**, **Promptfoo**, **Braintrust**

---

## 3. Guardrails

### What are guardrails?
**Guardrails** are safety/policy checks that sit around your LLM to prevent bad inputs (jailbreaks, PII, prompt injection) and bad outputs (toxicity, hallucinations, off-topic responses, leaked secrets).

Think of them as **firewalls for LLMs**.

### Where guardrails sit
```
User Input → [INPUT GUARDRAILS] → LLM → [OUTPUT GUARDRAILS] → User
```

### Common Input Guardrails
| Guardrail | Purpose |
|---|---|
| **PII detection** | Block/redact emails, SSNs, credit cards before they hit the LLM |
| **Prompt injection detection** | Catch "ignore previous instructions" style attacks |
| **Topic filtering** | Reject off-topic questions (e.g., a banking bot refusing medical questions) |
| **Jailbreak detection** | Block known jailbreak patterns (DAN, role-play exploits) |
| **Token/length limits** | Prevent abuse and runaway costs |

### Common Output Guardrails
| Guardrail | Purpose |
|---|---|
| **Toxicity/profanity filter** | Block harmful, hateful, or NSFW content |
| **PII leakage check** | Make sure the LLM doesn't return sensitive data |
| **Hallucination check** | Verify claims against trusted sources (often via RAG faithfulness) |
| **Format validation** | Ensure JSON is valid, schema matches |
| **Competitor mentions** | E.g., don't recommend Pepsi from a Coke chatbot |
| **Profanity, bias, sentiment** | Brand safety |

### How guardrails are implemented
1. **Rule-based / regex** — Fast, cheap, brittle (good for PII, profanity word lists)
2. **Classifier models** — Small fine-tuned models (e.g., toxicity classifiers like Detoxify)
3. **LLM-based** — Use another LLM call to judge the output (accurate but slow/expensive)
4. **Embedding-based similarity** — Compare output to known unsafe content

### Popular tools
- **NVIDIA NeMo Guardrails** — Programmable rails using a DSL called Colang
- **Guardrails AI** (the open-source library) — Schema-based validation with `.rail` specs
- **Llama Guard** (Meta) — Open-source LLM specifically trained for safety classification
- **Azure AI Content Safety**, **AWS Bedrock Guardrails**, **OpenAI Moderation API**

### Interview gotchas
- **Tradeoff**: Strict guardrails → high false-positive rate → bad UX. Loose guardrails → safety risk.
- **Latency cost**: Each guardrail adds latency. Run cheap ones first (regex), expensive ones last (LLM judge).
- **Parallel vs. serial**: Run independent guardrails in parallel to reduce latency.
- **Fail-open vs. fail-closed**: If a guardrail crashes, do you let the response through or block? Depends on risk tolerance.

---

## 4. Observability

### What is LLM observability?
Knowing **what's happening inside your LLM app in production** — every prompt, response, latency, cost, error, and user feedback — so you can debug, monitor, and improve.

It's **distributed tracing for LLM apps**.

### The 3 Pillars (borrowed from general observability)
1. **Logs** — every input/output, every tool call
2. **Metrics** — latency, cost, token usage, error rate, eval scores
3. **Traces** — the full chain of operations for a single request (retrieval → LLM call → tool call → final answer)

### Why it's uniquely hard for LLMs
- A single user query may trigger **many LLM calls, tool calls, and retrievals** (agentic workflows)
- You need to inspect prompts and outputs to debug — not just stack traces
- You're tracking **semantic quality**, not just up/down status
- **Cost** matters — every token has a dollar sign attached

### What you must track
| Category | Metrics |
|---|---|
| **Performance** | Latency (p50, p95, p99), time-to-first-token (TTFT), tokens/sec |
| **Cost** | Input tokens, output tokens, $ per request, $ per user/feature |
| **Quality** | Eval scores, user feedback (thumbs up/down), hallucination rate |
| **Reliability** | Error rate, timeout rate, guardrail trigger rate |
| **Usage** | Requests per user, most common prompts, drift in input distribution |

### Tracing in agentic systems
A single user message might span:
```
User Query
  → Embedding generation
  → Vector DB retrieval
  → LLM call 1 (decide tool)
  → Tool call (API)
  → LLM call 2 (synthesize answer)
  → Guardrail check
  → Final response
```
You need a **trace** that links all these spans together (think OpenTelemetry-style).

### Popular observability tools
- **LangSmith** (LangChain) — tracing, evals, monitoring
- **Langfuse** — open-source LLM observability
- **Arize Phoenix** — open-source, focused on RAG/agent debugging
- **Helicone** — proxy-based logging
- **Weights & Biases (W&B Weave)**, **Datadog LLM Observability**, **New Relic AI Monitoring**

### Interview-worthy concepts
- **Drift detection**: Has the distribution of user questions changed over time? Are eval scores dropping?
- **Feedback loops**: Capture user thumbs-down → feed into eval dataset → improve prompts/model
- **Replay**: Re-run past traces against a new prompt/model to see if it would have done better
- **PII-safe logging**: Redact sensitive data before logging — observability shouldn't create a data leak

---

## How They Fit Together (Likely Interview Question)

```
                    ┌─────────────────────────────────────────┐
                    │           LLM Application               │
User Request ───▶ Guardrails (in) ─▶ RAG/LLM ─▶ Guardrails (out) ─▶ Response
                    │         │                          │       │
                    │         └──────────┬───────────────┘       │
                    │                    ▼                       │
                    │             Observability                  │
                    │     (traces, logs, costs, metrics)         │
                    │                    │                       │
                    │                    ▼                       │
                    │              Evals + RAGAS                 │
                    │      (offline + online + feedback)         │
                    └─────────────────────────────────────────┘
```

- **Guardrails** keep it safe in real time
- **Observability** shows you what's happening
- **Evals/RAGAS** tell you if it's working well
- **LLMOps** is the discipline that ties them all together with CI/CD, prompt versioning, model versioning, and cost controls

---

## Common Interview Questions

1. **"How would you evaluate a RAG chatbot?"** → Use RAGAS metrics (faithfulness, answer relevancy, context precision/recall), build a golden dataset, run offline evals on every change, sample online traffic for live evals, collect user feedback.

2. **"Your chatbot is hallucinating. How do you debug?"** → Check observability traces. Look at retrieved context — was it relevant? Check faithfulness score. Try better chunking, re-ranking, lower temperature, stricter prompt ("answer only from context, say 'I don't know' otherwise"), add output guardrail to verify claims.

3. **"How do you prevent prompt injection?"** → Input guardrails (classifier, regex for known patterns), separate system prompt from user input clearly, use structured outputs, don't put untrusted data in system role, validate outputs (output guardrails), least-privilege tool access.

4. **"How do you monitor LLM costs?"** → Track tokens per request, tag requests by user/feature/model, set per-user rate limits, use cheaper models for simple tasks (routing/cascading), cache responses, monitor cost-per-conversion metrics.

5. **"What's LLM-as-a-judge? What are its limitations?"** → Using a strong LLM to score outputs from another LLM. Limits: position bias, verbosity bias, self-preference bias, expensive, inherits judge's mistakes. Mitigate with: pairwise comparison, randomize order, use multiple judges, validate against human labels.

6. **"What's the difference between MLOps and LLMOps?"** → MLOps focuses on training, retraining, model versioning, feature stores. LLMOps focuses on prompts (often you don't train), API/cost management, evals for free-form text, guardrails for safety, RAG pipeline management. LLMOps inherits a lot from MLOps but adds prompt-centric workflows.

---

**Quick mental model to remember:**
> **RAGAS** evaluates RAG. **Evals** evaluate everything. **Guardrails** enforce safety. **Observability** shows you reality. **LLMOps** is the umbrella practice.
