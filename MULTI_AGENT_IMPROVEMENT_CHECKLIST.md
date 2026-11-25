# Keigo Master: Track A Multi-Agent System Improvement Plan & Implementation Checklist

**Document Type:** Technical & Empirical Research Specification  
**Project Title:** Keigo Master: A LangGraph-Orchestrated Multi-Agent Architecture for Japanese Formality Conversion  
**Target Paper:** Formal NLP Conference Paper (EMNLP / ACL Style)  
**Track:** Track A (Proposed Multi-Agent System on Live Runtime)  
**Primary Tech Stack:** LangGraph, gpt-oss-120b (`openai/gpt-oss-120b` via Groq LPU; replaces Meta Llama 3.3 70B, which Groq no longer offers), Cohere v3 Embeddings (`embed-multilingual-v3.0`), Vector DB (Pinecone / Supabase)

---

## 1. Research Context & Strict Empirical Safeguards

1. **Dual-Track Comparative Objective:**
   - **Track A (Proposed):** LangGraph multi-agent architecture with dynamic few-shot exemplar retrieval and critique loops.
   - **Track B (Baseline Control):** Supervised Fine-Tuning (SFT) of `sonoisa/t5-base-japanese` using Hugging Face `Seq2SeqTrainer` on Google Colab T4 GPUs.
2. **Corpus & Split Integrity (Zero Test Leakage):**
   - Total Corpus: 923 parallel pairs in `.jsonl` format.
   - Partitioning: 80% `train.jsonl`, 10% `val.jsonl`, 10% `test.jsonl`.
   - **Rule:** `test.jsonl` is **strictly frozen**. It must never be indexed into the RAG vector store or referenced in few-shot prompt construction.
   - **Rule:** Unicode NFKC normalization (`unicodedata.normalize('NFKC', text)`) must be applied across all inputs and outputs.
3. **Linguistic Scope & Target Honorifics:**
   - Focus: Casual vernacular (*Tameguchi*) $\rightarrow$ Polished formal business Keigo.
   - Target Registers: Teineigo (~40%), Sonkeigo (~30%), Kenjougo (~30%).
   - Explicit rejection of double honorifics (*Nijuu Keigo*) and archaic phrasing.

---

## 2. Architectural Blueprint: Upgraded Track A Workflow

```
                               ┌───────────────────────────┐
                               │        START NODE         │
                               └─────────────┬─────────────┘
                                             │
                      ┌──────────────────────┴──────────────────────┐
                      ▼                                             ▼
         [Node: Language Analyzer]                      [Node: Context Retriever]
         - Hierarchy & Speaker Relationship             - Cohere v3 Top-k Embeddings
         - Tone & Workplace Domain Context              - Suppletive Verb Matching
                      │                                             │
                      └──────────────────────┬──────────────────────┘
                                             ▼
                                  [Node: Keigo Specialist]
                                  - Synthesizes Analyzer + RAG
                                  - Candidate Output Generation
                                             │
                                             ▼
                                 [Node: Reviewer Agent]
                                  - Deterministic Regex Filters
                                  - LLM Appropriateness Critique
                                             │
                        ┌────────────────────┴────────────────────┐
                   [Approved]                       [Rejected & Iteration < 3]
                        ▼                                         ▼
                     ( END )                        ( Route back to Specialist )
                        ▲
                        │ [Iteration >= 3]
         [Node: Conservative Fallback]
```

---

## 3. Concrete Implementation Checklist

### Phase 1: Structured Knowledge Base & RAG Enhancement

- [x] **1.1 Compile Irregular / Suppletive Verb Lexicon (`data/knowledge_base/suppletive_verbs.json`)**
  - Implement a deterministic dictionary for Japanese lexical suppletion (補充法) to prevent hallucinated regularized honorifics:
    ```json
    {
      "verbs": {
        "行く": {
          "sonkeigo": ["いらっしゃる", "おいでになる"],
          "kenjougo": ["参る", "伺う"]
        },
        "来る": {
          "sonkeigo": ["いらっしゃる", "おいでになる"],
          "kenjougo": ["参る"]
        },
        "いる": {
          "sonkeigo": ["いらっしゃる", "おいでになる"],
          "kenjougo": ["おる"]
        },
        "言う": {
          "sonkeigo": ["おっしゃる"],
          "kenjougo": ["申す", "申し上げる"]
        },
        "する": {
          "sonkeigo": ["なさる", "される"],
          "kenjougo": ["いたす"]
        },
        "食べる": {
          "sonkeigo": ["召し上がる"],
          "kenjougo": ["いただく"]
        },
        "飲む": {
          "sonkeigo": ["召し上がる"],
          "kenjougo": ["いただく"]
        },
        "見る": {
          "sonkeigo": ["ご覧になる"],
          "kenjougo": ["拝見する"]
        },
        "知る": {
          "sonkeigo": ["ご存知"],
          "kenjougo": ["存じる", "存じ上げる"]
        },
        "会う": {
          "sonkeigo": [],
          "kenjougo": ["お目にかかる"]
        },
        "聞く": {
          "sonkeigo": [],
          "kenjougo": ["伺う", "拝聴する"]
        },
        "尋ねる": {
          "sonkeigo": [],
          "kenjougo": ["伺う"]
        },
        "訪問する": {
          "sonkeigo": [],
          "kenjougo": ["伺う", "お邪魔する"]
        }
      }
    }
    ```
  - Implement a rule-based matching helper `get_suppletive_rules(text: str) -> List[str]` that extracts matched canonical constraints.

- [x] **1.2 Construct Workplace Domain Register Templates (`data/knowledge_base/domain_templates.json`)**
  - Standardize structural expectations across the 4 dataset workplace domains:
    - **Email/Written:** Formal opening (`いつも大変お世話になっております。`), concluding salutation (`何卒よろしくお願い申し上げます。`), subject line framing.
    - **Verbal/Meetings:** Spoken conversational softeners (`恐れ入りますが`, `あいにくですが`, `差し支えなければ`).
    - **Internal Chat (Slack):** Concise polite workplace tone, avoiding excessive archaic filler while maintaining appropriate distance.
    - **Customer Support:** Empathetic Kenjougo/Sonkeigo blends, structured acknowledgment and apologies.

- [x] **1.3 Refactor Vector Ingestion & Dynamic Retrieval (`src/rag/retriever.py`)**
  - Ingest **only** `train.jsonl` into the vector store using Cohere v3 embeddings (`embed-multilingual-v3.0`).
  - Implement top-$k$ dynamic retrieval ($k=2$ or $3$) filtering by domain where applicable.
  - Structure output exemplars into clean few-shot formatting:
    `Input (Casual): ... -> Output (Formal Keigo): ...`

---

### Phase 2: LangGraph State Schema & Graph Parallelism

- [x] **2.1 Define Strongly-Typed `AgentState` via `TypedDict` (`src/agents/state.py`)**
  - Track hierarchy metadata, retrieval contexts, candidate history, and revision counts:
    ```python
    from typing import TypedDict, List, Dict, Optional, Annotated, Any
    import operator

    class AgentState(TypedDict):
        # Input definition
        casual_input: str
        target_domain: Optional[str]                   # "Email", "Verbal", "Chat", "Customer Support"
        
        # Parallel extraction contexts
        hierarchy_metadata: Dict[str, str]            # speaker_role, listener_role, social_distance
        retrieved_exemplars: List[Dict[str, str]]      # Dynamically retrieved train.jsonl pairs
        suppletive_rules: List[str]                   # Matched irregular verb constraints
        
        # Generation, Critique & Routing
        current_candidate: str
        critique_history: Annotated[List[str], operator.add]
        iteration_count: int
        is_approved: bool
        rejection_reason: Optional[str]
        final_output: Optional[str]
        
        # Empirical research logging
        execution_trace: List[Dict[str, Any]]
    ```

- [x] **2.2 Implement Parallel Execution Branches (`src/agents/graph.py`)**
  - Fan-out from `START` to both `language_analyzer` and `context_retriever` concurrently:
    ```python
    from langgraph.graph import StateGraph, START, END

    builder = StateGraph(AgentState)

    # Register nodes
    builder.add_node("language_analyzer", language_analyzer_node)
    builder.add_node("context_retriever", context_retriever_node)
    builder.add_node("keigo_specialist", keigo_specialist_node)
    builder.add_node("reviewer", reviewer_node)
    builder.add_node("conservative_fallback", conservative_fallback_node)

    # Parallel fan-out
    builder.add_edge(START, "language_analyzer")
    builder.add_edge(START, "context_retriever")

    # Fan-in synchronization into Keigo Specialist
    builder.add_edge(["language_analyzer", "context_retriever"], "keigo_specialist")
    builder.add_edge("keigo_specialist", "reviewer")
    ```

- [x] **2.3 Integrate gpt-oss-120b via Groq LPU**
  - Configure `ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)` across all active agent nodes (set in `src/agent/config.py`). Originally Meta Llama 3.3 70B (`llama-3.3-70b-versatile`), switched after Groq stopped offering it.
  - Enforce system prompts requiring NFKC Unicode output and strict adherence to provided suppletive constraints.

---

### Phase 3: Linguistic Guardrails, Critique Loop & Fallback Control

- [x] **3.1 Deterministic Nijuu Keigo (二重敬語) Validator (`src/guardrails/honorific_rules.py`)**
  - Implement pre-review regex checks targeting classic double honorific combinations:
    ```python
    import re
    from typing import Tuple, List

    NIJUU_KEIGO_PATTERNS = [
        # Double honorific: お/ご + stem + になられる
        (r"(お|ご)[^ぁ-んァ-ヶー一-龠\s]+になられ(る|ます|た|ない)", "お〜になられる (Double Honorific: use お〜になる)"),
        # Irregular verb double honorifics
        (r"おっしゃられ(る|ます|た|ない)", "おっしゃられる (Double Honorific on 言う: use おっしゃる)"),
        (r"ご覧になられ(る|ます|た|ない)", "ご覧になられる (Double Honorific on 見る: use ご覧になる)"),
        (r"お召し上がりになられ(る|ます|た|ない)", "お召し上がりになられる (Double Honorific on 食べる: use 召し上がる)"),
        (r"いらっしゃられ(る|ます|た|ない)", "いらっしゃられる (Double Honorific on 行く/来る/いる: use いらっしゃる)"),
        # Redundant Kenjougo combinations
        (r"お伺いいた(す|します|した)", "お伺いいたす (Redundant Kenjougo: use 伺います)"),
        (r"ご案内いた(す|します|した)", "ご案内いたす (Potential redundancy: verify in context)")
    ]

    def audit_nijuu_keigo(text: str) -> Tuple[bool, List[str]]:
        violations = []
        for pattern, desc in NIJUU_KEIGO_PATTERNS:
            if re.search(pattern, text):
                violations.append(desc)
        return len(violations) > 0, violations
    ```

- [x] **3.2 Implement `ReviewerNode` with Combined Rule + LLM Logic (`src/agents/nodes/reviewer.py`)**
  - Execute deterministic regex audit first. If violations occur, reject immediately with precise feedback without wasting Groq LLM tokens.
  - If regex passes, invoke gpt-oss-120b on Groq with evaluation criteria:
    1. **Honorific Directionality:** Verify speaker actions use Kenjougo (never Sonkeigo) and listener actions use Sonkeigo (never Kenjougo).
    2. **Semantic Equivalence:** Verify propositional facts are neither omitted nor hallucinated.
    3. **Tone Appropriateness:** Verify register matches target workplace domain.
  - State updates:
    - Set `is_approved = True` if all criteria are satisfied.
    - Set `is_approved = False`, increment `iteration_count`, and append explicit remediation guidance to `critique_history`.

- [x] **3.3 Configure Bounded Conditional Routing & Conservative Fallback (`src/agents/graph.py`)**
  - Implement the routing function capping critique cycles at 3 iterations:
    ```python
    def review_decision(state: AgentState) -> str:
        if state.get("is_approved", False):
            return "accept"
        if state.get("iteration_count", 0) >= 3:
            return "fallback"
        return "retry"

    builder.add_conditional_edges(
        "reviewer",
        review_decision,
        {
            "accept": END,
            "retry": "keigo_specialist",
            "fallback": "conservative_fallback"
        }
    )
    builder.add_edge("conservative_fallback", END)
    ```
  - **`conservative_fallback_node` Implementation:**
    - If iterations reach 3 without approval, select the candidate with the fewest structural violations or apply a deterministic, robust Teineigo conversion (`〜です / 〜ます`) to guarantee execution convergence without pipeline failure.

---

### Phase 4: Verification, Instrumentation & Paper Metrics Logging

- [x] **4.1 Test on Edge-Case Test Suite (`tests/test_multi_agent_upgrades.py`)**
  - Verify system performance on 5 canonical linguistic edge cases:
    - **Test 1 (Suppletion):** `明日、先生のオフィスに行く` $\rightarrow$ Must produce `伺う` / `参る`, never `*お行きになる`.
    - **Test 2 (Honorific Directionality):** `社長が書類を見た` $\rightarrow$ Must produce `ご覧になった`, never `拝見した`.
    - **Test 3 (Nijuu Keigo Detection):** Candidate `先生がお書きになられた` $\rightarrow$ Must be rejected and corrected to `お書きになった`.
    - **Test 4 (Domain Salutation):** Input with domain `Email` $\rightarrow$ Must include appropriate opening/closing business salutations.
    - **Test 5 (Loop Bound):** Adversarial / ambiguous input $\rightarrow$ Must terminate within $\le 3$ iterations and output valid text.

- [x] **4.2 Instrument Evaluation Logging for Empirical Section (`src/evaluation/logger.py`)**
  - Log execution telemetry per sample during benchmark inference over `test.jsonl`:
    ```json
    {
      "sample_id": "test_042",
      "casual_input": "...",
      "final_output": "...",
      "iterations_to_converge": 2,
      "fallback_triggered": false,
      "rejection_reasons": ["Detected Nijuu Keigo: おっしゃられる"],
      "latency_seconds": 1.48
    }
    ```
  - These logs directly support **Section 4 (Results)** and **Section 5 (Ablation & Error Analysis)** in the EMNLP/ACL paper.

---

## 4. Execution & Acceptance Criteria

| Component | Target Requirement | Verification Method |
| :--- | :--- | :--- |
| **Suppletive Verbs** | 0% hallucinated regularized honorifics (e.g., `*お行きになる`) on irregular verbs | Targeted unit tests across 13 core irregular verb roots |
| **Nijuu Keigo** | 0 instances of detected double honorifics in final outputs | Regex validation run over all generated benchmark predictions |
| **Graph Parallelism** | Analyzer and Retriever nodes executed concurrently | LangGraph event logs confirm simultaneous dispatch |
| **Convergence Rate** | 100% of test samples produce a completed response within $\le 3$ cycles | Zero unhandled exceptions or infinite loops on `test.jsonl` |
| **Test Set Isolation** | `test.jsonl` unindexed and unseen prior to final evaluation | Checksum and vector DB metadata audit |
