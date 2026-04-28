# ComfyUI-PromptGenerator — Roadmap 2026 Q2

**Current state (v1.2.0):** 2 nodes (`PromptGeneratorNode`, `StyleApplierNode`), 9 style presets, Ollama streaming with health checks, LoRA model prioritization, CI/CD publish pipeline.

---

## Phase 1 — Foundation Hardening *(v1.3.0, ~2 weeks)*

| # | Task | Rationale |
|---|------|-----------|
| 1.1 | **Extract `ollama_client.py`** — Pull all Ollama logic (health check, streaming, model discovery, subprocess fallback) out of `PromptGeneratorNode` into a dedicated adapter module under `adapters/`. | SOLID single-responsibility; the node class is 744 lines with mixed concerns. |
| 1.2 | **Add unit tests** (`tests/unit/test_extract_final_prompt.py`, `test_style_presets.py`, `test_ollama_client.py`) with mocked Ollama responses. Target 70% branch coverage. | CLAUDE.md requires 70% minimum; currently 0 tests. |
| 1.3 | **Unify style sources** — `style_presets.py` has 9 styles, `templates.yaml` has 8 (missing `still_image` inconsistently), `DEFAULT_STYLES` has 9. Single source of truth via `config/styles.yaml` merging keyword data + Jinja2 templates. | Eliminates drift between three style registries. |
| 1.4 | **`ruff` + `pytest` in CI** — Extend `test.yml` to run `pytest tests/ -v --cov=nodes --cov-fail-under=70`. | Gate merges on actual behavior, not just lint. |

---

## Phase 2 — Prompt Chain Node *(v1.4.0, ~3 weeks)*

| # | Task | Rationale |
|---|------|-----------|
| 2.1 | **`PromptRefinerNode`** — New node: takes raw prompt STRING in, calls Ollama with a refinement system prompt, outputs refined STRING. Configurable passes (1–3). Supports `seed` for deterministic refinement. | Core "AI-driven prompt engineering" workflow: generate → refine → encode. |
| 2.2 | **`NegativePromptNode`** — Generates a negative prompt from the positive prompt + style. Uses a dedicated Jinja2 template with SD/XL-specific negative token lists. | Users currently have to manually write negatives; automating this is high-value. |
| 2.3 | **`PromptCombinerNode`** — Merges 2–4 prompt strings with configurable weighting (`blend`, `concat`, `weighted_average` as text interpolation). | Enables multi-concept compositions without manual string editing. |

**Node graph target:**
```
[PromptGenerator] → [PromptRefiner] → [PromptCombiner] → CLIP Text Encode
                            ↑
[NegativePrompt] ───────────┘
```

---

## Phase 3 — Multi-Backend & Advanced Routing *(v1.5.0, ~3 weeks)*

| # | Task | Rationale |
|---|------|-----------|
| 3.1 | **`LLMBackend` protocol** — Abstract interface (`generate(prompt, options) → str`). Implementations: `OllamaBackend` (existing), `OpenAIBackend` (API key input), `LlamaCppBackend` (local GGUF via `llama-cpp-python`). | Decouple from Ollama-only; users want OpenAI/cloud/local GGUF flexibility. |
| 3.2 | **`PromptRouterNode`** — Routes description to different backends/models based on a strategy: `round_robin`, `quality_first` (larger model), `speed_first` (smaller model), `style_match` (pick model trained on that style). | Multi-model orchestration; matches the LoRA prioritization pattern but at the workflow level. |
| 3.3 | **Backend health dashboard** — Extend `_check_ollama_health` into a reusable `HealthMonitor` that exposes status as a ComfyUI `OUTPUT_NODE` for on-canvas display. | Visibility into backend state without console logs. |

---

## Phase 4 — Prompt Intelligence Layer *(v1.6.0, ~4 weeks)*

| # | Task | Rationale |
|---|------|-----------|
| 4.1 | **`PromptAnalyzerNode`** — Takes a prompt STRING, outputs structured data: detected style, token count, quality score (0–100), suggested improvements as STRING. Uses a lightweight LLM call or rule-based heuristics. | Bidirectional workflow: analyze existing prompts to understand what the generator should produce. |
| 4.2 | **Style embedding cache** — Pre-compute style keyword embeddings (via `sentence-transformers` or Ollama embeddings API). Cache to disk. Enable semantic style matching: "closest style to user's freeform description." | Eliminates manual style dropdown; the system picks the best style from natural language. |
| 4.3 | **`PromptHistoryNode`** — `OUTPUT_NODE` that logs generated prompts to `output/prompt_history.jsonl` with metadata (style, model, timestamp, seed). Supports "load from history" input for reproducibility. | Critical for iterative workflows; users lose prompts when they restart ComfyUI. |
| 4.4 | **Custom style packs** — Allow users to drop `mypack.yaml` into `config/styles/` with a `style_pack` metadata header. Auto-discovered on restart. | Community extensibility without editing core files. |

---

## Phase 5 — Video & Multi-Modal Prompts *(v1.7.0, ~3 weeks)*

| # | Task | Rationale |
|---|------|-----------|
| 5.1 | **Expand `video_wan` style** — Add temporal keywords (camera movement, scene transitions, duration hints). Support WanVideo 2.3-specific prompt format. | The `video_wan` style is currently a minimal stub; video generation is the fastest-growing ComfyUI use case. |
| 5.2 | **`ImageToPromptNode`** — Takes IMAGE input, sends to a vision model (LLaVA via Ollama or GPT-4V), outputs a descriptive prompt STRING. | Image-to-prompt is a top-requested feature; bridges visual reference into the text pipeline. |
| 5.3 | **`PromptBatchNode`** — Generates N prompts from one description by varying style/mood/seed. Outputs `STRING` list for batch generation workflows. | Supports exploration and A/B testing of prompt variations. |

---

## Cross-Cutting Concerns (All Phases)

| Concern | Action |
|---------|--------|
| **Documentation** | Update `README.md` per release; add Mermaid node-graph diagrams for each workflow pattern. |
| **Registry** | Tag `v1.3.0`–`v1.7.0` per phase; `publish.yml` handles deployment automatically. |
| **Error handling** | Every new node wraps `generate()` in the same try/except pattern: `ValueError` for bad inputs, `RuntimeError` for OOM/backend failures, `ProcessingError` for unexpected model output. |
| **Logging** | Replace `print()` calls with `logging.getLogger(__name__)` per AGENTS.md §3.3. |
| **Security** | No API keys in source — OpenAI/backend keys via ComfyUI env or `optional` STRING input with tooltip warning. |

---

## Dependency Graph

```
Phase 1 (foundation) ──→ Phase 2 (chain nodes)
                              │
Phase 3 (multi-backend) ──────┘
         │
Phase 4 (intelligence) ←── requires Phase 3 protocol abstraction
         │
Phase 5 (video/multimodal) ←── requires Phase 4 analyzer for quality scoring
```

Phases 1 and 2 are **blocking** — everything else can be reordered based on user feedback.

---

## Immediate Next Steps

1. **Branch `feature/v1.3.0-foundation`** from `main`
2. Create `nodes/adapters/ollama_client.py` — extract from `prompt_generator_node.py`
3. Add `tests/unit/` with `pytest` fixtures for the three existing modules
4. Run `ruff check . && ruff format . && pytest --cov=nodes --cov-fail-under=70`
5. Tag `v1.3.0`, push, verify CI + auto-publish

---

*Generated 2026-04-27. Review quarterly or after each phase completion.*
