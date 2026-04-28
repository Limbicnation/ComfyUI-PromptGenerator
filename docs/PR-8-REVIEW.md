# PR #8 Review: Prompt Chain Nodes + Test Infrastructure

**Date:** 2026-04-28
**PR:** https://github.com/Limbicnation/ComfyUI-PromptGenerator/pull/8
**Branch:** `pr-7` → `main`
**Commits:** `89c8341` (feat), `8f2eca4` (fix: address code review issues)

## Summary

This PR extracts an `OllamaClient` adapter from `PromptGeneratorNode`, adds three new chain nodes (Refiner, Negative, Combiner), adds 29 unit tests, and centralizes style config. Two commits: feature + review fix.

## Issues Found

### 1. Bug — `_weighted_average` deduplication neutralizes weights (Medium)

**File:** `nodes/prompt_combiner_node.py:207-218`

The method repeats prompts proportionally to weight, then deduplicates them — which completely negates the repetition. With prompt A (weight 2.0) and prompt B (weight 1.0), A repeats 3x and B repeats 1x, but after dedup you get just `[A, B]` — identical to concat. The weights have zero effect.

```python
# Current (broken):
repeats = max(1, int((weight / total_weight) * 3))  # A→3, B→1
# After dedup: [A, B] — weights lost
```

**Severity:** Medium — Users selecting `weighted_average` mode will get functionally identical output to `concat`.

### 2. Silent exception swallowing in `generate_streaming` (Low)

**File:** `nodes/adapters/ollama_client.py:225-227`

The bare `except Exception` catches everything (including `TypeError`, `AttributeError`) and silently returns `None`. This hides bugs — if the ollama API changes or a coding error occurs, the caller gets `None` with only a log line, making debugging difficult.

**Severity:** Low — Pattern carried over from original code, but adapter extraction was an opportunity to improve it.

### 3. `print()` instead of `logging` (Low)

**File:** `nodes/adapters/ollama_client.py:56` and all new nodes

Per AGENTS.md: "`print()` in production code — use logging." All new nodes use `OllamaClient._log()` which wraps `print()`. The original `PromptGeneratorNode` also used `print()`, so this is a pre-existing pattern, but the new code perpetuates it.

**Severity:** Low — Pre-existing pattern, flagged since AGENTS.md explicitly forbids it.

### 4. `PromptRefinerNode` doesn't expose `top_p` (Low)

**File:** `nodes/prompt_refiner_node.py:117`

`top_p` is hardcoded to `0.9` in the `refine()` method. Other nodes (`NegativePromptNode`, `PromptGeneratorNode`) expose it as a user-configurable input. Minor inconsistency.

**Severity:** Low — Cosmetic inconsistency.

## What Looks Good

- Clean adapter extraction (`OllamaClient`) with proper thread-safe instance-level caching
- The `importlib.util.find_spec` pattern for optional imports is correct
- Tests cover critical paths: model discovery, health checks, subprocess fallback, prompt extraction
- All 29 tests pass (per commit message)
- `__init__.py` properly registers all new nodes with correct mappings

## Fixes Applied (2026-04-28)

All 4 issues have been patched in the source files:

1. **`_weighted_average`** — Replaced broken repeat+dedup logic with weight-ratio emphasis markers (ComfyUI `((...))` / `[...]` syntax). Weights now actually affect output.
2. **Silent exceptions** — `ollama_client.py` now catches specific exceptions (`ConnectionError`, `TimeoutError`, `TypeError`, `AttributeError`) and propagates API contract mismatches as `RuntimeError` with full chain. Bare `except Exception` eliminated.
3. **`print()` → `logging`** — All new nodes (`OllamaClient`, `PromptRefinerNode`, `NegativePromptNode`) now use `logging.getLogger(__name__)`. `PromptGeneratorNode` pre-existing prints left untouched.
4. **`top_p` exposed** — `PromptRefinerNode.INPUT_TYPES()` now includes `top_p` slider (0.1–1.0, default 0.9), matching `NegativePromptNode` and `PromptGeneratorNode`.

## Verdict

All issues resolved. Ready for merge.

**Recommendation:** Merge after CI passes. Consider adding a unit test for `_weighted_average` that asserts `weighted_average(A:2, B:1) != concat(A, B)` to prevent regression.
