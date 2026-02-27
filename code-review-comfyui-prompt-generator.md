# Code Review: ComfyUI-PromptGenerator

## Executive Summary

The codebase is well-structured with good error handling and graceful degradation patterns. However, there is a **critical inconsistency** in the style system between the two nodes, plus several architectural concerns worth addressing.

---

## 🚨 Critical Issues

### 1. Style System Inconsistency (High Priority)

**Problem:** The two nodes have completely different style options:

| Node | Available Styles |
|------|------------------|
| `PromptGeneratorNode` | cinematic, anime, photorealistic, fantasy, abstract, cyberpunk, sci-fi, video_wan (8 styles) |
| `StyleApplierNode` | cinematic, still_image (2 styles via `StylePreset.get_style_choices()`) |

The `style_presets.py` module only defines 2 styles, while `config/templates.yaml` and `DEFAULT_STYLES` define 8 styles.

**Impact:** Users expect consistency. If they select "anime" style in PromptGenerator, they cannot use StyleApplier with "anime" - it will fail validation.

**Recommendation:** Either:

- Expand `STYLE_DEFINITIONS` in `style_presets.py` to include all 8 styles
- Or consolidate to use templates.yaml as the single source of truth

---

## ⚠️ Architectural Concerns

### 2. Triple Source of Truth for Styles

The codebase has three different places defining styles:

1. `config/templates.yaml` - YAML templates
2. `DEFAULT_STYLES` dict in Python (`prompt_generator_node.py`)
3. `style_presets.py` - StyleKeywords dataclass

**Recommendation:** Consolidate to a single source. The YAML file is user-editable and should be the authoritative source.

### 3. Unused `style_presets.py` Module

The `StylePreset` class is only used by `StyleApplierNode`. The main `PromptGeneratorNode` uses its own `DEFAULT_STYLES` dict and YAML templates.

---

## 🔧 Code Quality Issues

### 4. Missing Style Validation in `generate()`

In `generate()`, there's no validation that the selected style exists - it silently falls back to cinematic.

### 5. Inconsistent Error Messages

- `PromptGeneratorNode` returns emoji-prefixed messages: `"⚠️ Please enter an image description."`
- `StyleApplierNode` returns bracket-prefixed: `"[StyleApplier] Error: Unknown style..."`

**Recommendation:** Standardize error message format.

### 6. Hardcoded Values

- `chunk_timeout = 30` is hardcoded in streaming
- Model list in `_get_available_models()` has hardcoded defaults

---

## ✅ Positive Patterns

1. **Graceful Degradation:** Excellent use of optional imports with fallbacks
2. **Streaming Implementation:** Well-designed background thread approach for timeout enforcement
3. **Cold Start Detection:** Smart 1.3x timeout multiplier for unloaded models
4. **Model Caching:** 60-second cache prevents UI freezes
5. **LoRA Prioritization:** Good sorting logic to surface fine-tuned models first

---

## 📋 Recommendations Priority List

| Priority | Issue | Action |
|----------|-------|--------|
| P0 | Style inconsistency | Sync `style_presets.py` with 8 styles or consolidate |
| P1 | Triple source of truth | Make templates.yaml the single source |
| P2 | Silent style fallback | Add validation/warning when style not found |
| P3 | Error message format | Standardize across nodes |
| P4 | Hardcoded values | Move to config/constants |
