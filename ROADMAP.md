# ComfyUI-PromptGenerator — Project Roadmap

> Compiled: 2026-04-23
> Jira: [CD-26](https://limbicnation.atlassian.net/browse/CD-26)
> Repo: [Limbicnation/ComfyUI-PromptGenerator](https://github.com/Limbicnation/ComfyUI-PromptGenerator)

---

## Current State

| | Local (ComfyUI) | GitHub (main) |
|---|---|---|
| **Version** | v1.1.5 | v1.2.0 |
| **Style loading** | Hardcoded list in `INPUT_TYPES()` | Dynamic via `_get_style_list()` + `templates.yaml` |
| **Style count** | 9 | 9 (source of truth: `config/templates.yaml`) |
| **Fallback behavior** | Crashes on unknown style | Graceful fallback to `cinematic` |
| **Chunk timeout** | Hardcoded `30` | `CHUNK_TIMEOUT = 30` class constant |
| **UI Preview** | None (text output only) | None |
| **Model support** | Qwen3-8B via Ollama (any Ollama model selectable) | Same + LoRA prioritization |
| **Deployment sync** | Manual `git pull` | GitHub Actions auto-publish on tag |

**Immediate action**: Sync local `custom_nodes/comfyui-prompt-generator/` to GitHub main (v1.2.0).

---

## Phase 1 — Stability & Alignment

### 1.1 Sync Local → GitHub
Pull `main` into local ComfyUI `custom_nodes/comfyui-prompt-generator/` to close the v1.1.5 → v1.2.0 gap.

### 1.2 Fix Error Prefix Standardization
Local version uses `⚠️` emoji prefixes in error messages; GitHub uses `[PromptGenerator]` brackets. Standardize all error returns to the bracketed form for log parsability.

### 1.3 Verify Ollama Streaming
Confirm `_generate_with_streaming()` with per-chunk timeouts works reliably. Test cold-start behavior with the 1.3x multiplier.

---

## Phase 2 — UI Preview Window

### 2.1 Prompt Preview Node
Create a companion `PromptPreviewNode` that receives the raw string output from `PromptGeneratorNode` and renders it in a ComfyUI preview panel.

**Implementation options**:
- Use ` comfy.sdxl_utils` / `nodesampler` preview widget if available
- Use `comfy.model_management` callback hooks to display text in the side panel
- Implement a dedicated `PrimitiveNode` wrapper that renders text as a preview widget

**Target behavior**:
- Shows formatted prompt with style label and parameters used
- Collapsible raw reasoning section (respects `include_reasoning`)
- Copy-to-clipboard button
- Character/word count display

### 2.2 Inline Preview in PromptGeneratorNode
Optionally embed a live preview pane directly in the node using `NodeGraph` UI extensions (ComfyUI 1.x supports custom widgets).

### 2.3 Workflow-Level Preview
A dedicated `PromptPreview` node that accepts `(prompt, style, model, parameters)` and shows a styled output card in the canvas.

---

## Phase 3 — Model-Agnostic Parameters

The node should expose parameters that map correctly to both **Flux** and **Qwen** model families.

### 3.1 Flux-Specific Parameters

| Parameter | ComfyUI Node Input | Notes |
|---|---|---|
| `flux_guidance` | `FluxGuidance` node or `flux_guidance` widget | Default: 3.5, range 1.0–7.0 |
| `steps` | `steps` widget | Flux typically uses 20–50 steps |
| `scheduler` | `scheduler` dropdown | `normal`, `simple`, `ddpm`, `sgm_uniform` |
| `cfg_scale` | KSampler `cfg` | Set to `1.0`; `FluxGuidance` handles actual guidance |
| `model` | `UNETLoader` → `FluxLoader` | FLUX.1 Dev/Schnell via `flux2` node |
| `clip` | `CLIPLoader` → `text_encoder_qwen3_8b_fp8mixed` | Qwen3 8B FP8 for FLUX.2 |
| `vae` | `VAELoader` → `vae_flux2_default` | 512-channel latent space (vs SD's 64-channel) |

### 3.2 Qwen-Specific Parameters

| Parameter | ComfyUI Node Input | Notes |
|---|---|---|
| `temperature` | `temperature` widget (0.1–1.0) | Default: 0.7 |
| `top_p` | `top_p` widget (0.1–1.0) | Default: 0.9 |
| `max_tokens` | `max_tokens` widget (50–4096) | Control generation length |
| `seed` | `seed` widget | Reproducibility |
| `include_reasoning` | `BOOLEAN` | Show/hide Qwen3 thinking blocks |
| `timeout` | `timeout` widget (30–600s) | Per-chunk: 30s; total: user-configurable |

### 3.3 Unified Parameter Mapping

Create a `ModelParamMapper` utility:

```python
class ModelParamMapper:
    @staticmethod
    def get_flux_params(prompt: str, **kwargs) -> dict:
        return {
            "flux_guidance": kwargs.get("flux_guidance", 3.5),
            "steps": kwargs.get("steps", 28),
            "scheduler": kwargs.get("scheduler", "normal"),
        }

    @staticmethod
    def get_qwen_params(prompt: str, **kwargs) -> dict:
        return {
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
            "max_tokens": kwargs.get("max_tokens", 512),
            "timeout": kwargs.get("timeout", 120),
        }
```

### 3.4 FLUX.1 vs FLUX.2 Compatibility Notes

> **Critical**: FLUX.1 ControlNets and LoRAs are **NOT compatible** with FLUX.2.
> - `flux-depth-controlnet-v3` → FLUX.1 only
> - `FLUX-dev-lora-add_details` → FLUX.1 only
> - Use plain `euler` sampler (not `euler_cfg_pp`) with FLUX.2 + ControlNet residue
> - `DepthEstimationNode` requires `apply_auto_contrast=true, apply_gamma=true` for correct grey output
> - FLUX.2 uses 512-channel latents; `EmptySD3LatentImage` will crash — use `EmptyFlux2LatentImage`

---

## Phase 4 — Deployment Synchronization

### 4.1 Local → ComfyUI Sync Script

```bash
#!/bin/bash
# sync_prompt_generator.sh
# Sync ComfyUI-PromptGenerator from GitHub to local ComfyUI custom_nodes/

set -e

REPO_DIR="${HOME}/GitHub/ComfyUI-PromptGenerator"
COMFY_NODES_DIR="${HOME}/GitHub/ComfyUI/custom_nodes/comfyui-prompt-generator"
BACKUP_DIR="${HOME}/GitHub/ComfyUI/custom_nodes/comfyui-prompt-generator.backup"

cd "$REPO_DIR"
git fetch origin main
LOCAL_HASH=$(git rev-parse HEAD)
REMOTE_HASH=$(git rev-parse origin/main)

if [ "$LOCAL_HASH" = "$REMOTE_HASH" ]; then
    echo "Already up to date (${LOCAL_HASH:0:8})"
    exit 0
fi

echo "Local: ${LOCAL_HASH:0:8} → Remote: ${REMOTE_HASH:0:8}"
read -p "Pull latest and sync to ComfyUI? [y/N] " confirm
[[ "$confirm" != "y" ]] && exit 1

# Backup current
[ -d "$COMFY_NODES_DIR" ] && mv "$COMFY_NODES_DIR" "$BACKUP_DIR.$(date +%Y%m%d%H%M%S)"

# Pull
git pull origin main

# Sync to ComfyUI
rsync -av --delete \
    --exclude '.git' \
    --exclude '.github' \
    --exclude 'agents' \
    --exclude 'workflow' \
    --exclude 'images' \
    --exclude 'CLAUDE.md' \
    --exclude 'ROADMAP.md' \
    "$REPO_DIR/" "$COMFY_NODES_DIR/"

# Restart ComfyUI if running
if curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; then
    echo "Restarting ComfyUI..."
    # Option A: manager restart (if available)
    # Option B: kill + relaunch
    pkill -f "python.*main.py.*ComfyUI" || true
    sleep 2
    cd "${HOME}/GitHub/ComfyUI"
    nohup python main.py --listen 127.0.0.1 --port 8188 > /tmp/comfyui.log 2>&1 &
    sleep 5
    curl -s http://127.0.0.1:8188/system_stats && echo "ComfyUI restarted OK"
fi

echo "Sync complete."
```

### 4.2 GitHub Actions CI/CD Enhancement

Extend `.github/workflows/publish.yml` to:
1. Run `comfy node validate` on every PR
2. Run integration test against local Ollama (via self-hosted runner)
3. Auto-tag on version bump in `pyproject.toml`
4. Notify ComfyUI Manager of new registry release

### 4.3 Version Alignment Checklist

| Version | Local Status | GitHub Status | Action |
|---|---|---|---|
| v1.1.5 | Installed | Behind | Sync to v1.2.0 |
| v1.2.0 | — | Latest main | Deploy to registry |
| v1.2.x | — | Planned | See phases above |

---

## Phase 5 — Extended Style System

### 5.1 Style Categories
Maintain 9 styles as single source of truth in `config/templates.yaml`:
`cinematic`, `anime`, `photorealistic`, `fantasy`, `abstract`, `cyberpunk`, `sci-fi`, `video_wan`, `still_image`

### 5.2 Style Parameter Overrides
Allow per-style temperature/top_p defaults via `templates.yaml`:

```yaml
cinematic:
  temperature: 0.8
  top_p: 0.85
  template: |
    ...
```

### 5.3 User-Defined Styles
Support loading custom styles from `${COMFYUI_DIR}/user/prompt_styles.yaml` as an override layer on top of `templates.yaml`.

---

## Phase 6 — Quality & Observability

### 6.1 Logging
Add structured logging via Python's `logging` module instead of `print()` statements:

```python
import logging
logger = logging.getLogger("PromptGenerator")
logger.setLevel(logging.DEBUG)
```

### 6.2 Generation Metadata
Return structured output alongside prompt:

```python
RETURN_TYPES = ("STRING", "DICT")
RETURN_NAMES = ("prompt", "metadata")
# metadata = {"style": "...", "model": "...", "tokens_used": N, "latency_ms": T, "template": "..."}
```

### 6.3 Prompt Quality Scoring
Optionally run a lightweight prompt quality check (length, bracket balance, style keyword presence) before returning.

---

## Priority Summary

| Priority | Task | Phase |
|---|---|---|
| P0 | Sync local → GitHub v1.2.0 | 1 |
| P0 | Fix Jira API token | infra |
| P1 | Ollama streaming reliability test | 1 |
| P1 | Deployment sync script (`sync_prompt_generator.sh`) | 4 |
| P2 | UI Preview Window | 2 |
| P2 | Model-agnostic parameter mapping | 3 |
| P3 | Structured metadata output | 6 |
| P3 | Extended style system | 5 |

---

## Open Questions

- [ ] What is the target ComfyUI version (1.x or 2.x) for Phase 2 UI features?
- [ ] Should the preview window be a standalone node or embedded in `PromptGeneratorNode`?
- [ ] Does the self-hosted runner have GPU for integration testing?
- [ ] Jira `CD-26` issue details unknown — need API token to fetch
