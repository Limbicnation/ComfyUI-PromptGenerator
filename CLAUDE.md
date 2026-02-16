# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI custom node that generates Stable Diffusion prompts using Ollama. Users provide a brief description and select a style preset; the node outputs a detailed, optimized prompt. Supports LoRA-enhanced models with automatic discovery and prioritization.

**Current version**: `1.1.6`

## Architecture

```
__init__.py                     # ComfyUI entry point, exports NODE_CLASS_MAPPINGS
nodes/
  prompt_generator_node.py      # PromptGeneratorNode class with all logic
config/
  templates.yaml                # Jinja2 style templates (editable by users)
  Modelfile.limbicnation        # Ollama Modelfile for LoRA-enhanced model
```

**Single node design**: All functionality is in `PromptGeneratorNode`. The node:

1. Loads style templates from YAML (or uses hardcoded defaults)
2. Renders the selected template with Jinja2 (or regex fallback)
3. Calls Ollama via streaming API with per-chunk timeouts and ComfyUI ProgressBar integration
4. Strips reasoning/thinking from output unless `include_reasoning=True`

**Graceful degradation**: Optional imports (`yaml`, `jinja2`, `ollama`) have fallbacks. The node works with subprocess calls even if Python packages aren't installed.

**Streaming & timeouts** (v1.1.6): `_generate_with_streaming()` uses a background thread to enforce per-chunk (30s) and total timeouts, avoiding the 120s blocking timeout issue. Cold starts get a 1.3x timeout multiplier.

## Development

**Prerequisites**:

- Ollama running locally with a model pulled (e.g. `qwen3:8b`)
- ComfyUI installation for integration testing

**Install dependencies**:

```bash
pip install -r requirements.txt
```

**Lint with ruff** (already configured in project):

```bash
ruff check .
ruff format .
```

**Test the node manually**: Restart ComfyUI, add the node from `text/generation` category, verify prompt generation works.

## Key Patterns

**ComfyUI node structure**:

- `INPUT_TYPES()`: Class method returning dict with `required` and `optional` inputs
- `RETURN_TYPES`, `RETURN_NAMES`: Output type definitions
- `FUNCTION`: Name of the method to call (`generate`)
- `CATEGORY`: Where node appears in ComfyUI menu (`text/generation`)

**Template system**: Templates in `config/templates.yaml` use Jinja2 syntax. Variables: `{{ description }}`, `{{ emphasis }}`, `{{ mood }}`. Conditionals: `{% if emphasis %}...{% endif %}`.

**Output cleaning**: `extract_final_prompt()` strips Qwen3's "Thinking..." blocks and markdown formatting from responses.

**Model discovery**: The node auto-discovers Ollama models and prioritizes those containing `lora` or `limbicnation` keywords. All available models appear in the dropdown.

## Publishing

**Registry**: [registry.comfy.org](https://registry.comfy.org) - Node ID: `comfyui-prompt-generator`

**Publish a new version**:

```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git add -A && git commit -m "chore: bump version to X.Y.Z"

# 3. Tag and push (triggers GitHub Actions workflow)
git tag vX.Y.Z
git push origin main && git push origin vX.Y.Z

# Or publish manually:
comfy node publish --confirm
```

**CI/CD**: `.github/workflows/publish.yml` auto-publishes on version tags (`v*.*.*`).

## Adding New Styles

1. Add entry to `config/templates.yaml` following existing format
2. Add style key to `INPUT_TYPES()` style combo list in `prompt_generator_node.py`
3. Optionally add to `DEFAULT_STYLES` dict for fallback when YAML unavailable

## LoRA-Enhanced Models

- **Training**: QLoRA on Qwen3-4B-Instruct-2507 using the [Limbicnation/Images-Diffusion-Prompt-Style](https://huggingface.co/datasets/Limbicnation/Images-Diffusion-Prompt-Style) dataset (750 prompts)
- **Quantization**: Merged LoRA and converted to Q8_0 GGUF for Ollama
- **Integration**: `PromptGeneratorNode` auto-discovers and prioritizes models containing `lora` or `limbicnation` keywords

### Creating a LoRA-Enhanced Model

1. Fine-tune a LoRA on the dataset above
2. Export as `.safetensors` (non-quantized recommended)
3. Create the Ollama model:
   ```bash
   # Edit config/Modelfile.limbicnation with your adapter path
   ollama create qwen3-limbicnation -f config/Modelfile.limbicnation
   ```
4. Restart ComfyUI — the new model will appear in the dropdown

### Modelfile Template

See `config/Modelfile.limbicnation` for a pre-configured template with:

- Limbicnation system prompt
- Optimal temperature/top_p settings
- ADAPTER placeholder for your LoRA
