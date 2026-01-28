# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ComfyUI custom node that generates Stable Diffusion prompts using Qwen3-8B via Ollama. Users provide a brief description and select a style preset; the node outputs a detailed, optimized prompt.

## Architecture

```
__init__.py                 # ComfyUI entry point, exports NODE_CLASS_MAPPINGS
nodes/
  prompt_generator_node.py  # PromptGeneratorNode class with all logic
config/
  templates.yaml            # Jinja2 style templates (editable by users)
```

**Single node design**: All functionality is in `PromptGeneratorNode`. The node:

1. Loads style templates from YAML (or uses hardcoded defaults)
2. Renders the selected template with Jinja2 (or regex fallback)
3. Calls Ollama via Python API (or subprocess fallback)
4. Strips reasoning/thinking from output unless `include_reasoning=True`

**Graceful degradation**: Optional imports (`yaml`, `jinja2`, `ollama`) have fallbacks. The node works with subprocess calls even if Python packages aren't installed.

## Development

**Prerequisites**:

- Ollama running locally with `qwen3:8b` model pulled
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
- `CATEGORY`: Where node appears in ComfyUI menu

**Template system**: Templates in `config/templates.yaml` use Jinja2 syntax. Variables: `{{ description }}`, `{{ emphasis }}`, `{{ mood }}`. Conditionals: `{% if emphasis %}...{% endif %}`.

**Output cleaning**: `extract_final_prompt()` strips Qwen3's "Thinking..." blocks and markdown formatting from responses.

## Publishing

**Current version**: `1.0.4` (Published 2024-12-14)

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
