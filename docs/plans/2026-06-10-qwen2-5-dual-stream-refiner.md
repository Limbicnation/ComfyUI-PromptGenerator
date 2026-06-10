# Qwen2.5 Dual-Stream Prompt Refiner — Implementation (Ollama/GGUF)

> **Superseded note:** An earlier draft of this plan loaded the model locally via
> `transformers` + PEFT. That approach was **invalid** — the HF repo
> `Limbicnation/qwen2-5-7b-dual-stream-prompt-lora` is a LoRA *adapter* with **no base
> weights**, so `AutoModelForCausalLM.from_pretrained(...)` on it crashes. It also added a
> heavy ML stack to an Ollama-only project and loaded a 7B model into ComfyUI's diffusion
> VRAM. This document replaces it with the implemented Ollama/GGUF approach.

**Goal:** A ComfyUI node that turns a description into a positive + negative prompt pair,
powered by the pre-merged Q8 GGUF the HF repo already ships, served through Ollama.

**Repo facts:** PEFT adapter (`adapter_config.json` + `adapter_model.safetensors`) **plus**
`qwen2-5-7b-dual-stream-q8.gguf` (8.1 GB). Base: `Qwen/Qwen2.5-7B-Instruct`. "Dual-stream" =
a single causal LM trained to emit positive + negative blocks (no special architecture). The
shipped `chat_template.jinja` is the stock Qwen2.5 ChatML template and does **not** define the
positive/negative format — so the node imposes the format via its instruction prompt and parses
defensively.

## What was implemented

1. **`nodes/prompt_dual_stream_refiner_node.py`** — `PromptDualStreamRefinerNode`
   (`Limbicnation_PromptDualStreamRefiner`, display "Prompt Dual-Stream Refiner",
   category `text/generation`). Inputs: `prompt`, `model` (dropdown from
   `OllamaClient.discover_models()`), optional `temperature`/`top_p`/`seed`/`timeout`, hidden
   `unique_id`. Outputs: `(positive_prompt, negative_prompt)`. Reuses `OllamaClient` for
   streaming, progress bar, timeout, llama-runner crash categorization, and subprocess fallback.
   Module-level `parse_dual_stream()` strips thinking/markdown via `extract_final_prompt` then
   splits on a delimiter-tolerant `Negative:` label.
2. **`config/Modelfile.dualstream`** — registers the GGUF with Ollama. Suggested model name
   `limbicnation-dualstream-prompt` (contains `prompt`, a `LORA_KEYWORDS` token, so it sorts to
   the top of the dropdown).
3. **`__init__.py`** — node registered in both `NODE_CLASS_MAPPINGS` and
   `NODE_DISPLAY_NAME_MAPPINGS`.
4. **Tests** — `tests/unit/test_prompt_dual_stream_refiner.py` (parser variants + refine flow);
   `tests/unit/test_node_registration.py` `EXPECTED_NODES` updated.

## No new runtime dependencies

`requirements.txt` stays `ollama` / `jinja2` / `pyyaml`.

## Setup & verification

```bash
# 1. Get the GGUF into Ollama
huggingface-cli download Limbicnation/qwen2-5-7b-dual-stream-prompt-lora \
  qwen2-5-7b-dual-stream-q8.gguf --local-dir config
ollama create limbicnation-dualstream-prompt -f config/Modelfile.dualstream

# 2. Confirm the trained output format BEFORE trusting the parser
ollama run limbicnation-dualstream-prompt "a mystical forest at twilight"
#   -> expect "Positive: ...\nNegative: ...". If different, tune
#      INSTRUCTION_PROMPT / parse_dual_stream to match.

# 3. Code checks
ruff check . && ruff format --check .
python -m pytest tests/unit/test_prompt_dual_stream_refiner.py tests/unit/test_node_registration.py -q

# 4. Manual: restart ComfyUI, add "Prompt Dual-Stream Refiner" (text/generation),
#    select the model, run a description, confirm two correct outputs + progress bar.
```
