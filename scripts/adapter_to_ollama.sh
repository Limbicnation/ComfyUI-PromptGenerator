#!/usr/bin/env bash
# Convert a Hugging Face PEFT LoRA adapter into an Ollama model for the
# ComfyUI-PromptGenerator nodes.
#
# Steps: download adapter -> merge into base (CPU) -> GGUF (q8_0) -> ollama create.
# The intermediate fp16 merge is deleted at the end to save disk.
#
# Usage:
#   scripts/adapter_to_ollama.sh <hf_adapter_repo> <ollama_model_name> [outtype]
# Example:
#   scripts/adapter_to_ollama.sh \
#     Limbicnation/qwen3-4b-deforum-video-dual-stream-lora-v1 \
#     qwen3-4b-deforum-dual-stream-lora:v1
#
# Name the Ollama model with one of the node's priority keywords
# (lora / limbicnation / fine / style / prompt) so it sorts to the top of the
# ComfyUI model dropdown.
set -euo pipefail

ADAPTER_REPO="${1:?HF adapter repo required}"
OLLAMA_NAME="${2:?Ollama model name required}"
OUTTYPE="${3:-q8_0}"

PYENV="$HOME/anaconda3/envs/prompt-lora-trainer/bin/python"   # has peft+transformers+torch
LLAMA_CPP="$HOME/GitHub/llama.cpp"                            # has convert_hf_to_gguf.py
WORK="$HOME/ollama-convert/${OLLAMA_NAME//[:\/]/_}"
MERGED="$WORK/merged"
GGUF="$WORK/model-${OUTTYPE}.gguf"

echo "==> Work dir: $WORK"
mkdir -p "$WORK"

echo "==> [1/4] Merging adapter into base (CPU)"
"$PYENV" - "$ADAPTER_REPO" "$MERGED" <<'PY'
import sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

adapter_repo, out_dir = sys.argv[1], sys.argv[2]
base = PeftConfig.from_pretrained(adapter_repo).base_model_name_or_path
print(f"    base model: {base}")
tok = AutoTokenizer.from_pretrained(adapter_repo)   # tokenizer travels with the adapter
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter_repo)
model = model.merge_and_unload()
model.save_pretrained(out_dir, safe_serialization=True)
tok.save_pretrained(out_dir)
print(f"    merged -> {out_dir}")
PY

echo "==> [2/4] Converting to GGUF ($OUTTYPE)"
"$PYENV" -m pip install -q gguf >/dev/null 2>&1 || true
"$PYENV" "$LLAMA_CPP/convert_hf_to_gguf.py" "$MERGED" --outfile "$GGUF" --outtype "$OUTTYPE"

echo "==> [3/4] Creating Ollama model: $OLLAMA_NAME"
cat > "$WORK/Modelfile" <<EOF
FROM $GGUF

# The ComfyUI dual-stream node sends its own Positive:/Negative: instruction
# prompt and temperature/top_p, so this SYSTEM line is a light fallback only.
SYSTEM """You are an expert Stable Diffusion / Deforum video prompt engineer. \
Respond with a vivid positive prompt and a matching negative prompt."""

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
EOF
ollama create "$OLLAMA_NAME" -f "$WORK/Modelfile"

echo "==> [4/4] Cleaning up fp16 merge ($(du -sh "$MERGED" | cut -f1))"
rm -rf "$MERGED"

echo "==> Done. GGUF kept at: $GGUF"
echo "    Test: ollama run $OLLAMA_NAME 'a neon cyberpunk alley, rain'"
