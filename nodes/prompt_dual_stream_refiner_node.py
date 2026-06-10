"""
Dual-Stream Prompt Refiner Node for ComfyUI.

Refines a raw description into a *pair* of prompts — a positive prompt and a
negative prompt — in a single generation pass, served via Ollama.

Designed for the `Limbicnation/qwen2-5-7b-dual-stream-prompt-lora` model after
its shipped Q8 GGUF has been registered with Ollama (see
config/Modelfile.dualstream), but works with any chat-capable Ollama model: the
"Positive:" / "Negative:" output format is imposed by the instruction prompt
rather than relying on the model's training, and parsing is delimiter-tolerant.
"""

import logging
import re
from typing import Any

from .adapters.ollama_client import OllamaClient
from .prompt_generator_node import extract_final_prompt

logger = logging.getLogger(__name__)


# Strip a leading "Positive[ prompt]:" / "Negative[ prompt]:" label, tolerating
# markdown bold (**) both before the label and after the separator, plus a ":"
# or "-" separator. e.g. "**Positive:** text".
_POS_LABEL = re.compile(r"^\s*\**\s*positive(?:\s+prompt)?\s*\**\s*[:\-]\s*\**\s*", re.IGNORECASE)
# Locate the start of the negative section anywhere in the text.
_NEG_LABEL = re.compile(r"\**\s*negative(?:\s+prompt)?\s*\**\s*[:\-]\s*\**\s*", re.IGNORECASE)


def parse_dual_stream(text: str) -> tuple[str, str]:
    """Split a model response into (positive_prompt, negative_prompt).

    Thinking/markdown noise is stripped first via ``extract_final_prompt`` so a
    "Negative:" mention inside a reasoning block can't cause a mis-split. The
    text is then split at the first "Negative:" label; everything before it is
    the positive prompt (with any leading "Positive:" label removed). If no
    negative label is found, the whole response is treated as the positive
    prompt and the negative is empty.
    """
    cleaned = (extract_final_prompt(text) or text).strip()

    neg_match = _NEG_LABEL.search(cleaned)
    if neg_match:
        positive = cleaned[: neg_match.start()]
        negative = cleaned[neg_match.end() :]
    else:
        positive = cleaned
        negative = ""

    positive = _POS_LABEL.sub("", positive, count=1).strip().strip('"').strip()
    negative = negative.strip().strip('"').strip()
    return positive, negative


class PromptDualStreamRefinerNode:
    """ComfyUI node that turns a description into a positive + negative prompt pair."""

    INSTRUCTION_PROMPT = """You are an expert Stable Diffusion prompt engineer.

From the description below, produce TWO prompts:
1. A detailed positive prompt — subject, style, lighting, composition, color, and quality tags.
2. A negative prompt — artifacts and qualities to avoid (e.g. blurry, lowres, deformed, watermark).

Respond in EXACTLY this format, with no explanations or markdown:
Positive: <positive prompt>
Negative: <negative prompt>

Description: {prompt}"""

    @classmethod
    def _get_available_models(cls) -> list[str]:
        """Fetch available Ollama models via OllamaClient (LoRA/prompt models first)."""
        client = OllamaClient(logger_prefix="PromptDualStreamRefiner")
        return client.discover_models()

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        available_models = cls._get_available_models()
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Description to expand into positive + negative prompts...",
                    },
                ),
                "model": (
                    available_models,
                    {
                        "default": available_models[0] if available_models else "qwen3:8b",
                        "tooltip": "Select Ollama model. Dual-stream/LoRA models appear first.",
                    },
                ),
            },
            "optional": {
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "top_p": (
                    "FLOAT",
                    {
                        "default": 0.9,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": -1,
                        "min": -1,
                        "max": 2**31 - 1,
                        "step": 1,
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 120,
                        "min": 30,
                        "max": 600,
                        "step": 10,
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("positive_prompt", "negative_prompt")
    FUNCTION = "refine"
    CATEGORY = "text/generation"
    OUTPUT_NODE = False

    def refine(
        self,
        prompt: str,
        model: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
        seed: int = -1,
        timeout: int = 120,
        unique_id: str | None = None,
    ) -> tuple[str, str]:
        """Generate a positive/negative prompt pair from a raw description.

        Returns:
            (positive_prompt, negative_prompt). On failure the error message is
            returned as the positive output and the negative is empty, matching
            the error-surfacing convention of PromptRefinerNode.
        """
        if not prompt.strip():
            return ("[PromptDualStreamRefiner] Please provide a description.", "")

        client = OllamaClient(logger_prefix="PromptDualStreamRefiner")
        pbar = client.create_progress_bar(unique_id)

        effective_seed: int | None = None if seed == -1 else seed
        instruction = self.INSTRUCTION_PROMPT.format(prompt=prompt.strip())

        logger.info("Dual-stream refine with model='%s'", model)
        result = client.generate_streaming(
            model=model,
            prompt=instruction,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
            pbar=pbar,
            seed=effective_seed,
        )

        if result.kind == "ok" and result.text is not None:
            output = result.text
        elif result.kind in ("model_crash", "server_error", "unavailable"):
            # Subprocess fallback won't help for these; surface directly.
            return (f"[PromptDualStreamRefiner] {result.message}", "")
        else:
            # timeout / transient — try the CLI subprocess fallback.
            success, output = client.generate_subprocess(model, instruction, timeout)
            if not success:
                return (f"[PromptDualStreamRefiner] {output}", "")

        positive, negative = parse_dual_stream(output)

        if pbar is not None:
            pbar.update_absolute(100)

        if not positive and not negative:
            logger.warning("Dual-stream parse produced empty output")
            return ("[PromptDualStreamRefiner] Model returned no usable prompt.", "")

        logger.info("Dual-stream complete: +%d / -%d chars", len(positive), len(negative))
        return (positive, negative)
