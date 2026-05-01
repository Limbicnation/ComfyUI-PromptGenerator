"""
Negative Prompt Generator Node for ComfyUI
Generates negative prompts from a positive prompt or description.
"""

import subprocess
import time
import threading
from typing import Any, Dict, List, Optional, Tuple

from .prompt_generator_node import extract_final_prompt

try:
    import ollama
    OLLAMA_API_AVAILABLE = True
except ImportError:
    OLLAMA_API_AVAILABLE = False

try:
    import comfy.utils
    COMFY_PROGRESS_AVAILABLE = True
except ImportError:
    COMFY_PROGRESS_AVAILABLE = False


# Default negative prompt categories for fallback
DEFAULT_NEGATIVE_CATEGORIES: Dict[str, List[str]] = {
    "quality": [
        "low quality",
        "worst quality",
        "bad anatomy",
        "bad proportions",
        "blurry",
        "out of focus",
        "deformed",
        "disfigured",
        "extra limbs",
        "mutated",
        "poorly drawn",
        "ugly",
    ],
    "artifacts": [
        "jpeg artifacts",
        "compression artifacts",
        "noise",
        "grainy",
        "pixelated",
        "oversaturated",
        "watermark",
        "signature",
        "text",
        "logo",
        "cropped",
        "out of frame",
    ],
    "people": [
        "bad face",
        "asymmetric eyes",
        "crossed eyes",
        "missing fingers",
        "extra fingers",
        "fused fingers",
        "too many fingers",
        "malformed hands",
        "bad hands",
        "missing arms",
        "missing legs",
        "extra arms",
        "extra legs",
    ],
    "style": [
        "cartoon",
        "anime",
        "3d render",
        "cgi",
        "plastic",
        "doll",
        "painting",
        "sketch",
        "drawing",
        "illustration",
    ],
}


class NegativePromptNode:
    """
    ComfyUI node for generating negative prompts.

    Supports two modes:
    - auto: Uses an LLM (Ollama) to generate a context-aware negative prompt
      based on the positive prompt content.
    - preset: Combines predefined negative keyword categories.

    Outputs:
        - negative_prompt: The generated negative prompt string
        - category_list: Comma-separated list of categories used (for reference)
    """

    SYSTEM_PROMPT = """You are an expert Stable Diffusion negative prompt engineer.

Given a positive prompt, generate a concise negative prompt that lists only
what should be avoided. Focus on:
- Quality issues (blurry, low quality, bad anatomy)
- Unwanted style elements (if the prompt specifies photorealistic, avoid cartoon/anime)
- Artifacts and technical problems
- Content that contradicts the positive prompt

Rules:
- Return ONLY the negative prompt text, comma-separated.
- No explanations, no markdown, no bullet points.
- Keep it under 200 tokens.
- Do NOT include positive concepts.

Positive prompt: {prompt}

Negative prompt:"""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "positive_prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Enter the positive prompt to generate negatives for...",
                    },
                ),
                "mode": (
                    ["auto", "preset"],
                    {"default": "preset"},
                ),
            },
            "optional": {
                "model": (
                    "STRING",
                    {
                        "default": "qwen3:8b",
                        "placeholder": "Ollama model (auto mode only)",
                    },
                ),
                "categories": (
                    "STRING",
                    {
                        "default": "quality,artifacts",
                        "placeholder": "Comma-separated: quality,artifacts,people,style",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.3,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 60,
                        "min": 10,
                        "max": 300,
                        "step": 10,
                    },
                ),
                "custom_negatives": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "Additional custom negative terms, comma-separated...",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("negative_prompt", "category_list")
    FUNCTION = "generate"
    CATEGORY = "text/generation"
    OUTPUT_NODE = False

    def _generate_streaming(
        self,
        model: str,
        prompt: str,
        temperature: float,
        timeout: int,
        pbar: object = None,
    ) -> Optional[str]:
        """Stream ollama.generate() with timeout enforcement."""
        chunks = []
        start = time.monotonic()
        first_chunk_timeout = min(timeout * 0.6, 45)
        chunk_timeout = 20
        got_first_chunk = False

        try:
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options={"temperature": temperature, "top_p": 0.9},
            )

            result_holder: Dict[str, Any] = {}

            def _iter_next(it):
                try:
                    result_holder["chunk"] = next(it)
                    result_holder["done"] = False
                except StopIteration:
                    result_holder["done"] = True
                except Exception as exc:
                    result_holder["error"] = exc

            it = iter(stream)
            chunk_count = 0

            while True:
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    print(f"[NegativePrompt] Total timeout ({timeout}s) reached")
                    break

                result_holder.clear()
                t = threading.Thread(target=_iter_next, args=(it,), daemon=True)
                t.start()

                wait_time = first_chunk_timeout if not got_first_chunk else chunk_timeout
                wait_time = min(wait_time, timeout - elapsed)
                t.join(timeout=wait_time)

                if t.is_alive():
                    label = "first chunk" if not got_first_chunk else "chunk"
                    print(f"[NegativePrompt] Timeout waiting for {label} ({wait_time:.0f}s)")
                    return None

                if "error" in result_holder:
                    raise result_holder["error"]

                if result_holder.get("done", False):
                    break

                chunk = result_holder.get("chunk")
                if chunk is None:
                    break

                text = chunk.get("response", "")
                if text:
                    chunks.append(text)
                    got_first_chunk = True
                    chunk_count += 1
                    if pbar is not None:
                        progress = min(5 + int(chunk_count * 5), 95)
                        pbar.update_absolute(progress)

        except Exception as e:
            print(f"[NegativePrompt] Streaming error: {e}")
            return None

        if not chunks:
            return None

        return "".join(chunks)

    def _build_preset_negative(
        self,
        categories_str: str,
        custom_negatives: str,
    ) -> Tuple[str, str]:
        """Build negative prompt from preset categories."""
        selected = [c.strip().lower() for c in categories_str.split(",") if c.strip()]
        terms: List[str] = []
        valid_categories: List[str] = []

        for cat in selected:
            if cat in DEFAULT_NEGATIVE_CATEGORIES:
                terms.extend(DEFAULT_NEGATIVE_CATEGORIES[cat])
                valid_categories.append(cat)
            else:
                print(f"[NegativePrompt] Unknown category '{cat}', skipping")

        if custom_negatives:
            custom_terms = [t.strip() for t in custom_negatives.split(",") if t.strip()]
            terms.extend(custom_terms)

        if not terms:
            return ("", ",".join(valid_categories))

        return (", ".join(terms), ",".join(valid_categories))

    def generate(
        self,
        positive_prompt: str,
        mode: str,
        model: str = "qwen3:8b",
        categories: str = "quality,artifacts",
        temperature: float = 0.3,
        timeout: int = 60,
        custom_negatives: str = "",
        unique_id: str = None,
    ) -> Tuple[str, str]:
        """
        Generate a negative prompt.

        Args:
            positive_prompt: The positive prompt to generate negatives for
            mode: "auto" (LLM-generated) or "preset" (keyword categories)
            model: Ollama model for auto mode
            categories: Comma-separated category names for preset mode
            temperature: Generation temperature for auto mode
            timeout: Max generation time for auto mode
            custom_negatives: Additional custom terms for preset mode
            unique_id: ComfyUI node ID for progress tracking

        Returns:
            Tuple of (negative_prompt, category_list)
        """
        if not positive_prompt.strip():
            return ("[NegativePrompt] Please provide a positive prompt.", "")

        pbar = None
        if COMFY_PROGRESS_AVAILABLE and unique_id is not None:
            try:
                pbar = comfy.utils.ProgressBar(100, node_id=unique_id)
                pbar.update_absolute(0)
            except Exception:
                pbar = None

        if mode == "preset":
            neg, cats = self._build_preset_negative(categories, custom_negatives)
            if pbar is not None:
                pbar.update_absolute(100)
            return (neg, cats)

        # Auto mode: use Ollama
        system_prompt = self.SYSTEM_PROMPT.format(prompt=positive_prompt.strip())

        if OLLAMA_API_AVAILABLE:
            output = self._generate_streaming(
                model=model,
                prompt=system_prompt,
                temperature=temperature,
                timeout=timeout,
                pbar=pbar,
            )

            if output is not None:
                cleaned = extract_final_prompt(output.strip())
                if cleaned:
                    if pbar is not None:
                        pbar.update_absolute(100)
                    return (cleaned, "auto")
                else:
                    print("[NegativePrompt] Auto mode returned empty, falling back to preset")

        # Fallback to subprocess or preset
        if OLLAMA_API_AVAILABLE:
            print("[NegativePrompt] Streaming failed, trying subprocess fallback")
        else:
            print("[NegativePrompt] Ollama API not available, using subprocess fallback")

        try:
            result = subprocess.run(
                ["ollama", "run", model, system_prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            if result.returncode == 0 and result.stdout.strip():
                cleaned = extract_final_prompt(result.stdout.strip())
                if cleaned:
                    if pbar is not None:
                        pbar.update_absolute(100)
                    return (cleaned, "auto")
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"[NegativePrompt] Subprocess fallback failed: {e}")

        # Final fallback: return preset negative
        print("[NegativePrompt] All auto methods failed, returning preset negative")
        neg, cats = self._build_preset_negative(categories, custom_negatives)
        if pbar is not None:
            pbar.update_absolute(100)
        return (neg, cats)
