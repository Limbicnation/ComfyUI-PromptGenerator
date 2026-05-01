"""
Prompt Refiner Node for ComfyUI
Refines a raw prompt through iterative LLM passes for higher quality output.
"""

import logging
import subprocess
import time
import threading
from typing import Any, Dict, Optional, Tuple

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

logger = logging.getLogger(__name__)


class PromptRefinerNode:
    """
    ComfyUI node for refining prompts using iterative LLM passes.

    Takes a raw prompt string, sends it to Ollama with a refinement system prompt,
    and returns an improved version. Supports 1-3 refinement passes.
    """

    REFINEMENT_PROMPT = """You are an expert prompt engineer for Stable Diffusion.

Refine the following prompt to improve its quality, specificity, and coherence.
Keep the core subject intact but enhance:
- Descriptive detail (textures, lighting, atmosphere)
- Technical quality markers (8k, highly detailed, masterpiece)
- Composition and framing cues
- Color palette hints

Return ONLY the refined prompt text. No explanations, no markdown formatting.

Original prompt: {prompt}

Refined prompt:"""

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Raw prompt to refine...",
                    },
                ),
                "model": (
                    "STRING",
                    {
                        "default": "qwen3:8b",
                        "placeholder": "Ollama model name",
                    },
                ),
            },
            "optional": {
                "passes": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 3,
                        "step": 1,
                        "display": "slider",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.5,
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
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("refined_prompt",)
    FUNCTION = "refine"
    CATEGORY = "text/generation"
    OUTPUT_NODE = False

    def _generate_streaming(
        self,
        model: str,
        prompt: str,
        temperature: float,
        top_p: float,
        timeout: int,
        pbar: object = None,
    ) -> Optional[str]:
        """
        Stream ollama.generate() with per-chunk and total timeout enforcement.
        Returns the full response text, or None on failure (caller should fallback).
        """
        chunks = []
        start = time.monotonic()
        first_chunk_timeout = min(timeout * 0.6, 90)
        chunk_timeout = 30
        got_first_chunk = False

        try:
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options={"temperature": temperature, "top_p": top_p},
            )

            result_holder: Dict[str, Any] = {}

            def _iter_next(it):
                """Get next chunk from iterator in a thread."""
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
                    print(f"[PromptRefiner] Total timeout ({timeout}s) reached")
                    break

                result_holder.clear()
                t = threading.Thread(target=_iter_next, args=(it,), daemon=True)
                t.start()

                wait_time = first_chunk_timeout if not got_first_chunk else chunk_timeout
                wait_time = min(wait_time, timeout - elapsed)
                t.join(timeout=wait_time)

                if t.is_alive():
                    label = "first chunk" if not got_first_chunk else "chunk"
                    print(f"[PromptRefiner] Timeout waiting for {label} ({wait_time:.0f}s)")
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
                        progress = min(5 + int(chunk_count * 2), 95)
                        pbar.update_absolute(progress)

        except Exception as e:
            print(f"[PromptRefiner] Streaming error: {e}")
            return None

        if not chunks:
            return None

        elapsed = time.monotonic() - start
        full_text = "".join(chunks)
        print(f"[PromptRefiner] Streaming complete: {len(full_text)} chars in {elapsed:.1f}s")
        return full_text

    def refine(
        self,
        prompt: str,
        model: str,
        passes: int = 1,
        temperature: float = 0.5,
        top_p: float = 0.9,
        seed: int = -1,
        timeout: int = 120,
        unique_id: str = None,
    ) -> Tuple[str]:
        """
        Refine a prompt through iterative LLM passes.

        Args:
            prompt: Raw prompt string to refine
            model: Ollama model to use
            passes: Number of refinement iterations (1-3)
            temperature: Generation temperature
            seed: Seed for deterministic generation (-1 for random)
            timeout: Maximum generation time per pass
            unique_id: ComfyUI node execution ID for progress tracking

        Returns:
            Tuple containing the refined prompt string
        """
        if not prompt.strip():
            return ("[PromptRefiner] Please provide a prompt to refine.",)

        # Initialize progress bar
        pbar = None
        if COMFY_PROGRESS_AVAILABLE and unique_id is not None:
            try:
                pbar = comfy.utils.ProgressBar(100, node_id=unique_id)
                pbar.update_absolute(0)
            except Exception:
                pbar = None

        current_prompt = prompt.strip()
        effective_seed: Optional[int] = None if seed == -1 else seed

        for i in range(passes):
            print(f"[PromptRefiner] Pass {i + 1}/{passes} with model='{model}'")

            if pbar is not None:
                progress = int((i / passes) * 100)
                pbar.update_absolute(progress)

            # Build refinement prompt
            refinement = self.REFINEMENT_PROMPT.format(prompt=current_prompt)

            # Derive per-pass seed so multi-pass refinement isn't a no-op
            pass_seed = None if effective_seed is None else effective_seed + i

            output = None
            if OLLAMA_API_AVAILABLE:
                try:
                    output = self._generate_streaming(
                        model=model,
                        prompt=refinement,
                        temperature=temperature,
                        top_p=top_p,
                        timeout=timeout,
                        pbar=pbar,
                    )
                except Exception as e:
                    print(f"[PromptRefiner] Streaming failed: {e}")

            if output is None:
                # Fallback to subprocess
                print("[PromptRefiner] Falling back to subprocess")
                try:
                    cmd = ["ollama", "run", model, refinement]
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                    if result.returncode != 0:
                        return (f"[PromptRefiner] Pass {i + 1} failed: {result.stderr}",)
                    output = result.stdout.strip()
                except subprocess.TimeoutExpired:
                    return (f"[PromptRefiner] Pass {i + 1} timed out after {timeout}s",)
                except FileNotFoundError:
                    return ("[PromptRefiner] Ollama not found. Install from: https://ollama.ai",)
                except Exception as e:
                    return (f"[PromptRefiner] Pass {i + 1} error: {e}",)

            # Clean the output
            cleaned = extract_final_prompt(output.strip())
            if cleaned:
                current_prompt = cleaned
                print(f"[PromptRefiner] Pass {i + 1} complete: {len(current_prompt)} chars")
            else:
                print(f"[PromptRefiner] Pass {i + 1} returned empty, keeping previous")

        if pbar is not None:
            pbar.update_absolute(100)

        return (current_prompt,)
