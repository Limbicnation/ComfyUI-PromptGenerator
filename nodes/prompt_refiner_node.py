"""
Prompt Refiner Node for ComfyUI
Refines a raw prompt through iterative LLM passes for higher quality output.
"""

import logging
from typing import Any

from .adapters.ollama_client import OllamaClient
from .prompt_generator_node import extract_final_prompt

logger = logging.getLogger(__name__)


class PromptRefinerNode:
    """
    ComfyUI node for refining prompts using iterative LLM passes.

    Takes a raw prompt string, sends it to Ollama with a refinement system prompt,
    and returns an improved version. Supports 1-3 refinement passes.

    VRAM safety: models are unloaded from Ollama VRAM immediately after
    execution via ``keep_alive="0s"``, and ``torch.cuda.empty_cache()`` is
    called asynchronously to prevent OOM in downstream diffusion nodes.
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
                "unload_model": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "Unload After Use",
                        "label_off": "Keep Loaded",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("refined_prompt",)
    FUNCTION = "refine"
    CATEGORY = "text/generation"
    OUTPUT_NODE = False

    def refine(
        self,
        prompt: str,
        model: str,
        passes: int = 1,
        temperature: float = 0.5,
        top_p: float = 0.9,
        seed: int = -1,
        timeout: int = 120,
        unload_model: bool = True,
        unique_id: str | None = None,
    ) -> tuple[str]:
        """
        Refine a prompt through iterative LLM passes.

        Args:
            prompt: Raw prompt string to refine
            model: Ollama model to use
            passes: Number of refinement iterations (1-3)
            temperature: Generation temperature
            top_p: Top-p sampling parameter
            seed: Seed for deterministic generation (-1 for random)
            timeout: Maximum generation time per pass
            unload_model: If True, unload Ollama model from VRAM after execution
            unique_id: ComfyUI node execution ID for progress tracking

        Returns:
            Tuple containing the refined prompt string
        """
        if not prompt.strip():
            return ("[PromptRefiner] Please provide a prompt to refine.",)

        client = OllamaClient(logger_prefix="PromptRefiner")
        pbar = client.create_progress_bar(unique_id)
        current_prompt = prompt.strip()

        # Determine effective seed
        effective_seed: int | None = None if seed == -1 else seed

        used_subprocess = False
        try:
            for i in range(passes):
                logger.info("Pass %d/%d with model='%s'", i + 1, passes, model)

                if pbar is not None:
                    progress = int((i / passes) * 100)
                    pbar.update_absolute(progress)

                # Build refinement prompt
                refinement = self.REFINEMENT_PROMPT.format(prompt=current_prompt)

                # Derive per-pass seed so multi-pass refinement isn't a no-op
                pass_seed = None if effective_seed is None else effective_seed + i

                # Generate refined version with keep_alive="0s" to unload
                # the model from Ollama VRAM immediately after each pass.
                result = client.generate_streaming(
                    model=model,
                    prompt=refinement,
                    temperature=temperature,
                    top_p=top_p,
                    timeout=timeout,
                    pbar=pbar,
                    seed=pass_seed,
                    keep_alive="0s",
                )

                if result.kind == "ok" and result.text is not None:
                    output = result.text
                elif result.kind in ("model_crash", "server_error", "unavailable"):
                    # Subprocess fallback would also fail; surface directly.
                    return (f"[PromptRefiner] Pass {i + 1}: {result.message}",)
                else:
                    # timeout / transient — try subprocess
                    used_subprocess = True
                    success, output = client.generate_subprocess(model, refinement, timeout)
                    if not success:
                        return (f"[PromptRefiner] Pass {i + 1} failed: {output}",)

                # Clean the output
                cleaned = extract_final_prompt(output.strip())
                if cleaned:
                    current_prompt = cleaned
                    logger.info("Pass %d complete: %d chars", i + 1, len(current_prompt))
                else:
                    logger.warning("Pass %d returned empty, keeping previous", i + 1)

            if pbar is not None:
                pbar.update_absolute(100)

            return (current_prompt,)

        finally:
            # Always release VRAM after node execution, regardless of success/failure.
            # keep_alive="0s" already handles Ollama-side unloading; this covers
            # the PyTorch CUDA allocator cache.
            if unload_model:
                OllamaClient.cleanup_async(
                    model=model,
                    logger_prefix="PromptRefiner",
                    # streaming path already evicted via keep_alive="0s"; only the
                    # subprocess fallback leaves a model loaded at the 5m default.
                    unload=used_subprocess,
                    release_cuda=True,
                )
