"""
Prompt Refiner Node for ComfyUI
Refines a raw prompt through iterative LLM passes for higher quality output.
"""

import logging
from typing import Any, Dict, Tuple, Optional

from .adapters.ollama_client import OllamaClient
from .prompt_generator_node import extract_final_prompt

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
    def INPUT_TYPES(cls) -> Dict[str, Any]:
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
        unique_id: Optional[str] = None,
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

        client = OllamaClient(logger_prefix="PromptRefiner")
        pbar = client.create_progress_bar(unique_id)
        current_prompt = prompt.strip()

        # Determine effective seed
        effective_seed: Optional[int] = None if seed == -1 else seed

        for i in range(passes):
            logger.info("Pass %d/%d with model='%s'", i + 1, passes, model)

            if pbar is not None:
                progress = int((i / passes) * 100)
                pbar.update_absolute(progress)

            # Build refinement prompt
            refinement = self.REFINEMENT_PROMPT.format(prompt=current_prompt)

            # Generate refined version
            output = client.generate_streaming(
                model=model,
                prompt=refinement,
                temperature=temperature,
                top_p=top_p,
                timeout=timeout,
                pbar=pbar,
                seed=effective_seed,
            )

            if output is None:
                # Fallback to subprocess
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
