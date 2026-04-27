"""
Negative Prompt Generator Node for ComfyUI
Generates negative prompts from positive prompts using style-aware templates.
"""

from typing import Any, Dict, Tuple

from .adapters.ollama_client import OllamaClient
from .prompt_generator_node import extract_final_prompt


class NegativePromptNode:
    """
    ComfyUI node for generating negative prompts from positive prompts.

    Uses a dedicated Jinja2 template with SD/XL-specific negative token lists,
    tailored to the selected style.
    """

    NEGATIVE_PROMPT_TEMPLATE = """You are an expert in Stable Diffusion negative prompts.

Given this positive prompt and style, generate a concise negative prompt that lists
what should be avoided to improve image quality.

Style: {style}
Positive prompt: {prompt}

Generate a comma-separated list of negative keywords (no explanations, no markdown).
Focus on common artifacts for this style: {style_hints}

Negative prompt:"""

    STYLE_HINTS = {
        "cinematic": "blurry, overexposed, underexposed, shaky cam, lens flare abuse, bad CGI",
        "anime": "3d render, realistic, western cartoon, bad anatomy, extra limbs, deformed",
        "photorealistic": "painting, illustration, cartoon, oversaturated, artificial look",
        "fantasy": "modern objects, sci-fi elements, mundane setting, low detail",
        "abstract": "recognizable objects, literal interpretation, cluttered composition",
        "cyberpunk": "medieval, natural landscape, low-tech, clean utopia, bright daylight",
        "sci-fi": "fantasy magic, medieval, contemporary, low detail, unscientific",
        "video_wan": "static image, still frame, jump cut, bad temporal coherence",
        "still_image": "motion blur, video artifacts, interlaced, low resolution",
    }

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        styles = list(cls.STYLE_HINTS.keys())
        return {
            "required": {
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Positive prompt to generate negative for...",
                    },
                ),
                "style": (
                    styles,
                    {"default": "cinematic"},
                ),
                "model": (
                    "STRING",
                    {"default": "qwen3:8b"},
                ),
            },
            "optional": {
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
                "timeout": (
                    "INT",
                    {
                        "default": 60,
                        "min": 30,
                        "max": 300,
                        "step": 10,
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative_prompt",)
    FUNCTION = "generate_negative"
    CATEGORY = "text/generation"
    OUTPUT_NODE = False

    def generate_negative(
        self,
        prompt: str,
        style: str,
        model: str,
        temperature: float = 0.3,
        top_p: float = 0.9,
        timeout: int = 60,
    ) -> Tuple[str]:
        """
        Generate a negative prompt from a positive prompt.

        Args:
            prompt: Positive prompt string
            style: Style category for style-aware negative hints
            model: Ollama model to use
            temperature: Generation temperature (lower = more conservative)
            timeout: Maximum generation time

        Returns:
            Tuple containing the negative prompt string
        """
        if not prompt.strip():
            return ("[NegativePrompt] Please provide a positive prompt.",)

        client = OllamaClient(logger_prefix="NegativePrompt")
        style_hints = self.STYLE_HINTS.get(style, "low quality, blurry, bad anatomy")

        # Build the negative generation prompt
        negative_prompt_text = self.NEGATIVE_PROMPT_TEMPLATE.format(
            style=style,
            prompt=prompt.strip(),
            style_hints=style_hints,
        )

        print(f"[NegativePrompt] Generating negative for style='{style}'")

        # Generate via streaming
        output = client.generate_streaming(
            model=model,
            prompt=negative_prompt_text,
            temperature=temperature,
            top_p=top_p,
            timeout=timeout,
        )

        if output is None:
            # Fallback to subprocess
            success, output = client.generate_subprocess(
                model, negative_prompt_text, timeout
            )
            if not success:
                return (f"[NegativePrompt] Generation failed: {output}",)

        # Clean the output
        negative = extract_final_prompt(output.strip())
        if negative:
            print(f"[NegativePrompt] Generated {len(negative)} characters")
            return (negative,)
        else:
            # Fallback to static hints if LLM fails
            print("[NegativePrompt] LLM returned empty, using static hints")
            return (style_hints,)
