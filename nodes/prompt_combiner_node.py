"""
Prompt Combiner Node for ComfyUI
Merges multiple prompt strings with configurable blending strategies.
"""

from enum import StrEnum
from typing import Any, Literal, get_args


class CombineMode(StrEnum):
    """Supported strategies for combining multiple prompts."""

    BLEND = "blend"
    CONCAT = "concat"
    WEIGHTED_AVERAGE = "weighted_average"


# Single source of truth for the choices exposed in INPUT_TYPES and accepted by combine().
ModeLiteral = Literal["blend", "concat", "weighted_average"]


class PromptCombinerNode:
    """
    ComfyUI node for combining multiple prompts into a single output.

    Supports:
    - blend: Weighted combination with emphasis markers
    - concat: Simple concatenation with separator
    - weighted_average: Text interpolation based on weights
    """

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        return {
            "required": {
                "prompt_1": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "First prompt...",
                    },
                ),
                "mode": (
                    list(get_args(ModeLiteral)),
                    {"default": CombineMode.BLEND.value},
                ),
            },
            "optional": {
                "prompt_2": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Second prompt (optional)...",
                    },
                ),
                "prompt_3": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Third prompt (optional)...",
                    },
                ),
                "prompt_4": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "Fourth prompt (optional)...",
                    },
                ),
                "weight_1": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "weight_2": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "weight_3": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "weight_4": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "display": "slider",
                    },
                ),
                "separator": (
                    "STRING",
                    {
                        "default": ", ",
                        "placeholder": "Separator for concat mode",
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("combined_prompt",)
    FUNCTION = "combine"
    CATEGORY = "text/generation"
    OUTPUT_NODE = False

    def combine(
        self,
        prompt_1: str,
        mode: ModeLiteral,
        prompt_2: str = "",
        prompt_3: str = "",
        prompt_4: str = "",
        weight_1: float = 1.0,
        weight_2: float = 1.0,
        weight_3: float = 1.0,
        weight_4: float = 1.0,
        separator: str = ", ",
    ) -> tuple[str]:
        """
        Combine multiple prompts using the selected mode.

        Args:
            prompt_1: First prompt (required)
            mode: Combination strategy
            prompt_2-4: Additional prompts (optional)
            weight_1-4: Weights for each prompt
            separator: Separator string for concat mode

        Returns:
            Tuple containing the combined prompt string
        """
        prompts: list[tuple[str, float]] = [
            (p.strip(), w)
            for p, w in (
                (prompt_1, weight_1),
                (prompt_2, weight_2),
                (prompt_3, weight_3),
                (prompt_4, weight_4),
            )
            if p and p.strip()
        ]

        if not prompts:
            return ("[PromptCombiner] At least one prompt is required.",)

        try:
            selected = CombineMode(mode)
        except ValueError:
            valid = ", ".join(m.value for m in CombineMode)
            return (f"[PromptCombiner] Unknown mode {mode!r}. Valid: {valid}",)

        if len(prompts) == 1:
            return (prompts[0][0],)

        match selected:
            case CombineMode.BLEND:
                return (self._blend(prompts),)
            case CombineMode.CONCAT:
                return (self._concat(prompts, separator),)
            case CombineMode.WEIGHTED_AVERAGE:
                return (self._weighted_average(prompts),)

    def _blend(self, prompts: list[tuple[str, float]]) -> str:
        """
        Blend prompts using ComfyUI-style emphasis markers.
        Higher weight = more parentheses emphasis.
        """
        parts = []
        for text, weight in prompts:
            if weight <= 0:
                continue
            # Map weight to emphasis levels
            if weight >= 1.5:
                parts.append(f"(({text}))")
            elif weight >= 1.2:
                parts.append(f"({text})")
            elif weight <= 0.5:
                parts.append(f"[{text}]")
            else:
                parts.append(text)
        return ", ".join(parts)

    def _concat(self, prompts: list[tuple[str, float]], separator: str) -> str:
        """Simple concatenation with separator."""
        texts = [p[0] for p in prompts]
        return separator.join(texts)

    def _weighted_average(self, prompts: list[tuple[str, float]]) -> str:
        """
        Weighted text combination using emphasis markers.
        Higher-weighted prompts receive stronger ComfyUI emphasis parentheses.
        """
        total_weight = sum(w for _, w in prompts)
        if total_weight == 0:
            return ", ".join(p[0] for p in prompts)

        # Normalize weights relative to average
        avg_weight = total_weight / len(prompts)

        parts = []
        for text, weight in prompts:
            # Compute emphasis level based on weight ratio to average
            ratio = weight / avg_weight if avg_weight > 0 else 1.0
            if ratio >= 2.0:
                # Strong emphasis: triple parens
                parts.append(f"((({text})))")
            elif ratio >= 1.5:
                # High emphasis: double parens
                parts.append(f"(({text}))")
            elif ratio >= 1.2:
                # Moderate emphasis: single parens
                parts.append(f"({text})")
            elif ratio <= 0.5:
                # De-emphasis: square brackets
                parts.append(f"[{text}]")
            else:
                # Neutral: no markers
                parts.append(text)

        return ", ".join(parts)
