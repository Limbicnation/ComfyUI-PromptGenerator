"""
Prompt Combiner Node for ComfyUI
Merges multiple prompt strings with configurable blending strategies.
"""

from typing import Any, Dict, List, Tuple


class PromptCombinerNode:
    """
    ComfyUI node for combining multiple prompts into a single output.

    Supports:
    - blend: Weighted combination with emphasis markers
    - concat: Simple concatenation with separator
    - weighted_average: Text interpolation based on weights
    """

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
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
                    ["blend", "concat", "weighted_average"],
                    {"default": "blend"},
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
        mode: str,
        prompt_2: str = "",
        prompt_3: str = "",
        prompt_4: str = "",
        weight_1: float = 1.0,
        weight_2: float = 1.0,
        weight_3: float = 1.0,
        weight_4: float = 1.0,
        separator: str = ", ",
    ) -> Tuple[str]:
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
        # Collect non-empty prompts with their weights
        prompts: List[Tuple[str, float]] = []
        for p, w in [
            (prompt_1, weight_1),
            (prompt_2, weight_2),
            (prompt_3, weight_3),
            (prompt_4, weight_4),
        ]:
            if p and p.strip():
                prompts.append((p.strip(), w))

        if not prompts:
            return ("[PromptCombiner] At least one prompt is required.",)

        if len(prompts) == 1:
            return (prompts[0][0],)

        if mode == "blend":
            return (self._blend(prompts),)
        elif mode == "concat":
            return (self._concat(prompts, separator),)
        elif mode == "weighted_average":
            return (self._weighted_average(prompts),)
        else:
            return (f"[PromptCombiner] Unknown mode: {mode}",)

    def _blend(self, prompts: List[Tuple[str, float]]) -> str:
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

    def _concat(self, prompts: List[Tuple[str, float]], separator: str) -> str:
        """Simple concatenation with separator."""
        texts = [p[0] for p in prompts]
        return separator.join(texts)

    def _weighted_average(self, prompts: List[Tuple[str, float]]) -> str:
        """
        Weighted text combination.
        Prompts with higher weights appear earlier and more frequently.
        """
        total_weight = sum(w for _, w in prompts)
        if total_weight == 0:
            return ", ".join(p[0] for p in prompts)

        # Build output with weighted repetition
        parts = []
        for text, weight in prompts:
            # Repeat prompt proportionally to its weight
            repeats = max(1, int((weight / total_weight) * 3))
            for _ in range(repeats):
                parts.append(text)

        # Deduplicate while preserving order
        seen = set()
        result = []
        for part in parts:
            if part not in seen:
                seen.add(part)
                result.append(part)

        return ", ".join(result)
