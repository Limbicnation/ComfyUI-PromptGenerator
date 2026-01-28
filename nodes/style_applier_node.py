"""
Style Applier Node for ComfyUI
Applies Cinematic or Still Image style keywords to prompts.
"""

from typing import Tuple


class StyleApplierNode:
    """
    ComfyUI node for applying Cinematic or Still Image style keywords to prompts.
    
    Inputs:
        - prompt: Base prompt text
        - style: "cinematic" or "still_image"
        - position: Where to add keywords ("prefix", "suffix", or "wrap")
        - emphasis: Optional emphasis level ("low", "medium", "high")
        - include_technical: Include camera/technical specs
    
    Outputs:
        - styled_prompt: The prompt with style keywords added
        - style_keywords: Just the style keywords (for reference)
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node."""
        from style_presets import StylePreset
        
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "Enter your base prompt..."
                }),
                "style": (StylePreset.get_style_choices(), {"default": "cinematic"}),
            },
            "optional": {
                "position": (["suffix", "prefix", "wrap"], {"default": "suffix"}),
                "emphasis": (["medium", "low", "high"], {"default": "medium"}),
                "include_technical": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("styled_prompt", "style_keywords",)
    FUNCTION = "apply_style"
    CATEGORY = "text/generation"
    
    def apply_style(
        self,
        prompt: str,
        style: str,
        position: str = "suffix",
        emphasis: str = "medium",
        include_technical: bool = True
    ) -> Tuple[str, str]:
        """Apply style keywords to a prompt."""
        from style_presets import StylePreset

        # Normalize inputs
        prompt = prompt.strip() if prompt else ""
        style = style.strip() if style else "cinematic"

        # Validate style
        available_styles = StylePreset.get_style_choices()
        if style not in available_styles:
            return (f"[StyleApplier] Error: Unknown style '{style}'. Available: {available_styles}", "")

        # Get style keywords
        try:
            style_keywords = StylePreset().get_style_prompt(
                style=style,
                emphasis=emphasis,
                include_technical=include_technical
            )
        except ValueError as e:
            return (f"[StyleApplier] Error getting style: {e}", "")

        # Handle empty prompt
        if not prompt:
            return (style_keywords, style_keywords)

        # Apply style based on position
        if position == "prefix":
            styled_prompt = f"{style_keywords}, {prompt}"
        elif position == "suffix":
            styled_prompt = f"{prompt}, {style_keywords}"
        else:  # wrap
            styled_prompt = f"{style_keywords}, {prompt}, {style_keywords}"

        return (styled_prompt, style_keywords)
