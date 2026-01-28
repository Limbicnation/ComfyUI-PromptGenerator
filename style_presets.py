"""
Style Presets Module for ComfyUI
Modular style system for prompt generation with Cinematic and Still Image modes.

Usage in ComfyUI nodes:
    from style_presets import StylePreset
    
    # Get style choices for INPUT_TYPES dropdown
    choices = StylePreset.get_style_choices()  # ("cinematic", "still_image")
    
    # Get style keywords as a prompt string
    preset = StylePreset()
    style_prompt = preset.get_style_prompt("cinematic")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class StyleMode(Enum):
    """Enumeration of available style modes."""
    CINEMATIC = "cinematic"
    STILL_IMAGE = "still_image"


@dataclass
class StyleKeywords:
    """Container for style-specific keywords and descriptors."""
    primary: List[str] = field(default_factory=list)
    lighting: List[str] = field(default_factory=list)
    technical: List[str] = field(default_factory=list)
    composition: List[str] = field(default_factory=list)
    texture: List[str] = field(default_factory=list)
    
    def to_prompt_string(self, separator: str = ", ") -> str:
        """Convert all keywords to a single prompt string."""
        all_keywords = (
            self.primary + self.lighting + self.technical + 
            self.composition + self.texture
        )
        return separator.join(all_keywords)
    
    def to_list(self) -> List[str]:
        """Return all keywords as a flat list (ComfyUI-compatible)."""
        return (
            self.primary + self.lighting + self.technical + 
            self.composition + self.texture
        )


@dataclass
class StyleDefinition:
    """Complete style definition with metadata and keywords."""
    name: str
    description: str
    keywords: StyleKeywords


# Style definitions - single source of truth
STYLE_DEFINITIONS: Dict[StyleMode, StyleDefinition] = {
    StyleMode.CINEMATIC: StyleDefinition(
        name="Cinematic",
        description="Film-like visuals with dramatic lighting and anamorphic qualities",
        keywords=StyleKeywords(
            primary=["cinematic shot", "film grain", "movie still", "dramatic scene"],
            lighting=["dramatic lighting", "volumetric light", "rim lighting", "chiaroscuro", "golden hour"],
            technical=["anamorphic lens", "shallow depth of field", "bokeh", "35mm film", "wide aspect ratio"],
            composition=["rule of thirds", "leading lines", "dynamic composition", "cinematic framing"],
            texture=["rich color grading", "film texture", "atmospheric haze"]
        )
    ),
    StyleMode.STILL_IMAGE: StyleDefinition(
        name="Still Image (Photography)",
        description="Sharp, realistic photography with technical camera specifications",
        keywords=StyleKeywords(
            primary=["professional photography", "high resolution", "sharp focus", "studio quality"],
            lighting=["natural lighting", "soft diffused light", "studio lighting", "balanced exposure"],
            technical=["f/2.8 aperture", "ISO 100", "sharp optics", "full frame sensor", "RAW quality"],
            composition=["centered composition", "clean framing", "balanced layout", "professional angle"],
            texture=["realistic textures", "fine details", "crisp definition", "accurate colors"]
        )
    )
}


def get_style_keywords(style: str) -> StyleKeywords:
    """Get keywords for a specific style."""
    try:
        mode = StyleMode(style.lower())
        return STYLE_DEFINITIONS[mode].keywords
    except ValueError:
        available = [s.value for s in StyleMode]
        raise ValueError(f"Unknown style '{style}'. Available: {', '.join(available)}")


def get_available_styles() -> List[str]:
    """Get list of available style names."""
    return [mode.value for mode in StyleMode]


class StylePreset:
    """ComfyUI-compatible style preset class."""
    
    def __init__(self):
        self._styles = STYLE_DEFINITIONS
    
    @staticmethod
    def get_style_choices() -> Tuple[str, ...]:
        """Get available style choices as tuple (ComfyUI dropdown format)."""
        return tuple(mode.value for mode in StyleMode)
    
    def get_style_keywords(self, style: str) -> List[str]:
        """Get style keywords as a list."""
        return get_style_keywords(style).to_list()
    
    def get_style_prompt(
        self,
        style: str,
        emphasis: Optional[str] = None,
        include_technical: bool = True
    ) -> str:
        """Get a formatted style prompt string."""
        keywords = get_style_keywords(style)
        
        parts = []
        parts.extend(keywords.primary)
        parts.extend(keywords.lighting)
        parts.extend(keywords.composition)
        parts.extend(keywords.texture)
        
        if include_technical:
            parts.extend(keywords.technical)
        
        if emphasis:
            parts.insert(0, f"emphasis on {emphasis}")
        
        return ", ".join(parts)


__all__ = [
    "StyleMode",
    "StyleKeywords",
    "StyleDefinition",
    "StylePreset",
    "get_style_keywords",
    "get_available_styles",
    "STYLE_DEFINITIONS"
]

