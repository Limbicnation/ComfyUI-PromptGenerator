"""
Style Presets Module for ComfyUI
Modular style system for prompt generation with keyword sets for each style.

Usage in ComfyUI nodes:
    from style_presets import StylePreset

    # Get style choices for INPUT_TYPES dropdown
    choices = StylePreset.get_style_choices()

    # Get style keywords as a prompt string
    preset = StylePreset()
    style_prompt = preset.get_style_prompt("cinematic")
"""

from dataclasses import dataclass, field
from enum import Enum


class StyleMode(Enum):
    """Enumeration of available style modes."""

    CINEMATIC = "cinematic"
    STILL_IMAGE = "still_image"
    ANIME = "anime"
    PHOTOREALISTIC = "photorealistic"
    FANTASY = "fantasy"
    ABSTRACT = "abstract"
    CYBERPUNK = "cyberpunk"
    SCI_FI = "sci-fi"
    VIDEO_WAN = "video_wan"


@dataclass
class StyleKeywords:
    """Container for style-specific keywords and descriptors."""

    primary: list[str] = field(default_factory=list)
    lighting: list[str] = field(default_factory=list)
    technical: list[str] = field(default_factory=list)
    composition: list[str] = field(default_factory=list)
    texture: list[str] = field(default_factory=list)

    def to_prompt_string(self, separator: str = ", ") -> str:
        """Convert all keywords to a single prompt string."""
        all_keywords = self.primary + self.lighting + self.technical + self.composition + self.texture
        return separator.join(all_keywords)

    def to_list(self) -> list[str]:
        """Return all keywords as a flat list (ComfyUI-compatible)."""
        return self.primary + self.lighting + self.technical + self.composition + self.texture


@dataclass
class StyleDefinition:
    """Complete style definition with metadata and keywords."""

    name: str
    description: str
    keywords: StyleKeywords


# Style definitions - single source of truth
STYLE_DEFINITIONS: dict[StyleMode, StyleDefinition] = {
    StyleMode.CINEMATIC: StyleDefinition(
        name="Cinematic",
        description="Film-like visuals with dramatic lighting and anamorphic qualities",
        keywords=StyleKeywords(
            primary=["cinematic shot", "film grain", "movie still", "dramatic scene"],
            lighting=[
                "dramatic lighting",
                "volumetric light",
                "rim lighting",
                "chiaroscuro",
                "golden hour",
            ],
            technical=[
                "anamorphic lens",
                "shallow depth of field",
                "bokeh",
                "35mm film",
                "wide aspect ratio",
            ],
            composition=[
                "rule of thirds",
                "leading lines",
                "dynamic composition",
                "cinematic framing",
            ],
            texture=["rich color grading", "film texture", "atmospheric haze"],
        ),
    ),
    StyleMode.STILL_IMAGE: StyleDefinition(
        name="Still Image (Photography)",
        description="Sharp, realistic photography with technical camera specifications",
        keywords=StyleKeywords(
            primary=[
                "professional photography",
                "high resolution",
                "sharp focus",
                "studio quality",
            ],
            lighting=[
                "natural lighting",
                "soft diffused light",
                "studio lighting",
                "balanced exposure",
            ],
            technical=[
                "f/2.8 aperture",
                "ISO 100",
                "sharp optics",
                "full frame sensor",
                "RAW quality",
            ],
            composition=[
                "centered composition",
                "clean framing",
                "balanced layout",
                "professional angle",
            ],
            texture=[
                "realistic textures",
                "fine details",
                "crisp definition",
                "accurate colors",
            ],
        ),
    ),
    StyleMode.ANIME: StyleDefinition(
        name="Anime",
        description="Vibrant anime-style illustration with dynamic colors",
        keywords=StyleKeywords(
            primary=[
                "anime style",
                "manga illustration",
                "cel shading",
                "vibrant colors",
            ],
            lighting=[
                "soft glow",
                "dramatic backlighting",
                "ambient light",
                "bloom effect",
            ],
            technical=[
                "clean linework",
                "flat color areas",
                "high contrast",
                "detailed eyes",
            ],
            composition=[
                "dynamic pose",
                "expressive composition",
                "layered background",
            ],
            texture=["smooth gradients", "soft skin tones", "vivid saturation"],
        ),
    ),
    StyleMode.PHOTOREALISTIC: StyleDefinition(
        name="Photorealistic",
        description="High-detail realistic images with natural lighting",
        keywords=StyleKeywords(
            primary=[
                "photorealistic",
                "ultra realistic",
                "lifelike detail",
                "high fidelity",
            ],
            lighting=[
                "natural lighting",
                "golden hour",
                "soft shadows",
                "ambient occlusion",
            ],
            technical=[
                "DSLR quality",
                "85mm lens",
                "shallow depth of field",
                "8K resolution",
            ],
            composition=["rule of thirds", "natural framing", "environmental portrait"],
            texture=[
                "realistic skin texture",
                "fine material detail",
                "natural imperfections",
            ],
        ),
    ),
    StyleMode.FANTASY: StyleDefinition(
        name="Fantasy",
        description="Magical elements and themes with ethereal atmosphere",
        keywords=StyleKeywords(
            primary=["fantasy art", "magical scene", "mythical", "enchanted"],
            lighting=[
                "ethereal glow",
                "mystical light rays",
                "bioluminescence",
                "aurora",
            ],
            technical=["digital painting", "matte painting", "concept art quality"],
            composition=[
                "epic scale",
                "layered depth",
                "grand vista",
                "ornate framing",
            ],
            texture=[
                "iridescent surfaces",
                "crystalline detail",
                "ancient stonework",
                "enchanted flora",
            ],
        ),
    ),
    StyleMode.ABSTRACT: StyleDefinition(
        name="Abstract",
        description="Artistic abstract compositions with creative expression",
        keywords=StyleKeywords(
            primary=["abstract art", "non-representational", "artistic composition"],
            lighting=["color field lighting", "gradient transitions", "luminous forms"],
            technical=["mixed media", "generative art", "high dynamic range"],
            composition=[
                "visual rhythm",
                "asymmetric balance",
                "flowing forms",
                "geometric patterns",
            ],
            texture=[
                "paint strokes",
                "textured layers",
                "organic forms",
                "splatter effects",
            ],
        ),
    ),
    StyleMode.CYBERPUNK: StyleDefinition(
        name="Cyberpunk",
        description="Neon lights, high technology, and urban dystopia",
        keywords=StyleKeywords(
            primary=[
                "cyberpunk",
                "neon noir",
                "dystopian future",
                "high tech low life",
            ],
            lighting=[
                "neon glow",
                "holographic reflections",
                "LED strips",
                "rain-soaked neon",
            ],
            technical=[
                "ray tracing",
                "volumetric fog",
                "chromatic aberration",
                "lens flare",
            ],
            composition=[
                "urban canyon",
                "vertical composition",
                "dense layering",
                "vanishing point",
            ],
            texture=[
                "wet asphalt reflections",
                "rust and chrome",
                "holographic surfaces",
                "circuit patterns",
            ],
        ),
    ),
    StyleMode.SCI_FI: StyleDefinition(
        name="Sci-Fi",
        description="Futuristic technology and space exploration scenes",
        keywords=StyleKeywords(
            primary=[
                "science fiction",
                "futuristic",
                "space opera",
                "advanced technology",
            ],
            lighting=["starlight", "plasma glow", "engine flare", "atmospheric entry"],
            technical=[
                "hard surface modeling",
                "concept art",
                "matte painting",
                "photobashing",
            ],
            composition=[
                "epic scale",
                "orbital view",
                "dramatic perspective",
                "vast emptiness",
            ],
            texture=[
                "polished metal",
                "energy fields",
                "alien materials",
                "hull plating",
            ],
        ),
    ),
    StyleMode.VIDEO_WAN: StyleDefinition(
        name="Video (WanVideo)",
        description="Minimalist keywords optimized for WanVideo LoRA",
        keywords=StyleKeywords(
            primary=["video shot", "motion capture", "fluid movement"],
            lighting=["cinematic lighting", "natural light"],
            technical=["smooth motion", "24fps", "high resolution video"],
            composition=["dynamic camera", "tracking shot"],
            texture=["temporal consistency", "clean frames"],
        ),
    ),
}


def get_style_keywords(style: str) -> StyleKeywords:
    """Get keywords for a specific style."""
    try:
        mode = StyleMode(style.lower())
        return STYLE_DEFINITIONS[mode].keywords
    except ValueError as err:
        available = [s.value for s in StyleMode]
        raise ValueError(f"Unknown style '{style}'. Available: {', '.join(available)}") from err


def get_available_styles() -> list[str]:
    """Get list of available style names."""
    return [mode.value for mode in StyleMode]


class StylePreset:
    """ComfyUI-compatible style preset class."""

    def __init__(self):
        self._styles = STYLE_DEFINITIONS

    @staticmethod
    def get_style_choices() -> tuple[str, ...]:
        """Get available style choices as tuple (ComfyUI dropdown format)."""
        return tuple(mode.value for mode in StyleMode)

    def get_style_keywords(self, style: str) -> list[str]:
        """Get style keywords as a list."""
        return get_style_keywords(style).to_list()

    def get_style_prompt(self, style: str, emphasis: str | None = None, include_technical: bool = True) -> str:
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
    "STYLE_DEFINITIONS",
    "StyleDefinition",
    "StyleKeywords",
    "StyleMode",
    "StylePreset",
    "get_available_styles",
    "get_style_keywords",
]
