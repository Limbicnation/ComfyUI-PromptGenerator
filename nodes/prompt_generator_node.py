"""
Prompt Generator Node for ComfyUI
Generate detailed Stable Diffusion prompts using Qwen3-8B via Ollama
"""

import re
from pathlib import Path
from typing import Any, ClassVar

from .adapters.ollama_client import OllamaClient

# Optional imports with graceful degradation
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from jinja2 import BaseLoader, Environment

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

OLLAMA_API_AVAILABLE = False
COMFY_PROGRESS_AVAILABLE = False

try:
    import importlib.util

    if importlib.util.find_spec("ollama") is not None:
        OLLAMA_API_AVAILABLE = True
except Exception:
    pass

try:
    import importlib.util

    if importlib.util.find_spec("comfy") is not None:
        COMFY_PROGRESS_AVAILABLE = True
except Exception:
    pass


def extract_final_prompt(text: str) -> str:
    """
    Extract the final prompt, stripping any thinking/reasoning process.
    Handles Qwen3's "Thinking..." pattern.
    """
    if not text:
        return text

    # Remove Qwen3 thinking blocks: "Thinking...\n...\n...done thinking.\n"
    text = re.sub(r"Thinking\.\.\.[\s\S]*?\.\.\.done thinking\.[\s]*", "", text, flags=re.DOTALL)

    # Remove common prefixes like "**Prompt:**" or "**Stable Diffusion Prompt:**"
    text = re.sub(r"\*\*(?:Stable Diffusion )?Prompt:\*\*\s*", "", text)
    text = re.sub(r"\*\*Prompt for Image Generation:\*\*\s*", "", text)

    # Remove stray "None" strings often seen in CLI/model leakage
    text = re.sub(r"\s*None\s*", "", text)

    # Clean up any leading/trailing whitespace and quotes
    text = text.strip().strip('"').strip()

    return text


class PromptGeneratorNode:
    """
    ComfyUI node for generating Stable Diffusion prompts using Qwen3-8B via Ollama.

    Features:
    - 9 style presets loaded from templates.yaml (single source of truth)
    - Temperature and Top-P sampling controls
    - Optional focus area (emphasis) and mood inputs
    - Reasoning toggle to show/hide model's thinking process
    """

    CHUNK_TIMEOUT = 30

    # Default style templates (fallback when YAML not available)
    DEFAULT_STYLES: ClassVar[dict[str, dict[str, str]]] = {
        "cinematic": {
            "name": "Cinematic",
            "description": "Dramatic lighting and composition for film-quality images",
            "template": """Write a detailed Stable Diffusion prompt for: {{ description }}

Style: Create a cinematic scene with dramatic lighting and composition.
{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}
{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}

Include specific details about:
- Composition and framing
- Lighting (dramatic, moody)
- Color palette
- Atmosphere and depth
- Technical qualities (8k, high detail)

Format the response as a single, detailed prompt.""",
        },
        "anime": {
            "name": "Anime",
            "description": "Vibrant anime-style illustration with dynamic colors",
            "template": """Write a detailed Stable Diffusion prompt for: {{ description }}

Style: Design an anime-style illustration with vibrant colors and expressive details.
{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}
{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}

Include specific details about:
- Anime art style elements
- Dynamic composition
- Vibrant color palette
- Character expression (if applicable)
- Background and atmosphere

Format the response as a single, detailed prompt in anime style.""",
        },
        "photorealistic": {
            "name": "Photorealistic",
            "description": "High-detail realistic images with natural lighting",
            "template": """Write a detailed Stable Diffusion prompt for: {{ description }}

Style: Generate a photorealistic image with high detail and natural lighting.
{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}
{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}

Include specific details about:
- Realistic textures and materials
- Natural lighting conditions
- Depth of field and focus
- Environmental details
- Camera and lens qualities (e.g., DSLR, 85mm)

Format the response as a single, detailed prompt for photorealistic output.""",
        },
        "fantasy": {
            "name": "Fantasy",
            "description": "Magical elements and themes with ethereal atmosphere",
            "template": """Write a detailed Stable Diffusion prompt for: {{ description }}

Style: Create a fantasy-themed illustration with magical elements.
{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}
{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}

Include specific details about:
- Magical and mystical elements
- Ethereal lighting and glow effects
- Rich fantasy color palette
- Atmospheric depth and wonder
- Intricate details and ornamentation

Format the response as a single, detailed fantasy prompt.""",
        },
        "abstract": {
            "name": "Abstract",
            "description": "Artistic abstract compositions with creative expression",
            "template": """Write a detailed Stable Diffusion prompt for: {{ description }}

Style: Design an abstract artistic composition.
{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}
{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}

Include specific details about:
- Abstract shapes and forms
- Color theory and palette
- Texture and pattern
- Visual rhythm and flow
- Emotional expression

Format the response as a single, detailed abstract art prompt.""",
        },
        "cyberpunk": {
            "name": "Cyberpunk",
            "description": "Neon lights, high technology, and urban dystopia",
            "template": """Write a detailed Stable Diffusion prompt for: {{ description }}

Style: Create a cyberpunk-themed image with neon lights, high technology, and urban dystopia.
{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}
{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}

Include specific details about:
- Neon lighting and reflections
- Futuristic technology elements
- Urban dystopian environment
- Rain/wet surfaces for reflections
- Cybernetic and tech details

Format the response as a single, detailed cyberpunk prompt.""",
        },
        "sci-fi": {
            "name": "Sci-Fi",
            "description": "Futuristic technology and space exploration scenes",
            "template": """Write a detailed Stable Diffusion prompt for: {{ description }}

Style: Generate a science fiction scene with futuristic technology.
{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}
{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}

Include specific details about:
- Futuristic technology and spacecraft
- Space environments or alien worlds
- Advanced materials and surfaces
- Dramatic sci-fi lighting
- Scale and grandeur

Format the response as a single, detailed sci-fi prompt.""",
        },
        "video_wan": {
            "name": "Video (WanVideo)",
            "description": "Minimalist template optimized for WanVideo LoRA",
            "template": (
                "Generate a video prompt for: {{ description }}"
                "{% if emphasis %} with focus on {{ emphasis }}{% endif %}"
                "{% if mood %}, mood is {{ mood }}{% endif %}"
            ),
        },
        "still_image": {
            "name": "Still Image (Photography)",
            "description": "Sharp, realistic photography with technical camera specifications",
            "template": """Write a detailed Stable Diffusion prompt for: {{ description }}

Style: Generate a professional photography still image with sharp focus.
{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}
{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}

Include specific details about:
- Camera settings (ISO 100, f/2.8 aperture, sharp optics)
- Natural or studio lighting
- Realistic textures and fine details
- Clean, balanced composition
- Professional photography qualities

Format the response as a single, detailed photography prompt.""",
        },
    }

    def __init__(self):
        """Initialize the node and load style templates."""
        self.style_templates = self._load_templates()

    @classmethod
    def _get_available_models(cls) -> list:
        """Fetch available Ollama models via OllamaClient adapter."""
        client = OllamaClient(logger_prefix="PromptGenerator")
        return client.discover_models()

    @classmethod
    def _get_style_list(cls) -> list:
        """Get style list from templates.yaml, falling back to DEFAULT_STYLES keys."""
        template_path = Path(__file__).parent.parent / "config" / "templates.yaml"
        if YAML_AVAILABLE and template_path.exists():
            try:
                with open(template_path) as f:
                    templates = yaml.safe_load(f)
                    if templates:
                        return list(templates.keys())
            except Exception as e:
                print(f"[PromptGenerator] Warning: Failed to load style list from templates.yaml: {e}")
        return list(cls.DEFAULT_STYLES.keys())

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, Any]:
        """Define input parameters for the node."""
        available_models = cls._get_available_models()
        available_styles = cls._get_style_list()

        return {
            "required": {
                "description": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "placeholder": "e.g., a mystical forest at twilight",
                    },
                ),
                "style": (
                    available_styles,
                    {"default": available_styles[0] if available_styles else "cinematic"},
                ),
                "model": (
                    available_models,
                    {
                        "default": available_models[0] if available_models else "qwen3:8b",
                        "tooltip": "Select Ollama model. LoRA-enhanced models appear first.",
                    },
                ),
            },
            "optional": {
                "emphasis": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "e.g., lighting, composition, details",
                    },
                ),
                "mood": (
                    "STRING",
                    {
                        "default": "",
                        "placeholder": "e.g., mysterious, serene, dramatic",
                    },
                ),
                "temperature": (
                    "FLOAT",
                    {
                        "default": 0.7,
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
                "include_reasoning": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "label_on": "Show Reasoning",
                        "label_off": "Hide Reasoning",
                    },
                ),
                "timeout": (
                    "INT",
                    {
                        "default": 120,
                        "min": 30,
                        "max": 600,
                        "step": 10,
                        "display": "slider",
                        "tooltip": "Maximum generation time in seconds. Increase for cold model starts.",
                    },
                ),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "generate"
    CATEGORY = "text/generation"
    OUTPUT_NODE = False

    def _load_templates(self) -> dict[str, Any]:
        """Load style templates from YAML file or use defaults."""
        template_path = Path(__file__).parent.parent / "config" / "templates.yaml"

        if YAML_AVAILABLE and template_path.exists():
            try:
                with open(template_path) as f:
                    templates = yaml.safe_load(f)
                    if templates:
                        print(f"[PromptGenerator] Loaded {len(templates)} styles from templates.yaml")
                        return templates
            except Exception as e:
                print(f"[PromptGenerator] Warning: Failed to load templates.yaml: {e}")

        print("[PromptGenerator] Using default style templates")
        return self.DEFAULT_STYLES

    def _render_template(
        self,
        style: str,
        description: str,
        emphasis: str | None = None,
        mood: str | None = None,
    ) -> str:
        """Render a Jinja2 template with the given variables."""
        template_data = self.style_templates.get(style)
        if template_data is None:
            print(f"[PromptGenerator] Warning: style '{style}' not found in templates, falling back to 'cinematic'")
            template_data = self.DEFAULT_STYLES.get("cinematic")

        # Handle YAML format with 'template' key
        if isinstance(template_data, dict) and "template" in template_data:
            template_str = template_data["template"]
        elif isinstance(template_data, str):
            template_str = template_data
        else:
            template_str = self.DEFAULT_STYLES["cinematic"]["template"]

        # Render with Jinja2 if available
        if JINJA2_AVAILABLE:
            env = Environment(loader=BaseLoader())
            template = env.from_string(template_str)
            return template.render(
                description=description,
                emphasis=emphasis if emphasis else None,
                mood=mood if mood else None,
            )
        else:
            # Simple string substitution fallback
            result = template_str.replace("{{ description }}", description)
            if emphasis:
                result = result.replace(
                    "{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}",
                    f"Focus particularly on: {emphasis}",
                )
            else:
                result = re.sub(r"\{% if emphasis %\}.*?\{% endif %\}", "", result)
            if mood:
                result = result.replace(
                    "{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}",
                    f"Mood/Atmosphere: {mood}",
                )
            else:
                result = re.sub(r"\{% if mood %\}.*?\{% endif %\}", "", result)
            return result

    def generate(
        self,
        description: str,
        style: str,
        emphasis: str = "",
        mood: str = "",
        temperature: float = 0.7,
        top_p: float = 0.9,
        include_reasoning: bool = False,
        model: str = "qwen3:8b",
        timeout: int = 120,
        unique_id: str | None = None,
    ) -> tuple[str]:
        """
        Generate a detailed image prompt using Ollama with streaming and progress.

        Args:
            description: Brief description to expand
            style: Style template to use
            emphasis: Optional focus area
            mood: Optional mood/atmosphere
            temperature: Generation temperature (0.1-1.0)
            top_p: Top-p sampling parameter (0.1-1.0)
            include_reasoning: If True, keep reasoning in output
            model: Ollama model to use
            timeout: Maximum generation time in seconds
            unique_id: ComfyUI node ID for progress reporting

        Returns:
            Tuple containing the generated prompt string
        """
        if not description.strip():
            return ("[PromptGenerator] Please enter an image description.",)

        # Render the template
        prompt = self._render_template(
            style,
            description.strip(),
            emphasis.strip() if emphasis else None,
            mood.strip() if mood else None,
        )

        print(
            f"[PromptGenerator] Generating with style='{style}', model='{model}', "
            f"temp={temperature}, top_p={top_p}, timeout={timeout}s"
        )

        # Initialize OllamaClient adapter and progress bar
        client = OllamaClient(logger_prefix="PromptGenerator")
        pbar = client.create_progress_bar(unique_id)

        # Use Ollama streaming API if available
        if OLLAMA_API_AVAILABLE:
            # Health check and cold-start detection
            effective_timeout = timeout
            is_healthy, health_msg, is_model_loaded = client.check_health(model)
            print(f"[PromptGenerator] Health: {health_msg}")

            if pbar is not None:
                pbar.update_absolute(5)

            if is_healthy and not is_model_loaded:
                # Cold start: add 30% buffer
                effective_timeout = min(int(timeout * 1.3), 600)
                print(f"[PromptGenerator] Cold start detected, effective timeout: {effective_timeout}s")

            output = client.generate_streaming(
                model=model,
                prompt=prompt,
                temperature=temperature,
                top_p=top_p,
                timeout=effective_timeout,
                pbar=pbar,
            )

            if output is not None:
                output = output.strip()
                if not include_reasoning:
                    output = extract_final_prompt(output)

                if pbar is not None:
                    pbar.update_absolute(100)

                if output:
                    print(f"[PromptGenerator] Generated {len(output)} characters")
                    return (output,)
                else:
                    return ("[PromptGenerator] Generation returned empty result.",)

            print("[PromptGenerator] Streaming failed, falling back to subprocess")

        # Fallback to subprocess (no temperature/top_p control)
        success, output = client.generate_subprocess(model, prompt, timeout)
        if not success:
            return (f"[PromptGenerator] {output}",)

        if not include_reasoning:
            output = extract_final_prompt(output)

        if pbar is not None:
            pbar.update_absolute(100)

        if output:
            print(f"[PromptGenerator] Generated {len(output)} characters (subprocess)")
            return (output,)
        else:
            return ("[PromptGenerator] Generation returned empty result.",)
