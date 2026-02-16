"""
Prompt Generator Node for ComfyUI
Generate detailed Stable Diffusion prompts using Qwen3-8B via Ollama
"""

import re
import subprocess
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Optional imports with graceful degradation
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from jinja2 import Environment, BaseLoader

    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

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


def extract_final_prompt(text: str) -> str:
    """
    Extract the final prompt, stripping any thinking/reasoning process.
    Handles Qwen3's "Thinking..." pattern.
    """
    if not text:
        return text

    # Remove Qwen3 thinking blocks: "Thinking...\n...\n...done thinking.\n"
    text = re.sub(
        r"Thinking\.\.\.[\s\S]*?\.\.\.done thinking\.[\s]*", "", text, flags=re.DOTALL
    )

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
    - 7 style presets (cinematic, anime, photorealistic, fantasy, abstract, cyberpunk, sci-fi)
    - Temperature and Top-P sampling controls
    - Optional focus area (emphasis) and mood inputs
    - Reasoning toggle to show/hide model's thinking process
    """

    # Default style templates (fallback when YAML not available)
    DEFAULT_STYLES = {
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
            "template": "Generate a video prompt for: {{ description }}{% if emphasis %} with focus on {{ emphasis }}{% endif %}{% if mood %}, mood is {{ mood }}{% endif %}",
        },
    }

    # Class-level cache for available models
    _cached_models = None
    _cache_time = 0

    def __init__(self):
        """Initialize the node and load style templates."""
        self.style_templates = self._load_templates()

    @classmethod
    def _get_available_models(cls) -> list:
        """
        Fetch available Ollama models with caching.
        Prioritizes LoRA-enhanced models (containing 'lora', 'limbicnation', 'fine').
        """
        # Cache for 60 seconds
        if cls._cached_models and (time.time() - cls._cache_time) < 60:
            return cls._cached_models

        default_models = ["qwen3:8b", "qwen3:4b", "llama3.2:latest"]

        if not OLLAMA_API_AVAILABLE:
            return default_models

        try:
            result = ollama.list()
            models = [
                m.get("model", "") for m in result.get("models", []) if "model" in m
            ]

            if not models:
                return default_models

            # Sort: LoRA/fine-tuned models first, then alphabetically
            lora_keywords = ["lora", "limbicnation", "fine", "style", "prompt"]

            def sort_key(name):
                name_lower = name.lower()
                is_lora = any(kw in name_lower for kw in lora_keywords)
                return (0 if is_lora else 1, name)

            models = sorted(models, key=sort_key)

            cls._cached_models = models
            cls._cache_time = time.time()

            print(f"[PromptGenerator] Found {len(models)} Ollama models")
            return models

        except Exception as e:
            print(f"[PromptGenerator] Could not fetch models: {e}")
            return default_models

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        """Define input parameters for the node."""
        available_models = cls._get_available_models()

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
                    [
                        "cinematic",
                        "anime",
                        "photorealistic",
                        "fantasy",
                        "abstract",
                        "cyberpunk",
                        "sci-fi",
                        "video_wan",
                    ],
                    {"default": "cinematic"},
                ),
                "model": (
                    available_models,
                    {
                        "default": available_models[0]
                        if available_models
                        else "qwen3:8b",
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

    def _load_templates(self) -> Dict[str, Any]:
        """Load style templates from YAML file or use defaults."""
        template_path = Path(__file__).parent.parent / "config" / "templates.yaml"

        if YAML_AVAILABLE and template_path.exists():
            try:
                with open(template_path, "r") as f:
                    templates = yaml.safe_load(f)
                    if templates:
                        print(
                            f"[PromptGenerator] Loaded {len(templates)} styles from templates.yaml"
                        )
                        return templates
            except Exception as e:
                print(f"[PromptGenerator] Warning: Failed to load templates.yaml: {e}")

        print("[PromptGenerator] Using default style templates")
        return self.DEFAULT_STYLES

    def _render_template(
        self,
        style: str,
        description: str,
        emphasis: Optional[str] = None,
        mood: Optional[str] = None,
    ) -> str:
        """Render a Jinja2 template with the given variables."""
        template_data = self.style_templates.get(
            style, self.DEFAULT_STYLES.get("cinematic")
        )

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

    def _check_ollama_health(self, model: str) -> tuple:
        """
        Quick health check: is Ollama running and is the model loaded?
        Returns (is_healthy, message, is_model_loaded).
        """
        if not OLLAMA_API_AVAILABLE:
            return (False, "Ollama API not available", False)

        try:
            ollama.list()
        except Exception as e:
            return (False, f"Ollama server not reachable: {e}", False)

        try:
            ps_response = ollama.ps()
            running_models = [m.model for m in ps_response.models]
            # Check if our model (or a prefix match) is loaded
            is_loaded = any(
                model == rm or model.startswith(rm.split(":")[0])
                for rm in running_models
            )
            if is_loaded:
                return (True, f"Model '{model}' is loaded in VRAM", True)
            else:
                return (
                    True,
                    f"Model '{model}' not loaded (cold start expected)",
                    False,
                )
        except Exception:
            # ps() failed but list() worked - server is up, model status unknown
            return (True, "Ollama running, model status unknown", False)

    def _generate_with_streaming(
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

            # Wrap the iterator so we can enforce per-chunk timeouts
            # using a background thread that advances the iterator.
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
                # Check total timeout
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    print(f"[PromptGenerator] Total timeout ({timeout}s) reached")
                    break

                result_holder.clear()
                t = threading.Thread(target=_iter_next, args=(it,), daemon=True)
                t.start()

                wait_time = (
                    first_chunk_timeout if not got_first_chunk else chunk_timeout
                )
                # Don't wait longer than remaining total timeout
                wait_time = min(wait_time, timeout - elapsed)
                t.join(timeout=wait_time)

                if t.is_alive():
                    label = "first chunk" if not got_first_chunk else "chunk"
                    print(
                        f"[PromptGenerator] Timeout waiting for {label} ({wait_time:.0f}s)"
                    )
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

                    # Update progress bar: 5-95 range for streaming
                    if pbar is not None:
                        progress = min(5 + int(chunk_count * 2), 95)
                        pbar.update_absolute(progress)

        except Exception as e:
            print(f"[PromptGenerator] Streaming error: {e}")
            return None

        if not chunks:
            return None

        elapsed = time.monotonic() - start
        full_text = "".join(chunks)
        print(
            f"[PromptGenerator] Streaming complete: {len(full_text)} chars in {elapsed:.1f}s"
        )
        return full_text

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
        unique_id: str = None,
    ) -> Tuple[str]:
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
            return ("⚠️ Please enter an image description.",)

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

        # Initialize progress bar
        pbar = None
        if COMFY_PROGRESS_AVAILABLE and unique_id is not None:
            try:
                pbar = comfy.utils.ProgressBar(100, node_id=unique_id)
                pbar.update_absolute(0)
            except Exception:
                pbar = None

        # Use Ollama streaming API if available
        if OLLAMA_API_AVAILABLE:
            # Health check and cold-start detection
            effective_timeout = timeout
            is_healthy, health_msg, is_model_loaded = self._check_ollama_health(model)
            print(f"[PromptGenerator] Health: {health_msg}")

            if pbar is not None:
                pbar.update_absolute(5)

            if is_healthy and not is_model_loaded:
                # Cold start: add 30% buffer
                effective_timeout = min(int(timeout * 1.3), 600)
                print(
                    f"[PromptGenerator] Cold start detected, effective timeout: {effective_timeout}s"
                )

            output = self._generate_with_streaming(
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
                    return ("⚠️ Generation returned empty result.",)

            print("[PromptGenerator] Streaming failed, falling back to subprocess")

        # Fallback to subprocess (no temperature/top_p control)
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                return (f"⚠️ Ollama error: {result.stderr}",)

            output = result.stdout.strip()

            if not include_reasoning:
                output = extract_final_prompt(output)

            if pbar is not None:
                pbar.update_absolute(100)

            if output:
                print(
                    f"[PromptGenerator] Generated {len(output)} characters (subprocess)"
                )
                return (output,)
            else:
                return ("⚠️ Generation returned empty result.",)

        except subprocess.TimeoutExpired:
            return (f"⚠️ Generation timed out after {timeout}s",)
        except FileNotFoundError:
            return ("⚠️ Ollama not found. Install from: https://ollama.ai",)
        except Exception as e:
            return (f"⚠️ Error: {str(e)}",)
