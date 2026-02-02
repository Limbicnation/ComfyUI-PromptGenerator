"""
Prompt Generator Node for ComfyUI
Generate detailed Stable Diffusion prompts using Qwen3-8B via Ollama
"""

import os
import re
import subprocess
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


def extract_final_prompt(text: str) -> str:
    """
    Extract the final prompt, stripping any thinking/reasoning process.
    Handles Qwen3's "Thinking..." pattern.
    """
    if not text:
        return text
    
    # Remove Qwen3 thinking blocks: "Thinking...\n...\n...done thinking.\n"
    text = re.sub(r'Thinking\.\.\.[\s\S]*?\.\.\.done thinking\.[\s]*', '', text, flags=re.DOTALL)
    
    # Remove common prefixes like "**Prompt:**" or "**Stable Diffusion Prompt:**"
    text = re.sub(r'\*\*(?:Stable Diffusion )?Prompt:\*\*\s*', '', text)
    text = re.sub(r'\*\*Prompt for Image Generation:\*\*\s*', '', text)
    
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

Format the response as a single, detailed prompt."""
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

Format the response as a single, detailed prompt in anime style."""
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

Format the response as a single, detailed prompt for photorealistic output."""
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

Format the response as a single, detailed fantasy prompt."""
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

Format the response as a single, detailed abstract art prompt."""
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

Format the response as a single, detailed cyberpunk prompt."""
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

Format the response as a single, detailed sci-fi prompt."""
        }
    }
    
    # Class-level cache for available models
    _cached_models = None
    _cache_time = 0
    
    def __init__(self):
        """Initialize the node and load style templates."""
        self.style_templates = self._load_templates()
        self.timeout = 120
    
    @classmethod
    def _get_available_models(cls) -> list:
        """
        Fetch available Ollama models with caching.
        Prioritizes LoRA-enhanced models (containing 'lora', 'limbicnation', 'fine').
        """
        import time
        
        # Cache for 60 seconds
        if cls._cached_models and (time.time() - cls._cache_time) < 60:
            return cls._cached_models
        
        default_models = ["qwen3:8b", "qwen3:4b", "llama3.2:latest"]
        
        if not OLLAMA_API_AVAILABLE:
            return default_models
        
        try:
            result = ollama.list()
            models = [m.get('model', '') for m in result.get('models', []) if 'model' in m]
            
            if not models:
                return default_models
            
            # Sort: LoRA/fine-tuned models first, then alphabetically
            lora_keywords = ['lora', 'limbicnation', 'fine', 'style', 'prompt']
            
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
                "description": ("STRING", {
                    "multiline": True,
                    "default": "",
                    "placeholder": "e.g., a mystical forest at twilight"
                }),
                "style": (["cinematic", "anime", "photorealistic", "fantasy", 
                           "abstract", "cyberpunk", "sci-fi"], {
                    "default": "cinematic"
                }),
                "model": (available_models, {
                    "default": available_models[0] if available_models else "qwen3:8b",
                    "tooltip": "Select Ollama model. LoRA-enhanced models appear first."
                }),
            },
            "optional": {
                "emphasis": ("STRING", {
                    "default": "",
                    "placeholder": "e.g., lighting, composition, details"
                }),
                "mood": ("STRING", {
                    "default": "",
                    "placeholder": "e.g., mysterious, serene, dramatic"
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.1,
                    "max": 1.0,
                    "step": 0.1,
                    "display": "slider"
                }),
                "include_reasoning": ("BOOLEAN", {
                    "default": False,
                    "label_on": "Show Reasoning",
                    "label_off": "Hide Reasoning"
                }),
            }
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
                with open(template_path, 'r') as f:
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
        emphasis: Optional[str] = None,
        mood: Optional[str] = None
    ) -> str:
        """Render a Jinja2 template with the given variables."""
        template_data = self.style_templates.get(style, self.DEFAULT_STYLES.get("cinematic"))
        
        # Handle YAML format with 'template' key
        if isinstance(template_data, dict) and 'template' in template_data:
            template_str = template_data['template']
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
                mood=mood if mood else None
            )
        else:
            # Simple string substitution fallback
            result = template_str.replace("{{ description }}", description)
            if emphasis:
                result = result.replace(
                    "{% if emphasis %}Focus particularly on: {{ emphasis }}{% endif %}",
                    f"Focus particularly on: {emphasis}"
                )
            else:
                result = re.sub(r'\{% if emphasis %\}.*?\{% endif %\}', '', result)
            if mood:
                result = result.replace(
                    "{% if mood %}Mood/Atmosphere: {{ mood }}{% endif %}",
                    f"Mood/Atmosphere: {mood}"
                )
            else:
                result = re.sub(r'\{% if mood %\}.*?\{% endif %\}', '', result)
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
        model: str = "qwen3:8b"
    ) -> Tuple[str]:
        """
        Generate a detailed image prompt using Qwen3-8B via Ollama.
        
        Args:
            description: Brief description to expand
            style: Style template to use
            emphasis: Optional focus area
            mood: Optional mood/atmosphere
            temperature: Generation temperature (0.1-1.0)
            top_p: Top-p sampling parameter (0.1-1.0)
            include_reasoning: If True, keep reasoning in output
            model: Ollama model to use
            
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
            mood.strip() if mood else None
        )
        
        print(f"[PromptGenerator] Generating with style='{style}', model='{model}', temp={temperature}, top_p={top_p}")
        
        # Use Ollama Python API if available (supports temperature/top_p)
        if OLLAMA_API_AVAILABLE:
            try:
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options={
                        "temperature": temperature,
                        "top_p": top_p
                    }
                )
                output = response.get('response', '').strip()
                
                if not include_reasoning:
                    output = extract_final_prompt(output)
                
                if output:
                    print(f"[PromptGenerator] Generated {len(output)} characters")
                    return (output,)
                else:
                    return ("⚠️ Generation returned empty result.",)
                    
            except Exception as e:
                print(f"[PromptGenerator] Ollama API error: {e}, falling back to subprocess")
        
        # Fallback to subprocess (no temperature/top_p control)
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=self.timeout
            )
            
            if result.returncode != 0:
                return (f"⚠️ Ollama error: {result.stderr}",)
            
            output = result.stdout.strip()
            
            if not include_reasoning:
                output = extract_final_prompt(output)
            
            if output:
                print(f"[PromptGenerator] Generated {len(output)} characters (subprocess)")
                return (output,)
            else:
                return ("⚠️ Generation returned empty result.",)
                
        except subprocess.TimeoutExpired:
            return (f"⚠️ Generation timed out after {self.timeout}s",)
        except FileNotFoundError:
            return ("⚠️ Ollama not found. Install from: https://ollama.ai",)
        except Exception as e:
            return (f"⚠️ Error: {str(e)}",)
