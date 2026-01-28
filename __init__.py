"""
ComfyUI Prompt Generator Node
Generate Stable Diffusion prompts using Qwen3-8B via Ollama
"""

from .nodes.prompt_generator_node import PromptGeneratorNode
from .nodes.style_applier_node import StyleApplierNode

NODE_CLASS_MAPPINGS = {
    "PromptGenerator": PromptGeneratorNode,
    "StyleApplier": StyleApplierNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator": "🎨 Prompt Generator (Qwen)",
    "StyleApplier": "🎬 Style Applier (Cinematic/Photo)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

