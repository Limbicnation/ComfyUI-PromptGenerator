"""
ComfyUI Prompt Generator Node
Generate Stable Diffusion prompts using Qwen3-8B via Ollama
"""

from .nodes.prompt_generator_node import PromptGeneratorNode

NODE_CLASS_MAPPINGS = {
    "PromptGenerator": PromptGeneratorNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator": "🎨 Prompt Generator (Qwen)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
