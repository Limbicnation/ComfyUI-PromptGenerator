"""
ComfyUI Prompt Generator Node
Generate Stable Diffusion prompts using Qwen3-8B via Ollama
"""

from .nodes.prompt_generator_node import PromptGeneratorNode
from .nodes.style_applier_node import StyleApplierNode

NODE_CLASS_MAPPINGS = {
    "Limbicnation_PromptGenerator": PromptGeneratorNode,
    "Limbicnation_StyleApplier": StyleApplierNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Limbicnation_PromptGenerator": "Prompt Generator (Qwen)",
    "Limbicnation_StyleApplier": "Style Applier",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
