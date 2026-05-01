"""
ComfyUI Prompt Generator Node
Generate Stable Diffusion prompts using Qwen3-8B via Ollama
"""

from .nodes.prompt_generator_node import PromptGeneratorNode
from .nodes.style_applier_node import StyleApplierNode
from .nodes.prompt_combiner_node import PromptCombinerNode
from .nodes.prompt_refiner_node import PromptRefinerNode
from .nodes.negative_prompt_node import NegativePromptNode

NODE_CLASS_MAPPINGS = {
    "PromptGenerator": PromptGeneratorNode,
    "StyleApplier": StyleApplierNode,
    "PromptCombiner": PromptCombinerNode,
    "PromptRefiner": PromptRefinerNode,
    "NegativePrompt": NegativePromptNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator": "🎨 Prompt Generator (Qwen)",
    "StyleApplier": "🎬 Style Applier (Cinematic/Photo)",
    "PromptCombiner": "🔗 Prompt Combiner",
    "PromptRefiner": "✨ Prompt Refiner",
    "NegativePrompt": "⛔ Negative Prompt Generator",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

