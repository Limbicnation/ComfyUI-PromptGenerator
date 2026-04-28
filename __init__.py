"""
ComfyUI Prompt Generator Node
Generate Stable Diffusion prompts using Qwen3-8B via Ollama
"""

try:
    from .nodes.prompt_generator_node import PromptGeneratorNode
    from .nodes.style_applier_node import StyleApplierNode
    from .nodes.prompt_refiner_node import PromptRefinerNode
    from .nodes.negative_prompt_node import NegativePromptNode
    from .nodes.prompt_combiner_node import PromptCombinerNode
except ImportError:
    # Fallback for test environments where ComfyUI isn't present
    from nodes.prompt_generator_node import PromptGeneratorNode
    from nodes.style_applier_node import StyleApplierNode
    from nodes.prompt_refiner_node import PromptRefinerNode
    from nodes.negative_prompt_node import NegativePromptNode
    from nodes.prompt_combiner_node import PromptCombinerNode

NODE_CLASS_MAPPINGS = {
    "Limbicnation_PromptGenerator": PromptGeneratorNode,
    "Limbicnation_StyleApplier": StyleApplierNode,
    "Limbicnation_PromptRefiner": PromptRefinerNode,
    "Limbicnation_NegativePrompt": NegativePromptNode,
    "Limbicnation_PromptCombiner": PromptCombinerNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Limbicnation_PromptGenerator": "Prompt Generator (Qwen)",
    "Limbicnation_StyleApplier": "Style Applier",
    "Limbicnation_PromptRefiner": "Prompt Refiner",
    "Limbicnation_NegativePrompt": "Negative Prompt Generator",
    "Limbicnation_PromptCombiner": "Prompt Combiner",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
