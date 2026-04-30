"""
ComfyUI Prompt Generator Node
Generate Stable Diffusion prompts using Qwen3-8B via Ollama
"""

try:
    from .nodes.negative_prompt_node import NegativePromptNode
    from .nodes.prompt_combiner_node import PromptCombinerNode
    from .nodes.prompt_generator_node import PromptGeneratorNode
    from .nodes.prompt_refiner_node import PromptRefinerNode
    from .nodes.style_applier_node import StyleApplierNode
except ImportError:
    # Fallback for test environments where ComfyUI isn't present
    from nodes.negative_prompt_node import NegativePromptNode
    from nodes.prompt_combiner_node import PromptCombinerNode
    from nodes.prompt_generator_node import PromptGeneratorNode
    from nodes.prompt_refiner_node import PromptRefinerNode
    from nodes.style_applier_node import StyleApplierNode

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

# Fail loudly at import time if mapping drift occurs — silent omission would just
# remove nodes from the ComfyUI menu without any error in the console.
_missing_display = set(NODE_CLASS_MAPPINGS) - set(NODE_DISPLAY_NAME_MAPPINGS)
_orphan_display = set(NODE_DISPLAY_NAME_MAPPINGS) - set(NODE_CLASS_MAPPINGS)
if _missing_display or _orphan_display:
    raise RuntimeError(
        "Node mapping mismatch — "
        f"missing display names: {sorted(_missing_display)}; "
        f"orphan display names: {sorted(_orphan_display)}"
    )

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
