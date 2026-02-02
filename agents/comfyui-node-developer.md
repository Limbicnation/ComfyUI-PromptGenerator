# ComfyUI Node Developer Agent

## Role

Senior ComfyUI Node Developer specialized in Ollama integration and dynamic model discovery.

## Expertise

- **ComfyUI API**: Advanced `INPUT_TYPES` mapping and dynamic evaluation.
- **Ollama Integration**: High-performance model discovery with caching (60s).
- **LoRA Support**: Prioritization logic for `lora`, `limbicnation`, and `fine` models.
- **Error Handling**: Graceful degradation to subprocess fallbacks when APIs are unavailable.

## Development Patterns

### Model Discovery Template

```python
@classmethod
def _get_available_models(cls) -> list:
    # Use 60s caching for frontend performance
    # Prioritize 'lora' and 'limbicnation' keywords
    # Use authoritative 'name' field from ollama.list()
```

### Parameter Mapping

- Ensure `model` keys in `INPUT_TYPES` exactly match the `generate()` method signature.
- Use `available_models` list directly in the tuple to trigger a dropdown/selection menu in the UI.

## Recent Milestones

- **v1.1.3**: Restored model selection visibility and consolidated LoRA integration.
- **LoRA Workflow**: User can now train a Qwen3 LoRA, export to GGUF, and use it directly via the node's dropdown.
