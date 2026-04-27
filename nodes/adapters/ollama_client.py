"""
Ollama Client Adapter for ComfyUI-PromptGenerator
Extracted from PromptGeneratorNode to follow SOLID single-responsibility.

Handles:
- Health checks (server reachability, model loaded status)
- Streaming generation with per-chunk and total timeout enforcement
- Model discovery with caching and LoRA prioritization
- Subprocess fallback when Python API unavailable
"""

import re
import subprocess
import time
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Optional imports with graceful degradation
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


class OllamaClient:
    """
    Adapter for Ollama LLM backend.

    Supports:
    - Streaming generation with timeout enforcement
    - Model discovery with LoRA prioritization
    - Health checks with cold-start detection
    - Subprocess fallback
    """

    CHUNK_TIMEOUT = 30
    DEFAULT_MODELS = ["qwen3:8b", "qwen3:4b", "llama3.2:latest"]
    LORA_KEYWORDS = ["lora", "limbicnation", "fine", "style", "prompt"]

    # Class-level cache for available models
    _cached_models: Optional[List[str]] = None
    _cache_time = 0.0

    def __init__(self, logger_prefix: str = "OllamaClient"):
        self.logger_prefix = logger_prefix

    def _log(self, message: str) -> None:
        print(f"[{self.logger_prefix}] {message}")

    def discover_models(self) -> List[str]:
        """
        Fetch available Ollama models with caching.
        Prioritizes LoRA-enhanced models.
        """
        if self._cached_models and (time.time() - self._cache_time) < 60:
            return self._cached_models

        if not OLLAMA_API_AVAILABLE:
            return self.DEFAULT_MODELS

        try:
            result = ollama.list()
            models = [
                m.get("model", "") for m in result.get("models", []) if "model" in m
            ]

            if not models:
                return self.DEFAULT_MODELS

            def sort_key(name: str) -> Tuple[int, str]:
                name_lower = name.lower()
                is_lora = any(kw in name_lower for kw in self.LORA_KEYWORDS)
                return (0 if is_lora else 1, name)

            models = sorted(models, key=sort_key)

            self._cached_models = models
            self._cache_time = time.time()

            self._log(f"Found {len(models)} Ollama models")
            return models

        except Exception as e:
            self._log(f"Could not fetch models: {e}")
            return self.DEFAULT_MODELS

    def check_health(self, model: str) -> Tuple[bool, str, bool]:
        """
        Quick health check: is Ollama running and is the model loaded?

        Returns:
            (is_healthy, message, is_model_loaded)
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
            return (True, "Ollama running, model status unknown", False)

    def generate_streaming(
        self,
        model: str,
        prompt: str,
        temperature: float,
        top_p: float,
        timeout: int,
        pbar: Optional[Any] = None,
    ) -> Optional[str]:
        """
        Stream ollama.generate() with per-chunk and total timeout enforcement.

        Returns:
            Full response text, or None on failure.
        """
        if not OLLAMA_API_AVAILABLE:
            return None

        chunks: List[str] = []
        start = time.monotonic()
        first_chunk_timeout = min(timeout * 0.6, 90)
        chunk_timeout = self.CHUNK_TIMEOUT
        got_first_chunk = False

        try:
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                stream=True,
                options={"temperature": temperature, "top_p": top_p},
            )

            result_holder: Dict[str, Any] = {}

            def _iter_next(it):
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
                elapsed = time.monotonic() - start
                if elapsed >= timeout:
                    self._log(f"Total timeout ({timeout}s) reached")
                    break

                result_holder.clear()
                t = threading.Thread(target=_iter_next, args=(it,), daemon=True)
                t.start()

                wait_time = first_chunk_timeout if not got_first_chunk else chunk_timeout
                wait_time = min(wait_time, timeout - elapsed)
                t.join(timeout=wait_time)

                if t.is_alive():
                    label = "first chunk" if not got_first_chunk else "chunk"
                    self._log(f"Timeout waiting for {label} ({wait_time:.0f}s)")
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

                    if pbar is not None:
                        progress = min(5 + int(chunk_count * 2), 95)
                        pbar.update_absolute(progress)

        except Exception as e:
            self._log(f"Streaming error: {e}")
            return None

        if not chunks:
            return None

        elapsed = time.monotonic() - start
        full_text = "".join(chunks)
        self._log(f"Streaming complete: {len(full_text)} chars in {elapsed:.1f}s")
        return full_text

    def generate_subprocess(
        self,
        model: str,
        prompt: str,
        timeout: int,
    ) -> Tuple[bool, str]:
        """
        Fallback generation via subprocess call to ollama CLI.

        Returns:
            (success, output_or_error_message)
        """
        try:
            result = subprocess.run(
                ["ollama", "run", model, prompt],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            if result.returncode != 0:
                return (False, f"Ollama error: {result.stderr}")

            output = result.stdout.strip()
            if output:
                self._log(f"Generated {len(output)} characters (subprocess)")
                return (True, output)
            else:
                return (False, "Generation returned empty result.")

        except subprocess.TimeoutExpired:
            return (False, f"Generation timed out after {timeout}s")
        except FileNotFoundError:
            return (False, "Ollama not found. Install from: https://ollama.ai")
        except Exception as e:
            return (False, f"Error: {str(e)}")

    def create_progress_bar(self, unique_id: Optional[str] = None) -> Optional[Any]:
        """Create a ComfyUI progress bar if available."""
        if COMFY_PROGRESS_AVAILABLE and unique_id is not None:
            try:
                pbar = comfy.utils.ProgressBar(100, node_id=unique_id)
                pbar.update_absolute(0)
                return pbar
            except Exception:
                pass
        return None
