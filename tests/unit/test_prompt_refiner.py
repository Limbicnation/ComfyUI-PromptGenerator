"""Unit tests for PromptRefinerNode seed behavior."""

from unittest.mock import patch

from nodes.prompt_refiner_node import PromptRefinerNode


class TestPromptRefinerSeed:
    """Test suite for per-pass seed derivation."""

    def test_per_pass_seed_increments(self):
        """Each pass should receive an incremented seed when seed != -1."""
        node = PromptRefinerNode()
        captured_seeds: list[int | None] = []

        def _capture_seed(*, model, prompt, temperature, top_p, timeout, pbar, seed):
            captured_seeds.append(seed)
            return f"refined with seed={seed}"

        with (
            patch.object(node, "REFINEMENT_PROMPT", "{prompt}"),
            patch(
                "nodes.prompt_refiner_node.OllamaClient.generate_streaming",
                side_effect=_capture_seed,
            ),
        ):
            result = node.refine(
                prompt="a forest",
                model="qwen3:8b",
                passes=3,
                seed=42,
                temperature=0.5,
                top_p=0.9,
                timeout=30,
            )

        assert captured_seeds == [42, 43, 44]
        assert "refined with seed=44" in result[0]

    def test_random_seed_none_for_all_passes(self):
        """When seed is -1, all passes should receive None (random)."""
        node = PromptRefinerNode()
        captured_seeds: list[int | None] = []

        def _capture_seed(*, model, prompt, temperature, top_p, timeout, pbar, seed):
            captured_seeds.append(seed)
            return "refined"

        with (
            patch.object(node, "REFINEMENT_PROMPT", "{prompt}"),
            patch(
                "nodes.prompt_refiner_node.OllamaClient.generate_streaming",
                side_effect=_capture_seed,
            ),
        ):
            node.refine(
                prompt="a forest",
                model="qwen3:8b",
                passes=2,
                seed=-1,
                temperature=0.5,
                top_p=0.9,
                timeout=30,
            )

        assert captured_seeds == [None, None]
