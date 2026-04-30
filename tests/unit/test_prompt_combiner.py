"""Smoke tests for PromptCombinerNode (pure logic, no I/O)."""

from __future__ import annotations

import pytest

from nodes.prompt_combiner_node import CombineMode, ModeLiteral, PromptCombinerNode


@pytest.fixture
def node() -> PromptCombinerNode:
    return PromptCombinerNode()


class TestCombineModeEnum:
    def test_enum_values_are_canonical_strings(self) -> None:
        assert CombineMode.BLEND.value == "blend"
        assert CombineMode.CONCAT.value == "concat"
        assert CombineMode.WEIGHTED_AVERAGE.value == "weighted_average"

    def test_input_types_dropdown_matches_literal(self) -> None:
        from typing import get_args

        choices = PromptCombinerNode.INPUT_TYPES()["required"]["mode"][0]
        assert choices == list(get_args(ModeLiteral))


class TestCombineEmptyInputs:
    def test_no_prompts_returns_helpful_message(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="", mode="blend")
        assert "At least one prompt" in out

    def test_whitespace_only_prompts_treated_as_empty(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="   ", mode="blend", prompt_2="\n\t")
        assert "At least one prompt" in out

    def test_single_prompt_returned_unmodified(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="a forest at dawn", mode="blend")
        assert out == "a forest at dawn"


class TestBlendMode:
    def test_high_weight_gets_double_parens(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="forest", weight_1=1.6, prompt_2="dawn", weight_2=1.0, mode="blend")
        assert "((forest))" in out
        assert "dawn" in out

    def test_moderate_weight_gets_single_parens(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="forest", weight_1=1.3, prompt_2="dawn", weight_2=1.0, mode="blend")
        assert "(forest)" in out

    def test_low_weight_gets_brackets(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="forest", weight_1=0.4, prompt_2="dawn", weight_2=1.0, mode="blend")
        assert "[forest]" in out

    def test_zero_weight_excluded(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="forest", weight_1=0.0, prompt_2="dawn", weight_2=1.0, mode="blend")
        assert "forest" not in out
        assert "dawn" in out


class TestConcatMode:
    def test_default_separator(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="a", mode="concat", prompt_2="b", prompt_3="c")
        assert out == "a, b, c"

    def test_custom_separator(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="a", mode="concat", prompt_2="b", separator=" | ")
        assert out == "a | b"


class TestWeightedAverageMode:
    def test_dominant_weight_gets_triple_parens(self, node: PromptCombinerNode) -> None:
        # Triple parens requires ratio >= 2.0, i.e. dominant weight at least 2x avg.
        # weights 2.0, 0.0 → avg 1.0 → hero ratio 2.0 → triple parens
        (out,) = node.combine(
            prompt_1="hero",
            weight_1=2.0,
            prompt_2="extras",
            weight_2=0.0,
            mode="weighted_average",
        )
        assert "(((hero)))" in out

    def test_zero_total_weight_falls_back_to_join(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(
            prompt_1="a",
            weight_1=0.0,
            prompt_2="b",
            weight_2=0.0,
            mode="weighted_average",
        )
        assert out == "a, b"


class TestUnknownMode:
    def test_unknown_mode_returns_friendly_error(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="x", prompt_2="y", mode="not_a_real_mode")  # type: ignore[arg-type]
        assert "Unknown mode" in out
        assert "blend" in out  # mentions valid options

    def test_unknown_mode_with_single_prompt_still_validated(self, node: PromptCombinerNode) -> None:
        (out,) = node.combine(prompt_1="x", mode="not_a_real_mode")  # type: ignore[arg-type]
        assert "Unknown mode" in out
