"""Unit tests for the dual-stream refiner: output parsing and refine() flow."""

from unittest.mock import patch

from nodes.adapters.ollama_client import StreamResult
from nodes.prompt_dual_stream_refiner_node import (
    PromptDualStreamRefinerNode,
    parse_dual_stream,
)


class TestParseDualStream:
    """parse_dual_stream must tolerate label/delimiter variants."""

    def test_plain_positive_negative(self):
        pos, neg = parse_dual_stream("Positive: a cat\nNegative: blurry, lowres")
        assert pos == "a cat"
        assert neg == "blurry, lowres"

    def test_prompt_suffix_labels(self):
        pos, neg = parse_dual_stream("Positive prompt: a dog\nNegative prompt: deformed")
        assert pos == "a dog"
        assert neg == "deformed"

    def test_markdown_bold_labels(self):
        pos, neg = parse_dual_stream("**Positive:** a fox\n**Negative:** ugly")
        assert pos == "a fox"
        assert neg == "ugly"

    def test_case_insensitive(self):
        pos, neg = parse_dual_stream("POSITIVE: bright\nNEGATIVE: dark")
        assert pos == "bright"
        assert neg == "dark"

    def test_no_negative_section(self):
        pos, neg = parse_dual_stream("Positive: just a positive prompt")
        assert pos == "just a positive prompt"
        assert neg == ""

    def test_unlabeled_text_is_positive(self):
        pos, neg = parse_dual_stream("a forest at twilight")
        assert pos == "a forest at twilight"
        assert neg == ""

    def test_multiline_blocks_preserved(self):
        text = "Positive: line one,\nline two\nNegative: bad, worse"
        pos, neg = parse_dual_stream(text)
        assert "line one" in pos and "line two" in pos
        assert neg == "bad, worse"

    def test_negative_space_in_body_is_not_a_split(self):
        """The art term 'negative space' must not be mistaken for a label."""
        pos, neg = parse_dual_stream("a portrait with strong negative space, studio lighting")
        assert pos == "a portrait with strong negative space, studio lighting"
        assert neg == ""

    def test_hyphenated_negative_space_in_body_is_not_a_split(self):
        pos, neg = parse_dual_stream("Positive: negative-space composition\nNegative: blurry")
        assert pos == "negative-space composition"
        assert neg == "blurry"


class TestRefineFlow:
    """refine() should parse a successful stream into two outputs."""

    def test_successful_generation_splits_streams(self):
        node = PromptDualStreamRefinerNode()
        with patch(
            "nodes.prompt_dual_stream_refiner_node.OllamaClient.generate_streaming",
            return_value=StreamResult(text="Positive: a knight\nNegative: blurry", kind="ok"),
        ):
            pos, neg = node.refine(prompt="a knight", model="qwen3:8b")
        assert pos == "a knight"
        assert neg == "blurry"

    def test_empty_prompt_short_circuits(self):
        node = PromptDualStreamRefinerNode()
        pos, neg = node.refine(prompt="   ", model="qwen3:8b")
        assert pos.startswith("[PromptDualStreamRefiner]")
        assert neg == ""

    def test_model_crash_surfaces_error(self):
        node = PromptDualStreamRefinerNode()
        with patch(
            "nodes.prompt_dual_stream_refiner_node.OllamaClient.generate_streaming",
            return_value=StreamResult(kind="model_crash", message="runner crashed"),
        ):
            pos, neg = node.refine(prompt="a knight", model="broken")
        assert "runner crashed" in pos
        assert neg == ""
