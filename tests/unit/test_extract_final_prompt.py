"""
Unit tests for extract_final_prompt utility.
"""

from nodes.prompt_generator_node import extract_final_prompt


class TestExtractFinalPrompt:
    """Test suite for the extract_final_prompt function."""

    def test_empty_input(self):
        """Empty string should return empty."""
        assert extract_final_prompt("") == ""

    def test_none_input(self):
        """None should return None."""
        assert extract_final_prompt(None) is None

    def test_qwen3_thinking_block(self):
        """Qwen3 thinking blocks should be stripped."""
        text = (
            "Thinking...\nThis is the reasoning\n...done thinking.\nFinal prompt here"
        )
        assert extract_final_prompt(text) == "Final prompt here"

    def test_prompt_prefix_removal(self):
        """Markdown prompt prefixes should be removed."""
        text = "**Prompt:** A beautiful sunset"
        assert extract_final_prompt(text) == "A beautiful sunset"

    def test_stable_diffusion_prefix(self):
        """Stable Diffusion prefix should be removed."""
        text = "**Stable Diffusion Prompt:** A cyberpunk city"
        assert extract_final_prompt(text) == "A cyberpunk city"

    def test_none_string_removal(self):
        """Stray 'None' strings should be removed."""
        text = "A prompt None with None values"
        result = extract_final_prompt(text)
        assert "None" not in result
        assert "A prompt" in result
        assert "with" in result
        assert "values" in result

    def test_quote_stripping(self):
        """Leading/trailing quotes should be stripped."""
        text = '"A quoted prompt"'
        assert extract_final_prompt(text) == "A quoted prompt"

    def test_whitespace_cleanup(self):
        """Leading/trailing whitespace should be stripped."""
        text = "   A prompt with spaces   "
        assert extract_final_prompt(text) == "A prompt with spaces"

    def test_multiple_thinking_blocks(self):
        """Multiple thinking blocks should all be removed."""
        text = "Thinking...\nFirst thought\n...done thinking.\nThinking...\nSecond thought\n...done thinking.\nFinal"
        assert extract_final_prompt(text) == "Final"

    def test_no_thinking_no_prefix(self):
        """Clean prompt should pass through unchanged."""
        text = "A clean prompt without any prefixes"
        assert extract_final_prompt(text) == "A clean prompt without any prefixes"

    def test_complex_real_world(self):
        """Complex real-world example with multiple artifacts."""
        text = (
            "Thinking...\n"
            "I need to create a cinematic prompt\n"
            "...done thinking.\n"
            "**Stable Diffusion Prompt:**\n"
            '"A mystical forest at twilight, dramatic lighting, 8k"\n'
            "None"
        )
        assert (
            extract_final_prompt(text)
            == "A mystical forest at twilight, dramatic lighting, 8k"
        )
