"""
Unit tests for style presets and template loading.
"""

from pathlib import Path
from unittest.mock import mock_open, patch

from nodes.prompt_generator_node import PromptGeneratorNode


class TestStylePresets:
    """Test suite for style preset loading and template rendering."""

    def test_default_styles_count(self):
        """DEFAULT_STYLES should contain exactly 9 styles."""
        assert len(PromptGeneratorNode.DEFAULT_STYLES) == 9

    def test_default_styles_keys(self):
        """DEFAULT_STYLES should contain expected style keys."""
        expected = {
            "cinematic",
            "anime",
            "photorealistic",
            "fantasy",
            "abstract",
            "cyberpunk",
            "sci-fi",
            "video_wan",
            "still_image",
        }
        assert set(PromptGeneratorNode.DEFAULT_STYLES.keys()) == expected

    def test_default_styles_have_template(self):
        """Each default style should have a template string."""
        for key, data in PromptGeneratorNode.DEFAULT_STYLES.items():
            assert "template" in data, f"Style '{key}' missing template"
            assert isinstance(data["template"], str), f"Style '{key}' template not a string"
            assert len(data["template"]) > 0, f"Style '{key}' template is empty"

    def test_get_style_list_fallback(self):
        """_get_style_list should fall back to DEFAULT_STYLES when YAML missing."""
        styles = PromptGeneratorNode._get_style_list()
        assert len(styles) == 9
        assert "cinematic" in styles

    def test_load_templates_yaml_mock(self):
        """_load_templates should parse YAML when available."""
        mock_yaml = {
            "test_style": {
                "name": "Test",
                "template": "Test template for {{ description }}",
            }
        }
        with (
            patch(
                "builtins.open",
                mock_open(read_data="test_style:\n  name: Test\n  template: Test template"),
            ),
            patch.object(Path, "exists", return_value=True),
            patch("yaml.safe_load", return_value=mock_yaml),
        ):
            node = PromptGeneratorNode()
            assert "test_style" in node.style_templates

    def test_render_template_jinja2(self):
        """_render_template should substitute Jinja2 variables."""
        node = PromptGeneratorNode()
        result = node._render_template("cinematic", "a forest", emphasis="lighting", mood="mysterious")
        assert "a forest" in result
        assert "lighting" in result
        assert "mysterious" in result

    def test_render_template_unknown_style_fallback(self):
        """Unknown style should fall back to cinematic."""
        node = PromptGeneratorNode()
        result = node._render_template("nonexistent", "a test")
        assert "a test" in result
        assert "cinematic" in result.lower() or "dramatic" in result.lower()

    def test_render_template_no_emphasis_mood(self):
        """Template should render cleanly without emphasis or mood."""
        node = PromptGeneratorNode()
        result = node._render_template("cinematic", "a mountain")
        assert "a mountain" in result
        # Emphasis/mood conditional blocks should be absent or minimal
        assert "Focus particularly on:" not in result or "None" not in result
