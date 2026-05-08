"""
Unit tests for OllamaClient adapter.
"""

from unittest.mock import MagicMock, patch

from nodes.adapters.ollama_client import OllamaClient, StreamResult


class TestOllamaClientDiscovery:
    """Test suite for model discovery."""

    def test_default_models_when_api_unavailable(self):
        """Should return defaults when ollama package not installed."""
        with patch("nodes.adapters.ollama_client.OLLAMA_API_AVAILABLE", False):
            client = OllamaClient()
            models = client.discover_models()
            assert models == ["qwen3:8b", "qwen3:4b", "llama3.2:latest"]

    def test_lora_prioritization(self):
        """LoRA models should be sorted before base models."""
        mock_models = [
            {"model": "qwen3:8b"},
            {"model": "my-lora-v1"},
            {"model": "limbicnation-style"},
        ]
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": mock_models}
        import nodes.adapters.ollama_client as client_module

        orig_ollama = getattr(client_module, "ollama", None)
        orig_available = getattr(client_module, "OLLAMA_API_AVAILABLE", False)
        try:
            client_module.ollama = mock_ollama
            client_module.OLLAMA_API_AVAILABLE = True
            client_module._MODEL_CACHE = None
            client_module._CACHE_TIME = 0
            client = OllamaClient()
            models = client.discover_models()
            # Both LoRA models should be first (sorted alphabetically within LoRA group)
            assert models[0] in ("my-lora-v1", "limbicnation-style")
            assert models[1] in ("my-lora-v1", "limbicnation-style")
            assert models[2] == "qwen3:8b"
        finally:
            client_module.ollama = orig_ollama
            client_module.OLLAMA_API_AVAILABLE = orig_available
            client_module._MODEL_CACHE = None
            client_module._CACHE_TIME = 0

    def test_caching(self):
        """Second call should return cached results within 60s."""
        mock_models = [{"model": "qwen3:8b"}]
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": mock_models}
        import nodes.adapters.ollama_client as client_module

        orig_ollama = getattr(client_module, "ollama", None)
        orig_available = getattr(client_module, "OLLAMA_API_AVAILABLE", False)
        try:
            client_module.ollama = mock_ollama
            client_module.OLLAMA_API_AVAILABLE = True
            client_module._MODEL_CACHE = None
            client_module._CACHE_TIME = 0
            client = OllamaClient()
            _ = client.discover_models()
            _ = client.discover_models()
            mock_ollama.list.assert_called_once()  # Only called once due to cache
        finally:
            client_module.ollama = orig_ollama
            client_module.OLLAMA_API_AVAILABLE = orig_available
            client_module._MODEL_CACHE = None
            client_module._CACHE_TIME = 0

    def test_cache_shared_across_instances(self):
        """Module-level cache should be shared between OllamaClient instances."""
        mock_models = [{"model": "qwen3:8b"}]
        mock_ollama = MagicMock()
        mock_ollama.list.return_value = {"models": mock_models}
        import nodes.adapters.ollama_client as client_module

        orig_ollama = getattr(client_module, "ollama", None)
        orig_available = getattr(client_module, "OLLAMA_API_AVAILABLE", False)
        try:
            client_module.ollama = mock_ollama
            client_module.OLLAMA_API_AVAILABLE = True
            client_module._MODEL_CACHE = None
            client_module._CACHE_TIME = 0
            client1 = OllamaClient()
            client2 = OllamaClient()
            _ = client1.discover_models()
            _ = client2.discover_models()
            mock_ollama.list.assert_called_once()  # Shared cache
        finally:
            client_module.ollama = orig_ollama
            client_module.OLLAMA_API_AVAILABLE = orig_available
            client_module._MODEL_CACHE = None
            client_module._CACHE_TIME = 0


class TestOllamaClientHealth:
    """Test suite for health checks."""

    def test_unhealthy_when_api_unavailable(self):
        """Should report unhealthy when ollama package missing."""
        with patch("nodes.adapters.ollama_client.OLLAMA_API_AVAILABLE", False):
            client = OllamaClient()
            healthy, msg, _loaded = client.check_health("qwen3:8b")
            assert healthy is False
            assert "not available" in msg

    def test_healthy_model_loaded(self):
        """Should detect loaded model."""
        mock_ps = MagicMock()
        mock_ps.models = [MagicMock(model="qwen3:8b")]
        mock_ollama = MagicMock()
        mock_ollama.ps.return_value = mock_ps
        mock_ollama.list.return_value = {"models": []}
        import nodes.adapters.ollama_client as client_module

        orig_ollama = getattr(client_module, "ollama", None)
        orig_available = getattr(client_module, "OLLAMA_API_AVAILABLE", False)
        try:
            client_module.ollama = mock_ollama
            client_module.OLLAMA_API_AVAILABLE = True
            client = OllamaClient()
            healthy, msg, loaded = client.check_health("qwen3:8b")
            assert healthy is True
            assert loaded is True
            assert "loaded" in msg
        finally:
            client_module.ollama = orig_ollama
            client_module.OLLAMA_API_AVAILABLE = orig_available

    def test_healthy_model_not_loaded(self):
        """Should detect cold start when model not loaded."""
        mock_ps = MagicMock()
        mock_ps.models = [MagicMock(model="llama3.2:latest")]
        mock_ollama = MagicMock()
        mock_ollama.ps.return_value = mock_ps
        mock_ollama.list.return_value = {"models": []}
        import nodes.adapters.ollama_client as client_module

        orig_ollama = getattr(client_module, "ollama", None)
        orig_available = getattr(client_module, "OLLAMA_API_AVAILABLE", False)
        try:
            client_module.ollama = mock_ollama
            client_module.OLLAMA_API_AVAILABLE = True
            client = OllamaClient()
            healthy, msg, loaded = client.check_health("qwen3:8b")
            assert healthy is True
            assert loaded is False
            assert "cold start" in msg
        finally:
            client_module.ollama = orig_ollama
            client_module.OLLAMA_API_AVAILABLE = orig_available

    def test_exact_tag_match_no_prefix_conflation(self):
        """qwen3:4b should NOT report loaded when only qwen3:8b is in VRAM."""
        mock_ps = MagicMock()
        mock_ps.models = [MagicMock(model="qwen3:8b")]
        mock_ollama = MagicMock()
        mock_ollama.ps.return_value = mock_ps
        mock_ollama.list.return_value = {"models": []}
        import nodes.adapters.ollama_client as client_module

        orig_ollama = getattr(client_module, "ollama", None)
        orig_available = getattr(client_module, "OLLAMA_API_AVAILABLE", False)
        try:
            client_module.ollama = mock_ollama
            client_module.OLLAMA_API_AVAILABLE = True
            client = OllamaClient()
            healthy, _msg, loaded = client.check_health("qwen3:4b")
            assert healthy is True
            assert loaded is False
        finally:
            client_module.ollama = orig_ollama
            client_module.OLLAMA_API_AVAILABLE = orig_available


class TestOllamaClientSubprocess:
    """Test suite for subprocess fallback."""

    def test_successful_subprocess(self):
        """Should return success on valid subprocess output."""
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Generated prompt text"
        with patch("subprocess.run", return_value=mock_result):
            client = OllamaClient()
            success, output = client.generate_subprocess("qwen3:8b", "prompt", 30)
            assert success is True
            assert output == "Generated prompt text"

    def test_failed_subprocess(self):
        """Should return failure on non-zero exit."""
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stderr = "Model not found"
        with patch("subprocess.run", return_value=mock_result):
            client = OllamaClient()
            success, output = client.generate_subprocess("qwen3:8b", "prompt", 30)
            assert success is False
            assert "Model not found" in output

    def test_subprocess_timeout(self):
        """Should handle timeout gracefully."""
        from subprocess import TimeoutExpired

        with patch("subprocess.run", side_effect=TimeoutExpired("ollama", 30)):
            client = OllamaClient()
            success, output = client.generate_subprocess("qwen3:8b", "prompt", 30)
            assert success is False
            assert "timed out" in output

    def test_subprocess_not_found(self):
        """Should handle missing ollama binary."""
        with patch("subprocess.run", side_effect=FileNotFoundError()):
            client = OllamaClient()
            success, output = client.generate_subprocess("qwen3:8b", "prompt", 30)
            assert success is False
            assert "not found" in output.lower() or "Install" in output


class _FakeResponseError(Exception):
    """Duck-typed stand-in for ollama.ResponseError with status_code attr."""

    def __init__(self, message: str, status_code: int):
        super().__init__(message)
        self.status_code = status_code


class TestOllamaClientStreamingErrors:
    """Test suite for generate_streaming error categorisation."""

    @staticmethod
    def _patch_module(mock_ollama, response_error_cls=None):
        """Helper: swap module-level ollama + ResponseError, return restore fn."""
        import nodes.adapters.ollama_client as client_module

        orig = (
            getattr(client_module, "ollama", None),
            getattr(client_module, "OLLAMA_API_AVAILABLE", False),
            getattr(client_module, "OllamaResponseError", None),
        )
        client_module.ollama = mock_ollama
        client_module.OLLAMA_API_AVAILABLE = True
        client_module.OllamaResponseError = response_error_cls

        def restore():
            client_module.ollama = orig[0]
            client_module.OLLAMA_API_AVAILABLE = orig[1]
            client_module.OllamaResponseError = orig[2]

        return restore

    def test_runner_crash_returns_model_crash_kind(self):
        """500 + 'runner terminated' should be classified as model_crash."""
        crash_exc = _FakeResponseError(
            "llama runner process has terminated: %!w(<nil>) (status code: 500)",
            status_code=500,
        )

        def _bad_iter():
            yield from ()
            raise crash_exc

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = _bad_iter()

        restore = self._patch_module(mock_ollama, response_error_cls=_FakeResponseError)
        try:
            client = OllamaClient()
            result = client.generate_streaming(
                model="qwen3-limbicnation",
                prompt="hello",
                temperature=0.7,
                top_p=0.9,
                timeout=30,
            )
        finally:
            restore()

        assert isinstance(result, StreamResult)
        assert result.kind == "model_crash"
        assert "qwen3-limbicnation" in result.message
        assert "restart Ollama" in result.message
        assert "qwen3:8b" in result.message
        assert result.text is None

    def test_runner_crash_via_duck_typed_error_when_class_unavailable(self):
        """Even without ollama.ResponseError, status_code=500 + signal triggers model_crash."""
        crash_exc = _FakeResponseError("Load failed: llama runner terminated", status_code=500)

        def _bad_iter():
            yield from ()
            raise crash_exc

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = _bad_iter()

        # Simulate older ollama package (no ResponseError export)
        restore = self._patch_module(mock_ollama, response_error_cls=None)
        try:
            client = OllamaClient()
            result = client.generate_streaming(
                model="bad-model",
                prompt="x",
                temperature=0.7,
                top_p=0.9,
                timeout=30,
            )
        finally:
            restore()

        assert result.kind == "model_crash"

    def test_generic_server_error_returns_server_error(self):
        """Non-crash 5xx (e.g. 503) should be classified as server_error."""
        exc = _FakeResponseError("service unavailable", status_code=503)

        def _bad_iter():
            yield from ()
            raise exc

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = _bad_iter()

        restore = self._patch_module(mock_ollama, response_error_cls=_FakeResponseError)
        try:
            client = OllamaClient()
            result = client.generate_streaming(
                model="qwen3:8b",
                prompt="x",
                temperature=0.7,
                top_p=0.9,
                timeout=30,
            )
        finally:
            restore()

        assert result.kind == "server_error"
        assert "503" in result.message

    def test_connection_error_returns_transient(self):
        """ConnectionError raised during streaming should be transient."""

        def _bad_iter():
            yield from ()
            raise ConnectionError("connection refused")

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = _bad_iter()

        restore = self._patch_module(mock_ollama, response_error_cls=_FakeResponseError)
        try:
            client = OllamaClient()
            result = client.generate_streaming(
                model="qwen3:8b",
                prompt="x",
                temperature=0.7,
                top_p=0.9,
                timeout=30,
            )
        finally:
            restore()

        assert result.kind == "transient"
        assert "not reachable" in result.message

    def test_success_returns_ok(self):
        """Normal streaming should yield kind='ok' with concatenated text."""

        def _good_iter():
            yield {"response": "hello "}
            yield {"response": "world"}

        mock_ollama = MagicMock()
        mock_ollama.generate.return_value = _good_iter()

        restore = self._patch_module(mock_ollama, response_error_cls=_FakeResponseError)
        try:
            client = OllamaClient()
            result = client.generate_streaming(
                model="qwen3:8b",
                prompt="x",
                temperature=0.7,
                top_p=0.9,
                timeout=30,
            )
        finally:
            restore()

        assert result.kind == "ok"
        assert result.text == "hello world"

    def test_unavailable_when_api_missing(self):
        """If ollama package is unavailable, kind should be 'unavailable'."""
        with patch("nodes.adapters.ollama_client.OLLAMA_API_AVAILABLE", False):
            client = OllamaClient()
            result = client.generate_streaming(
                model="qwen3:8b",
                prompt="x",
                temperature=0.7,
                top_p=0.9,
                timeout=30,
            )
        assert result.kind == "unavailable"
        assert result.text is None
