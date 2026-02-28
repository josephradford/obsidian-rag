"""Unit tests for config module."""
from config import get_pipeline_params
import pytest
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestConfig:
    """Test configuration loading and parameter management."""

    @patch.dict(os.environ, {
        'LLM_MODEL': 'test-llm',
        'EMBED_MODEL': 'test-embed',
        'CHUNK_SIZE': '256',
        'CHUNK_OVERLAP': '25',
        'TOP_K': '3',
        'RESPONSE_MODE': 'tree_summarize',
        'SYSTEM_PROMPT_VERSION': 'v2'
    })
    def test_get_pipeline_params_with_env_vars(self):
        """Test that pipeline params are correctly loaded from environment variables."""
        # Import config after setting env vars to ensure they're loaded
        import importlib
        import config
        importlib.reload(config)

        params = config.get_pipeline_params()

        assert params['llm_model'] == 'test-llm'
        assert params['embed_model'] == 'test-embed'
        assert params['chunk_size'] == 256
        assert params['chunk_overlap'] == 25
        assert params['top_k'] == 3
        assert params['response_mode'] == 'tree_summarize'
        assert params['system_prompt_version'] == 'v2'

    @patch.dict(os.environ, {}, clear=True)
    def test_get_pipeline_params_with_defaults(self):
        """Test that default values are used when environment variables are not set."""
        # Clear and reimport to test defaults
        import importlib
        import config
        importlib.reload(config)

        params = config.get_pipeline_params()

        assert params['llm_model'] == 'llama3.2:3b'
        assert params['embed_model'] == 'nomic-embed-text'
        assert params['chunk_size'] == 512
        assert params['chunk_overlap'] == 50
        assert params['top_k'] == 5
        assert params['response_mode'] == 'compact'
        assert params['system_prompt_version'] == 'v1'

    @patch.dict(os.environ, {'FILE_EXTENSIONS': '.md,.pdf,.txt'})
    def test_file_extensions_parsing(self):
        """Test that FILE_EXTENSIONS is correctly parsed into a list."""
        import importlib
        import config
        importlib.reload(config)

        assert config.FILE_EXTENSIONS == ['.md', '.pdf', '.txt']

    @patch.dict(os.environ, {'CHUNK_SIZE': 'invalid'})
    def test_invalid_chunk_size_raises_error(self):
        """Test that invalid CHUNK_SIZE raises ValueError."""
        with pytest.raises(ValueError):
            import importlib
            import config
            importlib.reload(config)

    def test_get_pipeline_params_returns_dict(self):
        """Test that get_pipeline_params returns a dictionary."""
        params = get_pipeline_params()
        assert isinstance(params, dict)
        assert len(params) == 7  # Should have 7 parameters

    def test_get_pipeline_params_has_required_keys(self):
        """Test that all required keys are present in pipeline params."""
        params = get_pipeline_params()
        required_keys = {
            'llm_model', 'embed_model', 'chunk_size', 'chunk_overlap',
            'top_k', 'response_mode', 'system_prompt_version'
        }
        assert set(params.keys()) == required_keys
