"""Unit tests for config module."""
import os
import tempfile
from unittest.mock import patch

import pytest


class TestGetPipelineParams:
    """Test get_pipeline_params returns correct structure."""

    def test_returns_dict(self):
        """get_pipeline_params returns a dictionary."""
        from config import get_pipeline_params
        assert isinstance(get_pipeline_params(), dict)

    def test_has_required_keys(self):
        """All expected keys are present in pipeline params."""
        from config import get_pipeline_params
        params = get_pipeline_params()
        required_keys = {
            'llm_model', 'embed_model', 'chunk_size', 'chunk_overlap',
            'top_k', 'response_mode', 'system_prompt_version',
            'file_extensions', 'ollama_request_timeout',
        }
        assert required_keys.issubset(params.keys())

    @patch.dict(os.environ, {
        'LLM_MODEL': 'test-llm',
        'EMBED_MODEL': 'test-embed',
        'CHUNK_SIZE': '256',
        'CHUNK_OVERLAP': '25',
        'TOP_K': '3',
        'RESPONSE_MODE': 'tree_summarize',
        'SYSTEM_PROMPT_VERSION': 'v2',
        'OLLAMA_REQUEST_TIMEOUT': '60.0',
    })
    def test_reflects_environment_variables(self):
        """Pipeline params reflect overridden environment variables."""
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
        assert params['ollama_request_timeout'] == 60.0

    @patch.dict(os.environ, {'FILE_EXTENSIONS': '.md,.pdf,.txt'})
    def test_file_extensions_parsed_as_list(self):
        """FILE_EXTENSIONS env var is split into a list."""
        import importlib
        import config
        importlib.reload(config)

        assert config.FILE_EXTENSIONS == ['.md', '.pdf', '.txt']

    def test_file_extensions_in_params_as_string(self):
        """file_extensions in pipeline params is a comma-joined string."""
        from config import get_pipeline_params
        params = get_pipeline_params()
        assert isinstance(params['file_extensions'], str)
        assert '.' in params['file_extensions']

    @patch.dict(os.environ, {'CHUNK_SIZE': 'not-a-number'})
    def test_invalid_chunk_size_raises_on_reload(self):
        """Non-numeric CHUNK_SIZE raises ValueError on module reload."""
        with pytest.raises(ValueError):
            import importlib
            import config
            importlib.reload(config)


class TestValidateConfig:
    """Test validate_config catches bad configuration."""

    def test_valid_config_passes(self, tmp_path):
        """A complete, valid configuration passes without error."""
        from config import validate_config
        with patch.dict(os.environ, {
            'OBSIDIAN_VAULT_PATH': str(tmp_path),
            'CHUNK_SIZE': '512',
            'CHUNK_OVERLAP': '50',
            'TOP_K': '5',
            'OLLAMA_REQUEST_TIMEOUT': '120.0',
            'FILE_EXTENSIONS': '.md,.pdf',
        }):
            validate_config()  # should not raise

    def test_missing_vault_path_raises(self):
        """Missing OBSIDIAN_VAULT_PATH raises ValueError."""
        from config import validate_config
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="OBSIDIAN_VAULT_PATH is required"):
                validate_config()

    def test_nonexistent_vault_path_raises(self):
        """A vault path that doesn't exist on disk raises ValueError."""
        from config import validate_config
        with patch.dict(os.environ, {'OBSIDIAN_VAULT_PATH': '/no/such/path'}):
            with pytest.raises(ValueError, match="does not exist"):
                validate_config()

    def test_overlap_equal_to_chunk_size_raises(self, tmp_path):
        """CHUNK_OVERLAP >= CHUNK_SIZE raises ValueError."""
        from config import validate_config
        with patch.dict(os.environ, {
            'OBSIDIAN_VAULT_PATH': str(tmp_path),
            'CHUNK_SIZE': '100',
            'CHUNK_OVERLAP': '100',
        }):
            with pytest.raises(ValueError, match="CHUNK_OVERLAP must be less than CHUNK_SIZE"):
                validate_config()

    def test_overlap_greater_than_chunk_size_raises(self, tmp_path):
        """CHUNK_OVERLAP > CHUNK_SIZE raises ValueError."""
        from config import validate_config
        with patch.dict(os.environ, {
            'OBSIDIAN_VAULT_PATH': str(tmp_path),
            'CHUNK_SIZE': '100',
            'CHUNK_OVERLAP': '200',
        }):
            with pytest.raises(ValueError, match="CHUNK_OVERLAP must be less than CHUNK_SIZE"):
                validate_config()

    def test_extension_without_dot_raises(self, tmp_path):
        """A file extension missing its leading dot raises ValueError."""
        from config import validate_config
        with patch.dict(os.environ, {
            'OBSIDIAN_VAULT_PATH': str(tmp_path),
            'FILE_EXTENSIONS': 'md',
        }):
            with pytest.raises(ValueError, match="must start with '.'"):
                validate_config()

    def test_chroma_dir_inside_vault_raises(self, tmp_path):
        """CHROMA_PERSIST_DIR inside the vault raises ValueError."""
        from config import validate_config
        chroma_inside = str(tmp_path / "chroma")
        with patch.dict(os.environ, {
            'OBSIDIAN_VAULT_PATH': str(tmp_path),
            'CHROMA_PERSIST_DIR': chroma_inside,
        }):
            with pytest.raises(ValueError, match="must not be inside the vault"):
                validate_config()

    def test_chroma_dir_outside_vault_passes(self, tmp_path):
        """CHROMA_PERSIST_DIR outside the vault passes validation."""
        from config import validate_config
        chroma_outside = str(tmp_path.parent / "chroma")
        with patch.dict(os.environ, {
            'OBSIDIAN_VAULT_PATH': str(tmp_path),
            'CHROMA_PERSIST_DIR': chroma_outside,
        }):
            validate_config()  # should not raise

    def test_multiple_errors_reported_together(self):
        """All config errors are collected and reported in one exception."""
        from config import validate_config
        with patch.dict(os.environ, {
            'OBSIDIAN_VAULT_PATH': '/no/such/path',
            'CHUNK_SIZE': '100',
            'CHUNK_OVERLAP': '200',
        }):
            with pytest.raises(ValueError) as exc_info:
                validate_config()
            msg = str(exc_info.value)
            assert "does not exist" in msg
            assert "CHUNK_OVERLAP must be less than CHUNK_SIZE" in msg
