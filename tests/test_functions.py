"""Tests for environment and configuration helper functions."""

import os
from unittest.mock import patch

from ssb_parquedit.functions import create_config
from ssb_parquedit.functions import get_bucket_name
from ssb_parquedit.functions import get_dapla_environment
from ssb_parquedit.functions import get_dapla_group
from ssb_parquedit.functions import get_team_name

_GROUP = "dapla-ffunk-developers"
_ENV = "test"


def test_get_dapla_group_returns_env_var() -> None:
    with patch.dict(os.environ, {"DAPLA_GROUP_CONTEXT": _GROUP}):
        assert get_dapla_group() == _GROUP


def test_get_dapla_group_defaults_to_empty() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert get_dapla_group() == ""


def test_get_dapla_environment_lowercases_value() -> None:
    with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "TEST"}):
        assert get_dapla_environment() == "test"


def test_get_dapla_environment_defaults_to_empty() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert get_dapla_environment() == ""


def test_get_team_name_extracts_prefix() -> None:
    with patch.dict(os.environ, {"DAPLA_GROUP_CONTEXT": _GROUP}):
        assert get_team_name() == "dapla-ffunk"


def test_get_bucket_name_builds_name() -> None:
    with patch.dict(
        os.environ, {"DAPLA_GROUP_CONTEXT": _GROUP, "DAPLA_ENVIRONMENT": _ENV}
    ):
        assert get_bucket_name() == "ssb-dapla-ffunk-data-produkt-test"


def test_create_config_returns_all_required_keys() -> None:
    with patch.dict(
        os.environ, {"DAPLA_GROUP_CONTEXT": _GROUP, "DAPLA_ENVIRONMENT": _ENV}
    ):
        config = create_config()
        for key in ("dbname", "dbuser", "data_path", "catalog_name", "metadata_schema"):
            assert key in config


def test_create_config_normalizes_hyphens_in_catalog() -> None:
    with patch.dict(
        os.environ, {"DAPLA_GROUP_CONTEXT": _GROUP, "DAPLA_ENVIRONMENT": _ENV}
    ):
        config = create_config()
        assert "-" not in config["catalog_name"]
        assert "-" not in config["metadata_schema"]
