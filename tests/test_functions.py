import os
from unittest.mock import patch

from ssb_parquedit.functions import get_bucket_name
from ssb_parquedit.functions import get_dapla_group
from ssb_parquedit.functions import get_team_name


class TestGetDaplaGroup:
    def test_returns_env_variable(self) -> None:
        with patch.dict(os.environ, {"DAPLA_GROUP_CONTEXT": "dapla-ffunk-developers"}):
            assert get_dapla_group() == "dapla-ffunk-developers"

    def test_returns_empty_string_when_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert get_dapla_group() == ""


class TestGetTeamName:
    def test_extracts_team_name(self) -> None:
        with patch.dict(os.environ, {"DAPLA_GROUP_CONTEXT": "dapla-ffunk-developers"}):
            assert get_team_name() == "dapla-ffunk"

    def test_returns_empty_string_when_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert get_team_name() == ""


class TestGetBucketName:
    def test_builds_bucket_name(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DAPLA_GROUP_CONTEXT": "dapla-ffunk-developers",
                "DAPLA_ENVIRONMENT": "TEST",
            },
        ):
            assert get_bucket_name() == "ssb-dapla-ffunk-data-produkt-test"

    def test_environment_is_lowercased(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DAPLA_GROUP_CONTEXT": "dapla-ffunk-developers",
                "DAPLA_ENVIRONMENT": "PROD",
            },
        ):
            assert get_bucket_name() == "ssb-dapla-ffunk-data-produkt-prod"

    def test_returns_empty_environment_when_not_set(self) -> None:
        with patch.dict(
            os.environ, {"DAPLA_GROUP_CONTEXT": "dapla-ffunk-developers"}, clear=True
        ):
            assert get_bucket_name() == "ssb-dapla-ffunk-data-produkt-"
