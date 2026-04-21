import os
from unittest.mock import patch

from ssb_parquedit.functions import get_dapla_environment
from ssb_parquedit.functions import get_dapla_group


def test_dapla_group_reads_env() -> None:
    with patch.dict(os.environ, {"DAPLA_GROUP_CONTEXT": "dapla-ffunk-developers"}):
        assert get_dapla_group() == "dapla-ffunk-developers"


def test_dapla_environment_lowercases() -> None:
    with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "PROD"}):
        assert get_dapla_environment() == "prod"
