
import pytest

from ssb_parquedit.functions import create_config
from ssb_parquedit.functions import get_bucket_name
from ssb_parquedit.functions import get_dapla_environment
from ssb_parquedit.functions import get_dapla_group
from ssb_parquedit.functions import get_dapla_user
from ssb_parquedit.functions import get_port_number
from ssb_parquedit.functions import get_team_name


class TestGetDaplaGroup:
    def test_dapla_group_env_var_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAPLA_GROUP_CONTEXT", False)
        out = get_dapla_group()
        assert out == ""

    def test_dapla_group_env_var_is_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "test")
        out = get_dapla_group()
        assert out == "test"

class TestGetTeamName:
    def test_get_team_name_env_var_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAPLA_GROUP_CONTEXT", False)
        out = get_team_name()
        assert out == ""

    def test_get_team_name_name_has_no_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "test")
        out = get_team_name()
        assert out == "tes"

    def test_get_team_name_has_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "testa-testb")
        out = get_team_name()
        assert out == "testa"

    def test_has_two_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "testa-testb-testc")
        out = get_team_name()
        assert out == "testa-testb"

    def test_lines_side_by_side(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "testa-testb--testc")
        out = get_team_name()
        assert out == "testa-testb-"

    def test_line_at_end(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "testa-")
        out = get_team_name()
        assert out == "testa"


class TestGetBucketName:
    def test_no_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAPLA_ENVIRONMENT", False)
        monkeypatch.delenv("DAPLA_GROUP_CONTEXT", False)
        out = get_bucket_name()

        assert out == "ssb--data-produkt-"

    def test_only_team_name_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "testa-testb")
        monkeypatch.delenv("DAPLA_ENVIRONMENT", False)
        out = get_bucket_name()

        assert out == "ssb-testa-data-produkt-"

    def test_only_dapla_environment_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAPLA_GROUP_CONTEXT", False)
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "testb")
        out = get_bucket_name()

        assert out == "ssb--data-produkt-testb"

    def test_both_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_GROUP_CONTEXT", "testa-testb")
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "testc")
        out = get_bucket_name()

        assert out == "ssb-testa-data-produkt-testc"


class TestGetDaplaEnvironment:
    def test_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAPLA_ENVIRONMENT", False)
        out = get_dapla_environment()

        assert out == ""

    def test_env_set_upper_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "TEST")
        out = get_dapla_environment()

        assert out == "test"

    def test_env_set_lower_case(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "test")
        out = get_dapla_environment()

        assert out == "test"

class TestGetPortNumber:
    def test_env_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PARQEDIT_DB_PORT", False)
        out = get_port_number()

        assert out == ""

    def test_env_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("PARQEDIT_DB_PORT", "1234")
        out = get_port_number()

        assert out == "1234"

class TestCreateConfig:
    def test_enviroment_test(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "test")
        out = create_config()

        assert out["dbname"] == "dapla-ffunk"
        assert "@dapla-group-sa-t-57.iam" in out["dbuser"]

    def test_enviroment_prod(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_ENVIRONMENT", "prod")
        out = create_config()

        assert out["dbname"] == "parquedit"
        assert "@dapla-group-sa-p-ye.iam" in out["dbuser"]
        assert "team_" in out["metadata_schema"]

    def test_environment_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAPLA_ENVIRONMENT", False)
        out = create_config()

        assert out["dbname"] == "dapla-ffunk"
        assert "@dapla-group-sa-t-57.iam" in out["dbuser"]


class TestGetDaplaUser:
    def test_env_var_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DAPLA_USER", False)
        out = get_dapla_user()

        assert out == ""

    def test_env_var_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DAPLA_USER", "test")
        out = get_dapla_user()

        assert out == "test"