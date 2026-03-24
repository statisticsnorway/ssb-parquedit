"""Tests for DDL (Data Definition Language) operations."""

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from ssb_parquedit.utils import SQLInjectionError

# Fixtures are imported from conftest.py: stub_external_modules, fake_conn, db_config


@pytest.fixture
def sut_ddl() -> Any:
    """Import and return the DDLOperations class."""
    import importlib

    module = importlib.import_module("ssb_parquedit.ddl")
    importlib.reload(module)
    return module.DDLOperations


class TestCreateTableFromDataFrame:
    """Test table creation from DataFrame."""

    def test_create_from_dataframe_basic(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test basic table creation from DataFrame."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        ddl_ops.create_table("users", df)

        # Should have called execute
        assert fake_conn.execute.called

    def test_create_from_dataframe_validates_table_name(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        with pytest.raises(ValueError, match="Invalid table name"):
            ddl_ops.create_table("123invalid", df)

    def test_create_from_dataframe_creates_id_column(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that _id column is created."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        ddl_ops.create_table("users", df)

        # Should contain _id in the DDL (as a cast or column name)
        assert any("_id" in str(call) for call in fake_conn.execute.call_args_list)

    def test_create_from_dataframe_creates_empty_table(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that table is created empty (WHERE 1=2 pattern)."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        ddl_ops.create_table("users", df)

        # Should register the dataframe and create empty table
        assert fake_conn.register.called


class TestCreateTableFromParquet:
    """Test table creation from Parquet file."""

    def test_create_from_parquet_basic(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test basic table creation from Parquet file."""
        ddl_ops = sut_ddl(fake_conn)

        ddl_ops.create_table("users", "gs://bucket/users.parquet")

        # Should have called execute
        assert fake_conn.execute.called

    def test_create_from_parquet_uses_parameterized_query(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that Parquet path uses parameterized query."""
        ddl_ops = sut_ddl(fake_conn)

        parquet_path = "gs://bucket/users.parquet"
        ddl_ops.create_table("users", parquet_path)

        # Should call execute with parameters
        execute_calls = fake_conn.execute.call_args_list
        assert any(
            len(call[0]) > 1 or call[1].get("parameters", []) for call in execute_calls
        ), "Expected parameterized query"

    def test_create_from_parquet_validates_table_name(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated."""
        ddl_ops = sut_ddl(fake_conn)

        with pytest.raises(ValueError, match="Invalid table name"):
            ddl_ops.create_table("invalid-name", "gs://bucket/file.parquet")


class TestCreateTableFromSchema:
    """Test table creation from JSON Schema."""

    def test_create_from_schema_basic(self, sut_ddl: Any, fake_conn: MagicMock) -> None:
        """Test basic table creation from schema."""
        ddl_ops = sut_ddl(fake_conn)

        schema = {
            "properties": {
                "id": {"type": "integer"},
                "name": {"type": "string"},
            },
            "required": ["id"],
        }

        ddl_ops.create_table("users", schema)

        # Should have called execute
        assert fake_conn.execute.called

    def test_create_from_schema_includes_id_column(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that _id column is added to schema-based tables."""
        ddl_ops = sut_ddl(fake_conn)

        schema = {
            "properties": {
                "id": {"type": "integer"},
            },
        }

        ddl_ops.create_table("users", schema)

        # Should include _id column
        assert any(
            "_id VARCHAR" in str(call_args[0])
            for (call_args, _) in fake_conn.execute.call_args_list
        )

    def test_create_from_schema_validates_table_name(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that table name is validated for schema-based creation."""
        ddl_ops = sut_ddl(fake_conn)

        schema = {"properties": {"id": {"type": "integer"}}}

        with pytest.raises(ValueError, match="Invalid table name"):
            ddl_ops.create_table("123invalid", schema)


class TestCreateTableWithPartitioning:
    """Test table creation with partitioning."""

    def test_create_with_part_columns(self, sut_ddl: Any, fake_conn: MagicMock) -> None:
        """Test table creation with partition columns."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        ddl_ops.create_table("users", df, part_columns=["region", "year"])

        # Should have called execute for both CREATE TABLE and ALTER TABLE
        assert fake_conn.execute.called

        # Check for ALTER TABLE PARTITIONED BY
        partition_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert any(
            "ALTER TABLE" in call and "PARTITIONED BY" in call
            for call in partition_calls
        )

    def test_create_with_empty_part_columns(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that empty partition columns list is handled."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        ddl_ops.create_table("users", df, part_columns=[])

        # Should not make ALTER TABLE call for empty partition list
        partition_calls = [
            str(call_args[0]) for (call_args, _) in fake_conn.execute.call_args_list
        ]
        assert not any(
            "ALTER TABLE" in call and "PARTITIONED BY" in call
            for call in partition_calls
        )

    def test_create_with_invalid_partition_column_name(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that invalid partition column names raise error."""
        ddl_ops = sut_ddl(fake_conn)

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        # Invalid column name with dash
        with pytest.raises(SQLInjectionError):  # SQLInjectionError
            ddl_ops.create_table("users", df, part_columns=["invalid-column"])


class TestCreateTableTypeErrors:
    """Test error handling for invalid source types."""

    def test_create_with_invalid_source_type(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that invalid source type raises TypeError."""
        ddl_ops = sut_ddl(fake_conn)

        with pytest.raises(TypeError, match="source must be a DataFrame"):
            ddl_ops.create_table("users", 12345)

    def test_create_with_list_source_raises_error(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that list source raises TypeError."""
        ddl_ops = sut_ddl(fake_conn)

        with pytest.raises(TypeError, match="source must be a DataFrame"):
            ddl_ops.create_table("users", [1, 2, 3])


class TestDropTable:
    """Test table dropping with environment restrictions."""

    def test_drop_table_succeeds_in_test_environment(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that drop_table succeeds when DAPLA_ENVIRONMENT=test."""
        import os
        from unittest.mock import patch

        ddl_ops = sut_ddl(fake_conn)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            ddl_ops.drop_table("users")

            # Verify execute was called with DROP TABLE
            fake_conn.execute.assert_called()
            call_args = fake_conn.execute.call_args[0][0]
            assert "DROP TABLE users" in call_args

    def test_drop_table_fails_in_prod_environment(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that drop_table raises PermissionError when DAPLA_ENVIRONMENT=prod."""
        import os
        from unittest.mock import patch

        ddl_ops = sut_ddl(fake_conn)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "prod"}):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                ddl_ops.drop_table("users")

    def test_drop_table_fails_when_environment_not_set(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that drop_table raises PermissionError when DAPLA_ENVIRONMENT is not set."""
        import os
        from unittest.mock import patch

        ddl_ops = sut_ddl(fake_conn)

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(PermissionError, match="only allowed in TEST"):
                ddl_ops.drop_table("users")

    def test_drop_table_case_insensitive_test_env(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that TEST environment check is case-insensitive."""
        import os
        from unittest.mock import patch

        ddl_ops = sut_ddl(fake_conn)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "TEST"}):
            ddl_ops.drop_table("users")
            fake_conn.execute.assert_called()

    def test_drop_table_validates_table_name(
        self, sut_ddl: Any, fake_conn: MagicMock
    ) -> None:
        """Test that drop_table validates the table name."""
        import os
        from unittest.mock import patch

        ddl_ops = sut_ddl(fake_conn)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            with pytest.raises(ValueError, match="Invalid table name"):
                ddl_ops.drop_table("123invalid")

    def test_drop_table_with_cleanup_disabled(
        self, sut_ddl: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that drop_table with cleanup=False skips cleanup operations."""
        import os
        from unittest.mock import patch

        ddl_ops = sut_ddl(fake_conn, db_config)

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            ddl_ops.drop_table("users", cleanup=False)

            # Verify DROP TABLE was called
            fake_conn.execute.assert_called()
            call_args = fake_conn.execute.call_args[0][0]
            assert "DROP TABLE users" in call_args

    def test_drop_table_with_cleanup_enabled(
        self, sut_ddl: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that drop_table with cleanup=True attempts cleanup operations."""
        import os
        from unittest.mock import patch, MagicMock as MockMock

        db_config_with_path = db_config.copy()
        db_config_with_path["data_path"] = "gs://test-bucket/test-path"

        ddl_ops = sut_ddl(fake_conn, db_config_with_path)

        # Mock table location retrieval
        fake_conn.execute.return_value = MockMock()
        fake_conn.execute.return_value.fetchall.return_value = [
            ("gs://test-bucket/test-path/users",)
        ]

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            with patch("ssb_parquedit.ddl.gcsfs.GCSFileSystem") as mock_gcs:
                mock_fs = MockMock()
                mock_fs.exists.return_value = True
                mock_gcs.return_value = mock_fs

                ddl_ops.drop_table("users", cleanup=True)

                # Verify DROP TABLE was called
                assert any("DROP TABLE users" in str(call) for call in fake_conn.execute.call_args_list)
                # Verify GCS cleanup was attempted
                mock_fs.rm.assert_called()

    def test_drop_table_cleanup_falls_back_to_constructed_path(
        self, sut_ddl: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that cleanup falls back to constructed path if location query fails."""
        import os
        from unittest.mock import patch, MagicMock as MockMock

        db_config_with_path = db_config.copy()
        db_config_with_path["data_path"] = "gs://test-bucket/test-path"

        ddl_ops = sut_ddl(fake_conn, db_config_with_path)

        # Mock execute to fail for table location query but succeed for DROP
        def execute_side_effect(*args: Any, **kwargs: Any) -> Any:
            if "SELECT location" in str(args[0]):
                raise Exception("Query failed")
            return MockMock()

        fake_conn.execute.side_effect = execute_side_effect

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            with patch("ssb_parquedit.ddl.gcsfs.GCSFileSystem") as mock_gcs:
                mock_fs = MockMock()
                mock_fs.exists.return_value = True
                mock_gcs.return_value = mock_fs

                # Should succeed and use fallback path: gs://test-bucket/test-path/users
                ddl_ops.drop_table("users", cleanup=True)

                # Verify GCS cleanup used fallback path
                expected_path = "gs://test-bucket/test-path/users"
                assert any(
                    expected_path in str(call) for call in mock_fs.rm.call_args_list
                )

    def test_drop_table_cleanup_handles_missing_gcs_path(
        self, sut_ddl: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that cleanup gracefully handles missing GCS paths."""
        import os
        from unittest.mock import patch, MagicMock as MockMock

        ddl_ops = sut_ddl(fake_conn, db_config)

        def execute_side_effect(*args: Any, **kwargs: Any) -> Any:
            if "SELECT location" in str(args[0]):
                raise Exception("Query failed")
            return MockMock()

        fake_conn.execute.side_effect = execute_side_effect

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            # Should succeed even though cleanup will fail due to no db_config data_path
            ddl_ops.drop_table("users", cleanup=True)

            # Verify DROP TABLE was still called
            assert any("DROP TABLE users" in str(call) for call in fake_conn.execute.call_args_list)

    def test_drop_table_cleanup_handles_invalid_gcs_path_format(
        self, sut_ddl: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that cleanup handles invalid GCS path format."""
        import os
        from unittest.mock import patch, MagicMock as MockMock

        db_config_invalid = db_config.copy()
        db_config_invalid["data_path"] = "/invalid/local/path"

        ddl_ops = sut_ddl(fake_conn, db_config_invalid)

        fake_conn.execute.return_value = MockMock()
        fake_conn.execute.return_value.fetchall.return_value = [
            ("/invalid/local/path/users",)
        ]

        with patch.dict(os.environ, {"DAPLA_ENVIRONMENT": "test"}):
            # Should succeed despite invalid path format
            ddl_ops.drop_table("users", cleanup=True)

            # Verify DROP TABLE was called
            assert any("DROP TABLE users" in str(call) for call in fake_conn.execute.call_args_list)
