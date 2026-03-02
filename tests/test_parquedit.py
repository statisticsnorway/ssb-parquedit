import sys
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import call

# Fixtures are imported from conftest.py: stub_external_modules, sut, fake_conn, db_config


# -------------------- Behavior tests --------------------


def test_init_registers_fs_and_loads_extensions_and_uses_catalog(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    sut(db_config=db_config, conn=fake_conn)

    # Should not call duckdb.connect when conn is provided
    # Assert initial setup interactions
    assert fake_conn.register_filesystem.called, "Filesystem should be registered"

    # Extensions install/load for ducklake and postgres
    expected_ext_calls = [
        call("INSTALL ducklake"),
        call("LOAD ducklake"),
        call("INSTALL postgres"),
        call("LOAD postgres"),
    ]
    # Ensure the order is preserved as implemented
    fake_conn.sql.assert_has_calls(expected_ext_calls, any_order=False)

    # Attach and USE catalog
    # We don't reconstruct the exact multi-line string; just ensure calls occurred and last USE is correct
    assert any(
        "ATTACH 'ducklake:postgres" in args[0]
        for (args, _) in fake_conn.sql.call_args_list
    ), "ATTACH call missing"
    assert any(
        f"USE {db_config['catalog_name']}" in args[0]
        for (args, _) in fake_conn.sql.call_args_list
    ), "USE catalog call missing"


def test_context_manager_closes_only_if_owns_connection(
    sut: Any, db_config: dict[str, str]
) -> None:
    # Case 1: Owns connection -> close called on exit (no conn passed => owns)
    pe1 = sut(db_config=db_config)
    with pe1:
        pass
    pe1._connection._conn.close.assert_called_once()

    # Case 2: Manually close closes if owns
    prev_calls = pe1._connection._conn.close.call_count
    pe1.close()
    assert pe1._connection._conn.close.call_count == prev_calls + 1


def test_create_table_from_dataframe_routes_and_applies_flags(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    # Create a fake DataFrame instance compatible with our Fake pandas
    DF = sys.modules["pandas"].DataFrame
    source = DF()

    pe = sut(db_config=db_config, conn=fake_conn)

    # Spy on internal helpers
    pe._create_from_dataframe = MagicMock()
    pe._add_table_partition = MagicMock()
    pe.fill_table = MagicMock()
    pe._add_table_description = MagicMock()

    pe.create_table(
        table_name="t",
        source=source,
        part_columns=["c1", "c2"],
        fill=True,
    )

    pe._create_from_dataframe.assert_called_once_with("t", source)
    pe._add_table_partition.assert_called_once_with("t", ["c1", "c2"])
    pe.fill_table.assert_called_once_with("t", source)
    pe._add_table_description.assert_called_once_with("t", "desc")


def test_create_table_from_parquet_routes_and_applies_flags(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    pe = sut(db_config=db_config, conn=fake_conn)

    pe._create_from_parquet = MagicMock()
    pe._add_table_partition = MagicMock()
    pe.fill_table = MagicMock()
    pe._add_table_description = MagicMock()

    pe.create_table(
        table_name="t",
        source="gs://bucket/path/file.parquet",
        # table_description="desc",
        part_columns=None,  # should treat as [] and not call _add_table_partition
        fill=False,
    )

    pe._create_from_parquet.assert_called_once()
    pe._add_table_partition.assert_not_called()
    pe.fill_table.assert_not_called()
    pe._add_table_description.assert_called_once_with("t", "desc")


def test_fill_table_routes_to_correct_helper(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    pe = sut(db_config=db_config, conn=fake_conn)
    pe._fill_from_dataframe = MagicMock()
    pe._fill_from_parquet = MagicMock()

    DF = sys.modules["pandas"].DataFrame
    df = DF()
    pe.fill_table("t", df)
    pe._fill_from_dataframe.assert_called_once_with("t", data=df)

    pe.fill_table("t", "gs://bucket/data.parquet")
    pe._fill_from_parquet.assert_called_once_with(
        "t", parquet_path="gs://bucket/data.parquet"
    )


def test_create_from_dataframe_registers_and_creates_empty_table(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    pe = sut(db_config=db_config, conn=fake_conn)

    DF = sys.modules["pandas"].DataFrame
    df = DF()

    pe._create_from_dataframe("mytable", df)

    fake_conn.register.assert_called_once_with("data", df)
    # Ensure the CREATE TABLE ... WHERE 1=2 pattern is used
    assert any(
        call_args[0].startswith("CREATE TABLE mytable AS SELECT * FROM data WHERE 1=2")
        for (call_args, _) in fake_conn.execute.call_args_list
    )


def test_add_table_partition_executes_alter_table(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    pe = sut(db_config=db_config, conn=fake_conn)

    pe._add_table_partition("t", ["a", "b"])

    fake_conn.execute.assert_called_with("ALTER TABLE t SET PARTITIONED BY (a,b);")


def test_add_table_description_executes_comment(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    pe = sut(db_config=db_config, conn=fake_conn)

    pe._add_table_description("t", "some desc")

    fake_conn.execute.assert_called_with("COMMENT ON TABLE t IS 'some desc';")


# -------------------- Delegation Tests --------------------


class TestParquEditDelegation:
    """Test that ParquEdit properly delegates to internal operations."""

    def test_create_table_delegates_to_ddl(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that create_table delegates to DDL operations."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the DDL operations
        pe._ddl.create_table = MagicMock()
        pe._dml.insert_data = MagicMock()

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        pe.create_table("users", df)

        # Should call DDL create_table
        pe._ddl.create_table.assert_called_once()

    def test_create_table_with_fill_calls_insert(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that create_table with fill=True calls insert_data."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the operations
        pe._ddl.create_table = MagicMock()
        pe._dml.insert_data = MagicMock()

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        pe.create_table("users", df, fill=True)

        # Should call insert_data
        pe._dml.insert_data.assert_called_once_with("users", df)

    def test_create_table_without_fill_does_not_insert(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that create_table with fill=False does not call insert."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the operations
        pe._ddl.create_table = MagicMock()
        pe._dml.insert_data = MagicMock()

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        pe.create_table("users", df, fill=False)

        # Should not call insert_data
        pe._dml.insert_data.assert_not_called()

    def test_insert_data_delegates_to_dml(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that insert_data delegates to DML operations."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the DML operations
        pe._dml.insert_data = MagicMock()

        DF = sys.modules["pandas"].DataFrame
        df = DF()

        pe.insert_data("users", df)

        # Should call DML insert_data
        pe._dml.insert_data.assert_called_once_with("users", df)

    def test_view_delegates_to_query(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that view delegates to Query operations."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the Query operations
        pe._query.view = MagicMock(return_value=MagicMock())

        pe.view("users", limit=10)

        # Should call Query view
        pe._query.view.assert_called_once()

    def test_view_passes_all_parameters(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that view passes all parameters to Query operations."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the Query operations
        pe._query.view = MagicMock(return_value=MagicMock())

        filters = {"column": "id", "operator": ">", "value": 10}
        pe.view(
            "users",
            limit=20,
            offset=5,
            columns=["id", "name"],
            filters=filters,
            order_by="id ASC",
        )

        # Should pass all parameters
        call_args = pe._query.view.call_args
        assert call_args[0][0] == "users"
        assert call_args[1]["limit"] == 20
        assert call_args[1]["offset"] == 5

    def test_count_delegates_to_query(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that count delegates to Query operations."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the Query operations
        pe._query.count = MagicMock(return_value=42)

        result = pe.count("users")

        # Should call Query count and return result
        pe._query.count.assert_called_once_with("users", None)
        assert result == 42

    def test_count_with_filters(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that count passes filters to Query operations."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the Query operations
        pe._query.count = MagicMock(return_value=10)

        filters = {"column": "status", "operator": "=", "value": "active"}
        result = pe.count("users", filters=filters)

        # Should pass filters
        pe._query.count.assert_called_once_with("users", filters)
        assert result == 10

    def test_exists_delegates_to_query(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that exists delegates to Query operations."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Mock the Query operations
        pe._query.table_exists = MagicMock(return_value=True)

        result = pe.exists("users")

        # Should call Query table_exists
        pe._query.table_exists.assert_called_once_with("users")
        assert result is True


# -------------------- Context Manager Tests --------------------


class TestParquEditContextManager:
    """Test ParquEdit as context manager."""

    def test_context_manager_enter_returns_self(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that __enter__ returns the instance."""
        pe = sut(db_config=db_config, conn=fake_conn)

        with pe as context_pe:
            assert context_pe is pe

    def test_context_manager_exit_closes_connection(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that __exit__ calls close."""
        pe = sut(db_config=db_config, conn=fake_conn)

        # Reset to clear init calls
        fake_conn.reset_mock()

        with pe:
            pass

        # Should not close external connection
        fake_conn.close.assert_not_called()

    def test_context_manager_closes_owned_connection(
        self, sut: Any, db_config: dict[str, str]
    ) -> None:
        """Test that __exit__ closes connection if owned."""
        pe = sut(db_config=db_config)  # No conn provided, so it's owned

        with pe:
            pass

        # Should have closed the connection
        pe._connection._conn.close.assert_called()

    def test_manual_close_works_outside_context(
        self, sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
    ) -> None:
        """Test that close can be called manually."""
        pe = sut(db_config=db_config, conn=fake_conn)

        fake_conn.reset_mock()
        pe.close()

        # Should not close external connection
        fake_conn.close.assert_not_called()
