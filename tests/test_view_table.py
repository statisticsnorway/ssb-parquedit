import importlib
import sys
from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import call

import pytest


# ---- Test scaffolding: stub external modules before importing the SUT ----
@pytest.fixture(autouse=True)
def stub_external_modules(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None, None, None]:
    """Stub external heavy dependencies (duckdb, gcsfs, pandas) so tests run hermetically.

    We inject minimal fakes into sys.modules prior to importing ssb_parquedit.parquedit.
    """

    # Fake duckdb module with minimal API surface
    class FakeDuckDB:
        class DuckDBPyConnection:  # only for type hints; runtime uses MagicMock
            pass

        def connect(self) -> MagicMock:
            # Return a MagicMock connection to simulate owned connections
            conn = MagicMock()
            conn.sql = MagicMock()
            conn.execute = MagicMock()
            conn.register = MagicMock()
            conn.register_filesystem = MagicMock()
            conn.close = MagicMock()
            return conn

    # Fake gcsfs module with a GCSFileSystem type
    class FakeGCSFS:
        class GCSFileSystem:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.created = True

    # Fake pandas with just a DataFrame type for isinstance checks
    class FakePandas:
        class DataFrame:
            pass

    monkeypatch.setitem(sys.modules, "duckdb", FakeDuckDB())
    monkeypatch.setitem(sys.modules, "gcsfs", FakeGCSFS())
    monkeypatch.setitem(sys.modules, "pandas", FakePandas())

    yield

    # Cleanup is automatic by pytest monkeypatch fixture


@pytest.fixture
def sut() -> Any:
    """Import and return the ParquEdit class with stubs injected."""
    module = importlib.import_module("ssb_parquedit.parquedit")
    importlib.reload(module)
    return module.ParquEdit


@pytest.fixture
def fake_conn() -> MagicMock:
    """A MagicMock simulating a DuckDB connection."""
    conn = MagicMock()
    # Provide attributes/methods that ParquEdit expects
    # - register_filesystem(fs)
    # - sql(str)
    # - execute(str)
    # - register(name, obj)
    # - close()
    # Use wraps to capture SQL calls distinctly
    conn.sql = MagicMock()
    conn.execute = MagicMock()
    conn.register = MagicMock()
    conn.register_filesystem = MagicMock()
    conn.close = MagicMock()
    return conn


@pytest.fixture
def db_config() -> dict[str, str]:
    return {
        "dbname": "testdb",
        "dbuser": "testuser",
        "catalog_name": "testcat",
        "data_path": "gs://bucket/path",
        "metadata_schema": "meta_schema",
    }


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
    pe1._conn.close.assert_called_once()

    # Case 2: Manually close closes if owns
    prev_calls = pe1._conn.close.call_count
    pe1.close()
    assert pe1._conn.close.call_count == prev_calls + 1


@pytest.mark.parametrize(
    "name,valid",
    [
        ("valid_name", True),
        ("_underscore_ok", True),
        ("1starts_with_digit", False),
        ("has-dash", False),
        ("has space", False),
    ],
)
def test_validate_table_name(sut: Any, name: str, valid: bool) -> None:
    if valid:
        sut._validate_table_name(name)  # should not raise
    else:
        with pytest.raises(ValueError):
            sut._validate_table_name(name)


@pytest.mark.parametrize(
    "prop,expected",
    [
        ({"type": "string"}, "VARCHAR"),
        ({"type": ["null", "string"]}, "VARCHAR"),
        ({"type": "string", "format": "date"}, "DATE"),
        ({"type": "string", "format": "date-time"}, "TIMESTAMP"),
        ({"type": "integer"}, "BIGINT"),
        ({"type": "number"}, "DOUBLE"),
        ({"type": "boolean"}, "BOOLEAN"),
        ({"type": "array", "items": {"type": "integer"}}, "LIST<BIGINT>"),
        (
            {
                "type": "object",
                "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            },
            "STRUCT(a VARCHAR, b BIGINT)",
        ),
        ({"type": "object"}, "JSON"),  # object with no properties -> JSON
        ({}, "JSON"),  # fallback
    ],
)
def test_translate_jsonschema_property(
    sut: Any, prop: dict[str, object], expected: str
) -> None:
    assert sut.translate(prop) == expected


def test_jsonschema_to_duckdb_builds_correct_ddl(sut: Any) -> None:
    schema = {
        "properties": {
            "id": {"type": "integer"},
            "name": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "meta": {
                "type": "object",
                "properties": {
                    "active": {"type": "boolean"},
                    "score": {"type": ["null", "number"]},
                },
            },
        },
        "required": ["id", "name"],
    }

    ddl = sut.jsonschema_to_duckdb(schema, "t")
    # Expected columns and constraints
    assert "CREATE TABLE t (" in ddl
    assert "id BIGINT NOT NULL" in ddl
    assert "name VARCHAR NOT NULL" in ddl
    assert "tags LIST<VARCHAR>" in ddl
    assert "meta STRUCT(active BOOLEAN, score DOUBLE)" in ddl
    assert ddl.strip().endswith(");"), "DDL should end with semicolon"


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
        table_description="desc",
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
        table_description="desc",
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


# -------------------- view_table tests --------------------


def test_view_table_default_parameters_builds_correct_query(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test view_table with default parameters builds correct SQL query."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    # Mock the result of execute().df()
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table")
    
    # Verify execute was called with correct query
    fake_conn.execute.assert_called_once()
    query = fake_conn.execute.call_args[0][0]
    
    assert "SELECT * FROM my_table" in query
    assert "LIMIT 10" in query
    assert "OFFSET" not in query


def test_view_table_with_custom_limit(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test view_table with custom limit."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", limit=5)
    
    query = fake_conn.execute.call_args[0][0]
    assert "LIMIT 5" in query


def test_view_table_with_no_limit(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test view_table with limit=None returns all rows."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", limit=None)
    
    query = fake_conn.execute.call_args[0][0]
    assert "LIMIT" not in query


def test_view_table_with_offset(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test view_table with offset."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", limit=10, offset=20)
    
    query = fake_conn.execute.call_args[0][0]
    assert "LIMIT 10" in query
    assert "OFFSET 20" in query


def test_view_table_offset_zero_not_included(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test that offset=0 doesn't add OFFSET clause."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", offset=0)
    
    query = fake_conn.execute.call_args[0][0]
    assert "OFFSET" not in query


def test_view_table_select_specific_columns(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test viewing specific columns."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", columns=["id", "name", "age"])
    
    query = fake_conn.execute.call_args[0][0]
    assert "SELECT id, name, age FROM my_table" in query


def test_view_table_with_where_clause(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test viewing table with WHERE clause."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", where="age > 25")
    
    query = fake_conn.execute.call_args[0][0]
    assert "WHERE age > 25" in query


def test_view_table_with_order_by(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test viewing table with ORDER BY clause."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", order_by="created_at DESC")
    
    query = fake_conn.execute.call_args[0][0]
    assert "ORDER BY created_at DESC" in query


def test_view_table_with_all_parameters(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test viewing table with all parameters combined."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table(
        "my_table",
        limit=100,
        offset=50,
        columns=["id", "name"],
        where="age > 25",
        order_by="name ASC"
    )
    
    query = fake_conn.execute.call_args[0][0]
    assert "SELECT id, name FROM my_table" in query
    assert "WHERE age > 25" in query
    assert "ORDER BY name ASC" in query
    assert "LIMIT 100" in query
    assert "OFFSET 50" in query


def test_view_table_sql_clause_order(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test that query clauses appear in correct SQL order."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table(
        "my_table",
        columns=["id"],
        where="age > 25",
        order_by="id",
        limit=10,
        offset=5
    )
    
    query = fake_conn.execute.call_args[0][0]
    
    # Check order: SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT ... OFFSET
    where_pos = query.find("WHERE")
    order_pos = query.find("ORDER BY")
    limit_pos = query.find("LIMIT")
    offset_pos = query.find("OFFSET")
    
    assert where_pos < order_pos < limit_pos < offset_pos


def test_view_table_validates_table_name(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test that invalid table names raise ValueError."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    with pytest.raises(ValueError, match="Invalid table name"):
        pe.view_table("invalid-table-name")
    
    # Execute should not be called if validation fails
    fake_conn.execute.assert_not_called()


def test_view_table_empty_columns_list_selects_all(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test that empty columns list behaves like None (select all)."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", columns=[])
    
    query = fake_conn.execute.call_args[0][0]
    assert "SELECT * FROM my_table" in query


@pytest.mark.parametrize(
    "limit,offset,has_limit,has_offset",
    [
        (5, 0, True, False),
        (10, 10, True, True),
        (None, 0, False, False),
        (100, 50, True, True),
        (0, 0, True, False),
    ],
)
def test_view_table_limit_offset_combinations(
    sut: Any, 
    fake_conn: MagicMock, 
    db_config: dict[str, str],
    limit: int | None,
    offset: int,
    has_limit: bool,
    has_offset: bool
) -> None:
    """Test various combinations of limit and offset."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", limit=limit, offset=offset)
    
    query = fake_conn.execute.call_args[0][0]
    
    if has_limit:
        assert "LIMIT" in query
    else:
        assert "LIMIT" not in query
    
    if has_offset:
        assert "OFFSET" in query
    else:
        assert "OFFSET" not in query


def test_view_table_returns_dataframe_result(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test that view_table returns the result of execute().df()."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    expected_df = MagicMock()
    mock_result = MagicMock()
    mock_result.df.return_value = expected_df
    fake_conn.execute.return_value = mock_result
    
    result = pe.view_table("my_table")
    
    assert result is expected_df
    mock_result.df.assert_called_once()


def test_view_table_complex_where_clause(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test viewing table with complex WHERE clause."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", where="age > 25 AND city = 'Oslo'")
    
    query = fake_conn.execute.call_args[0][0]
    assert "WHERE age > 25 AND city = 'Oslo'" in query


def test_view_table_multiple_order_by_columns(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test ORDER BY with multiple columns."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", order_by="age DESC, name ASC")
    
    query = fake_conn.execute.call_args[0][0]
    assert "ORDER BY age DESC, name ASC" in query


def test_view_table_offset_without_limit(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test that offset works even when limit is None."""
    pe = sut(db_config=db_config, conn=fake_conn)
    
    mock_result = MagicMock()
    mock_result.df.return_value = MagicMock()
    fake_conn.execute.return_value = mock_result
    
    pe.view_table("my_table", limit=None, offset=50)
    
    query = fake_conn.execute.call_args[0][0]
    assert "OFFSET 50" in query
    assert "LIMIT" not in query