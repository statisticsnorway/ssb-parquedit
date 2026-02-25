import sys
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import call

import pytest

# Fixtures are imported from conftest.py: stub_external_modules, sut, fake_conn, db_config, query_test_setup


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


# -------------------- view tests --------------------
# Most view parameter combinations are covered by test_view_limit_offset_combinations parametrized test.
# This section tests specific query building and validation logic.


def test_view_select_specific_columns(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test selecting specific columns."""
    pe, fake_conn = query_test_setup

    pe.view("my_table", columns=["id", "name", "age"])

    query = fake_conn.execute.call_args[0][0]
    assert "SELECT id, name, age FROM my_table" in query


def test_view_with_filters(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test view with structured filters."""
    pe, fake_conn = query_test_setup

    pe.view("my_table", filters={"column": "age", "operator": ">", "value": 25})

    query = fake_conn.execute.call_args[0][0]
    assert "WHERE age > ?" in query


def test_view_with_order_by(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test view with ORDER BY clause."""
    pe, fake_conn = query_test_setup

    pe.view("my_table", order_by="created_at DESC")

    query = fake_conn.execute.call_args[0][0]
    assert "ORDER BY created_at DESC" in query


def test_view_with_all_parameters(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test view with all parameters combined."""
    pe, fake_conn = query_test_setup

    pe.view(
        "my_table",
        limit=100,
        offset=50,
        columns=["id", "name"],
        filters=[
            {"column": "age", "operator": ">", "value": 25},
            {"column": "status", "operator": "=", "value": "active"},
        ],
        order_by="name ASC",
    )

    query = fake_conn.execute.call_args[0][0]
    assert "SELECT id, name FROM my_table" in query
    assert "WHERE age > ? AND status = ?" in query
    assert "ORDER BY name ASC" in query
    assert "LIMIT 100" in query
    assert "OFFSET 50" in query


def test_view_sql_clause_order(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test that query clauses appear in correct SQL order."""
    pe, fake_conn = query_test_setup

    pe.view(
        "my_table",
        columns=["id"],
        filters=[
            {"column": "age", "operator": ">", "value": 25},
            {"column": "city", "operator": "=", "value": "Oslo"},
        ],
        order_by="id",
        limit=10,
        offset=5,
    )

    query = fake_conn.execute.call_args[0][0]

    # Check order: SELECT ... FROM ... WHERE ... ORDER BY ... LIMIT ... OFFSET
    where_pos = query.find("WHERE")
    order_pos = query.find("ORDER BY")
    limit_pos = query.find("LIMIT")
    offset_pos = query.find("OFFSET")

    assert where_pos < order_pos < limit_pos < offset_pos


def test_view_validates_table_name(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test that invalid table names raise ValueError."""
    pe = sut(db_config=db_config, conn=fake_conn)

    with pytest.raises(ValueError, match="Invalid table name"):
        pe.view("invalid-table-name")

    # Execute should not be called if validation fails
    fake_conn.execute.assert_not_called()


def test_view_empty_columns_list_selects_all(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test that empty columns list behaves like None (select all)."""
    pe, fake_conn = query_test_setup

    pe.view("my_table", columns=[])

    query = fake_conn.execute.call_args[0][0]
    assert "SELECT * FROM my_table" in query


@pytest.mark.parametrize(
    "limit,offset,has_limit,has_offset",
    [
        (5, 0, True, False),  # custom limit only
        (10, 10, True, True),  # limit with offset
        (None, 0, False, False),  # no limit, no offset
        (100, 50, True, True),  # large limit with offset
        (0, 0, True, False),  # zero limit (should still appear in LIMIT clause)
        (10, 20, True, True),  # standard pagination
    ],
)
def test_view_limit_offset_combinations(
    query_test_setup: tuple[Any, MagicMock],
    limit: int | None,
    offset: int,
    has_limit: bool,
    has_offset: bool,
) -> None:
    """Test various combinations of limit and offset parameters.

    Covers:
    - test_view_with_custom_limit (limit=5)
    - test_view_with_no_limit (limit=None)
    - test_view_with_offset (limit=10, offset=20)
    - test_view_offset_zero_not_included (offset=0 case)
    """
    pe, fake_conn = query_test_setup

    pe.view("my_table", limit=limit, offset=offset)

    query = fake_conn.execute.call_args[0][0]

    if has_limit:
        assert "LIMIT" in query, f"Expected LIMIT in query when limit={limit}"
    else:
        assert "LIMIT" not in query, f"Expected no LIMIT when limit={limit}"

    if has_offset:
        assert "OFFSET" in query, f"Expected OFFSET in query when offset={offset}"
    else:
        assert "OFFSET" not in query, f"Expected no OFFSET when offset={offset}"


def test_view_returns_dataframe_result(
    sut: Any, fake_conn: MagicMock, db_config: dict[str, str]
) -> None:
    """Test that view returns the result of execute().df()."""
    pe = sut(db_config=db_config, conn=fake_conn)

    expected_df = MagicMock()
    mock_result = MagicMock()
    mock_result.df.return_value = expected_df
    fake_conn.execute.return_value = mock_result

    result = pe.view("my_table")

    assert result is expected_df
    mock_result.df.assert_called_once()


def test_view_with_multiple_and_filters(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test view with multiple AND filters."""
    pe, fake_conn = query_test_setup

    pe.view(
        "my_table",
        filters=[
            {"column": "age", "operator": ">", "value": 25},
            {"column": "city", "operator": "=", "value": "Oslo"},
        ],
    )

    query = fake_conn.execute.call_args[0][0]
    assert "WHERE age > ? AND city = ?" in query


def test_view_with_multiple_order_by_columns(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test ORDER BY with multiple columns."""
    pe, fake_conn = query_test_setup

    pe.view("my_table", order_by="age DESC, name ASC")

    query = fake_conn.execute.call_args[0][0]
    assert "ORDER BY age DESC, name ASC" in query


def test_view_offset_without_limit(
    query_test_setup: tuple[Any, MagicMock],
) -> None:
    """Test that offset works even when limit is None."""
    pe, fake_conn = query_test_setup

    pe.view("my_table", limit=None, offset=50)

    query = fake_conn.execute.call_args[0][0]
    assert "OFFSET 50" in query
    assert "LIMIT" not in query
