"""Unit tests for SchemaUtils."""

import pandas as pd
import polars as pl
import pytest

from ssb_parquedit.utils import SchemaUtils

# ── is_dataframe ────────────────────────────────────────────────────────────────


class TestIsDataframe:
    """is_dataframe() must recognize both pandas and polars DataFrames."""

    def test_true_for_pandas_dataframe(self) -> None:
        assert SchemaUtils.is_dataframe(pd.DataFrame({"id": [1]})) is True

    def test_true_for_polars_dataframe(self) -> None:
        assert SchemaUtils.is_dataframe(pl.DataFrame({"id": [1]})) is True

    @pytest.mark.parametrize(
        "value", [{"id": [1]}, "gs://bucket/data.parquet", [1, 2], None]
    )
    def test_false_for_non_dataframe_values(self, value: object) -> None:
        assert SchemaUtils.is_dataframe(value) is False


# ── validate_column_names ──────────────────────────────────────────────────────


class TestValidateColumnNames:
    """validate_column_names() enforces PostgreSQL's 63-byte identifier limit."""

    def test_accepts_short_column_names(self) -> None:
        SchemaUtils.validate_column_names(["id", "name", "a" * 63])  # must not raise

    def test_raises_for_column_over_63_bytes(self) -> None:
        with pytest.raises(ValueError, match="63-byte"):
            SchemaUtils.validate_column_names(["a" * 64])

    def test_raises_for_multibyte_column_over_63_bytes(self) -> None:
        """64 chars of Norwegian text can be 67 UTF-8 bytes due to \u00e6/\u00f8/\u00e5."""
        name = "distriktstilskuddforfruktb\u00e6rveksthusgr\u00f8nnsakerinklsalatp\u00e5friland"
        assert len(name) == 64
        assert len(name.encode("utf-8")) == 67
        with pytest.raises(ValueError, match="63-byte"):
            SchemaUtils.validate_column_names([name])

    def test_error_message_reports_byte_length(self) -> None:
        with pytest.raises(ValueError, match=r"64 bytes"):
            SchemaUtils.validate_column_names(["a" * 64])

    def test_error_message_lists_only_offending_columns(self) -> None:
        with pytest.raises(ValueError) as exc_info:
            SchemaUtils.validate_column_names(["short_col", "a" * 64])
        assert "short_col" not in str(exc_info.value)
