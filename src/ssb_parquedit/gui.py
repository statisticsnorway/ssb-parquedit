"""Interactive ipywidgets GUI for editing and deleting rows in ParquEdit tables.

Designed to run inside a Jupyter notebook (e.g. on DaplaLab). It lets the user
pick a table, search for rows with a SQL ``WHERE`` filter, edit cell values, or
select whole rows for deletion. All writes go through the public ParquEdit API
(:meth:`ParquEdit.edit` and :meth:`ParquEdit.delete_row`) and are therefore
logged to the DuckLake changelog.

Example:
    >>> # doctest: +SKIP
    >>> from ssb_parquedit.gui import ParquEditGUI
    >>> ParquEditGUI()  # opens the GUI backed by a local catalog
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import ipywidgets as widgets
import pandas as pd
from IPython.display import display

from .parquedit import ParquEdit

logger = logging.getLogger(__name__)

#: Allowed values for ``change_event_reason`` accepted by ParquEdit.
CHANGE_EVENT_REASONS = [
    "OTHER_SOURCE",
    "REVIEW",
    "OWNER",
    "MARGINAL_UNIT",
    "DUPLICATE",
    "OTHER",
]

#: Name of the internal row identifier column that ParquEdit assigns.
ROWID_COLUMN = "rowid"

_CELL_WIDTH = "160px"
_SELECT_WIDTH = "60px"
_ROW_HEIGHT = "34px"


def _spinner_img() -> str:
    """Return an ``<img>`` tag with a self-animating SVG spinner (data URI).

    Uses SVG SMIL animation so it spins without relying on page CSS or
    ``@keyframes`` (which notebook HTML sanitizers may strip).
    """
    from urllib.parse import quote

    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 50 50'>"
        "<circle cx='25' cy='25' r='20' fill='none' stroke='#ccc' stroke-width='5'/>"
        "<path fill='none' stroke='#1f77b4' stroke-width='5' stroke-linecap='round'"
        " d='M25 5 a20 20 0 0 1 20 20'>"
        "<animateTransform attributeName='transform' type='rotate'"
        " from='0 25 25' to='360 25 25' dur='0.8s' repeatCount='indefinite'/>"
        "</path></svg>"
    )
    return (
        "<img width='14' height='14' "
        "style='vertical-align:middle;margin-right:6px' "
        f"src='data:image/svg+xml,{quote(svg)}'>"
    )


#: Pre-rendered spinner ``<img>`` shown while the GUI is querying/writing.
_SPINNER_IMG = _spinner_img()


def _px(value: str) -> int:
    """Return the integer pixel count from a CSS ``px`` string (e.g. ``"60px"``)."""
    return int(value.removesuffix("px"))


class ParquEditGUI:
    """An ipywidgets front-end for editing and deleting rows in ParquEdit tables.

    The GUI wraps a :class:`~ssb_parquedit.ParquEdit` instance and exposes:

    - a table selector populated from :meth:`ParquEdit.list_tables`,
    - a SQL ``WHERE`` search box (plus row limit) to fetch rows,
    - an editable grid where non-``rowid`` cells can be changed,
    - per-row selection checkboxes for deletion,
    - a shared ``change_comment`` and ``change_event_reason`` used when
      committing edits or deletions.
    """

    def __init__(
        self,
        con: ParquEdit | None = None,
        local_path: str | None = None,
        page_size: int = 50,
        auto_display: bool = True,
        load_on_start: bool = False,
    ) -> None:
        """Build the GUI and (optionally) render it.

        Args:
            con: An existing ParquEdit instance to use. When ``None`` a local
                SQLite-backed catalog is created via :meth:`ParquEdit.local`.
            local_path: Directory for the local catalog when ``con`` is ``None``.
                Ignored when ``con`` is provided.
            page_size: Default maximum number of rows to load per search.
            auto_display: If ``True``, immediately display the GUI (suitable for
                a notebook cell). Set to ``False`` to display it manually later.
            load_on_start: If ``True``, immediately load rows for the initially
                selected table. Defaults to ``False`` so the GUI renders quickly;
                rows load when a table is selected or *Search / Load* is clicked.
        """
        if con is not None:
            self.con = con
        elif local_path is not None:
            self.con = ParquEdit.local(local_path)
        else:
            self.con = ParquEdit.local()

        self.page_size = page_size
        self._df: pd.DataFrame | None = None
        self._columns: list[str] = []
        self._dtypes: dict[str, Any] = {}
        self._cell_widgets: dict[int, dict[str, widgets.Text]] = {}
        self._select_widgets: dict[int, widgets.Checkbox] = {}
        self._suppress_autoload = False
        self._where: str | None = None
        self._match_count = 0
        self._pending_delete_where: str | None = None

        self._build_widgets()
        self._refresh_tables()
        if self.table_dropdown.value:
            self._update_table_info(self.table_dropdown.value)
        if load_on_start and self.table_dropdown.value:
            self._on_load()

        if auto_display:
            self.display()

    # ============ Public API ============

    def display(self) -> None:
        """Render the GUI in the current notebook output cell."""
        display(self.ui)  # type: ignore[no-untyped-call]

    # ============ Widget construction ============

    def _build_widgets(self) -> None:
        """Create all widgets and assemble the root container ``self.ui``."""
        self.table_dropdown = widgets.Dropdown(
            description="Table:",
            options=[],
            layout=widgets.Layout(width="320px"),
        )
        self.refresh_tables_btn = widgets.Button(
            description="Refresh tables",
            icon="rotate-right",
            layout=widgets.Layout(width="150px"),
        )
        self.table_info_html = widgets.HTML()

        self.where_text = widgets.Text(
            description="WHERE:",
            placeholder="e.g. age > 25 AND name = 'Alice'",
            layout=widgets.Layout(width="520px"),
        )
        self.limit_int = widgets.BoundedIntText(
            description="Limit:",
            value=self.page_size,
            min=1,
            max=10000,
            layout=widgets.Layout(width="150px"),
        )
        self.load_btn = widgets.Button(
            description="Search / Load",
            button_style="primary",
            icon="magnifying-glass",
            layout=widgets.Layout(width="150px"),
        )
        self.sort_chk = widgets.Checkbox(
            value=False,
            description="Sort by rowid",
            indent=False,
            layout=widgets.Layout(width="150px"),
        )
        self.result_info_html = widgets.HTML()

        self.reason_dropdown = widgets.Dropdown(
            description="Reason:",
            options=CHANGE_EVENT_REASONS,
            value=CHANGE_EVENT_REASONS[0],
            layout=widgets.Layout(width="320px"),
        )
        self.comment_text = widgets.Text(
            description="Comment:",
            placeholder="Describe why you are making this change",
            layout=widgets.Layout(width="520px"),
        )

        self.select_all_chk = widgets.Checkbox(
            value=False,
            description="Select all",
            indent=False,
            layout=widgets.Layout(width="120px"),
        )
        self.save_btn = widgets.Button(
            description="Save edits",
            button_style="success",
            icon="floppy-disk",
            layout=widgets.Layout(width="150px"),
        )
        self.delete_btn = widgets.Button(
            description="Delete selected",
            button_style="danger",
            icon="trash",
            layout=widgets.Layout(width="170px"),
        )
        self.delete_all_btn = widgets.Button(
            description="Delete ALL matching WHERE",
            button_style="danger",
            icon="triangle-exclamation",
            layout=widgets.Layout(width="230px"),
        )
        self.show_log_btn = widgets.Button(
            description="Show edit log",
            icon="clock-rotate-left",
            layout=widgets.Layout(width="150px"),
        )

        # Confirmation strip for the destructive delete-all action.
        self.confirm_label = widgets.HTML()
        self.confirm_yes_btn = widgets.Button(
            description="Yes, delete", button_style="danger",
            layout=widgets.Layout(width="140px"),
        )
        self.confirm_cancel_btn = widgets.Button(
            description="Cancel", layout=widgets.Layout(width="100px")
        )
        self.confirm_box = widgets.HBox(
            [self.confirm_label, self.confirm_yes_btn, self.confirm_cancel_btn],
            layout=widgets.Layout(display="none", align_items="center"),
        )

        self.grid_box = widgets.VBox(
            layout=widgets.Layout(overflow="auto", max_height="500px")
        )
        self.busy_html = widgets.HTML()
        self.status_out = widgets.Output(
            layout=widgets.Layout(border="1px solid #ddd", padding="4px")
        )
        self.log_out = widgets.Output(
            layout=widgets.Layout(
                border="1px solid #ddd", padding="4px", overflow="auto"
            )
        )

        # Wire up event handlers.
        self.refresh_tables_btn.on_click(self._on_refresh_tables)
        self.load_btn.on_click(self._on_load)
        self.save_btn.on_click(self._on_save)
        self.delete_btn.on_click(self._on_delete)
        self.delete_all_btn.on_click(self._on_delete_all_request)
        self.confirm_yes_btn.on_click(self._on_confirm_delete_all)
        self.confirm_cancel_btn.on_click(self._on_cancel_delete_all)
        self.show_log_btn.on_click(self._on_show_log)
        self.select_all_chk.observe(self._on_select_all, names="value")
        self.table_dropdown.observe(self._on_table_change, names="value")

        self.ui = widgets.VBox(
            [
                widgets.HTML("<h3>ParquEdit — edit &amp; delete rows</h3>"),
                widgets.HBox([self.table_dropdown, self.refresh_tables_btn]),
                self.table_info_html,
                widgets.HBox(
                    [self.where_text, self.limit_int, self.sort_chk, self.load_btn]
                ),
                self.result_info_html,
                widgets.HTML("<hr>"),
                widgets.HBox([self.reason_dropdown, self.comment_text]),
                widgets.HBox(
                    [
                        self.select_all_chk,
                        self.save_btn,
                        self.delete_btn,
                        self.delete_all_btn,
                        self.show_log_btn,
                        self.busy_html,
                    ]
                ),
                self.confirm_box,
                self.grid_box,
                self.status_out,
                widgets.HTML("<b>Edit log</b>"),
                self.log_out,
            ]
        )

    # ============ Data grid rendering ============

    def _frozen_columns(self, df: pd.DataFrame) -> list[str]:
        """Return the ``user_defined_id`` columns to freeze for the active table.

        Args:
            df: The currently loaded frame (used to keep only existing columns).

        Returns:
            The business-key columns (in ``user_defined_id`` order) that are
            present in ``df``; empty if the table has no stored key.
        """
        from .query import QueryOperations

        try:
            info = QueryOperations(self.con._get_connection())._get_tag_info(
                self.table_dropdown.value
            )
        except Exception:
            info = None
        uid = (info or {}).get("user_defined_id") or []
        return [c for c in uid if c in df.columns and c != ROWID_COLUMN]

    def _make_cell(self, rowid: int, col: str, value: Any) -> widgets.Text:
        """Create an editable cell widget and register it for later commits.

        Args:
            rowid: The row identifier the cell belongs to.
            col: The column name.
            value: The current value to display.

        Returns:
            The created :class:`ipywidgets.Text` widget.
        """
        text = widgets.Text(
            value="" if pd.isna(value) else str(value),
            layout=widgets.Layout(width=_CELL_WIDTH),
        )
        self._cell_widgets[rowid][col] = text
        return text

    def _render_grid(self, df: pd.DataFrame) -> None:
        """Render ``df`` in a single scrollable grid with frozen key columns.

        Select, ``rowid`` and the table's ``user_defined_id`` columns are pinned
        to the left with CSS ``position: sticky`` while the remaining columns
        scroll horizontally. Everything lives in one scroll container, so there
        is exactly one vertical and one horizontal scrollbar.

        Args:
            df: The rows to display. Must contain a ``rowid`` column.
        """
        self._df = df
        self._dtypes = {col: df[col].dtype for col in df.columns}
        self._columns = [c for c in df.columns if c != ROWID_COLUMN]
        self._cell_widgets = {}
        self._select_widgets = {}
        self.select_all_chk.value = False

        if ROWID_COLUMN not in df.columns:
            self.grid_box.children = [
                widgets.HTML(
                    "<b>Table has no 'rowid' column — editing/deleting is unavailable.</b>"
                )
            ]
            return

        frozen_cols = self._frozen_columns(df)
        scroll_cols = [c for c in self._columns if c not in frozen_cols]
        # Number of leading grid columns to freeze: Select + rowid + key columns.
        n_frozen = 2 + len(frozen_cols)

        def freeze(widget: widgets.Widget, idx: int) -> widgets.Widget:
            if idx < n_frozen:
                widget.add_class(f"pe-freeze-col{idx}")
            return widget

        def header(text: str, width: str = _CELL_WIDTH) -> widgets.HTML:
            cell = widgets.HTML(f"<b>{text}</b>", layout=widgets.Layout(width=width))
            cell.add_class("pe-header")
            return cell

        grid_children: list[widgets.Widget] = [
            freeze(header("Select", _SELECT_WIDTH), 0),
            freeze(header(ROWID_COLUMN), 1),
            *[freeze(header(c), 2 + k) for k, c in enumerate(frozen_cols)],
            *[header(c) for c in scroll_cols],
        ]

        for _, record in df.iterrows():
            rowid = int(record[ROWID_COLUMN])
            self._cell_widgets[rowid] = {}

            select_chk = widgets.Checkbox(
                value=False, indent=False, layout=widgets.Layout(width=_SELECT_WIDTH)
            )
            self._select_widgets[rowid] = select_chk
            rowid_label = widgets.HTML(
                f"<code>{rowid}</code>", layout=widgets.Layout(width=_CELL_WIDTH)
            )

            grid_children += [freeze(select_chk, 0), freeze(rowid_label, 1)]
            grid_children += [
                freeze(self._make_cell(rowid, col, record[col]), 2 + k)
                for k, col in enumerate(frozen_cols)
            ]
            grid_children += [
                self._make_cell(rowid, col, record[col]) for col in scroll_cols
            ]

        data_cols = len(frozen_cols) + len(scroll_cols)
        template = " ".join([_SELECT_WIDTH, _CELL_WIDTH, *([_CELL_WIDTH] * data_cols)])
        grid = widgets.GridBox(
            children=grid_children,
            layout=widgets.Layout(
                grid_template_columns=template, grid_auto_rows=_ROW_HEIGHT
            ),
        )

        children: list[widgets.Widget] = [self._freeze_style(n_frozen), grid]
        if len(df) == 0:
            children.append(widgets.HTML("<i>No rows matched.</i>"))
        self.grid_box.children = children

    def _freeze_style(self, n_frozen: int) -> widgets.HTML:
        """Build a ``<style>`` widget for sticky columns and a sticky header row.

        The first ``n_frozen`` columns are pinned horizontally (``left``), the
        header row is pinned vertically (``top``), and the corner header cells
        (frozen columns in the header) stick in both directions.

        Args:
            n_frozen: Number of leading columns to make horizontally sticky.

        Returns:
            An HTML widget containing the sticky CSS rules.
        """
        bg = "background:var(--jp-layout-color0, white);"
        widths = [_px(_SELECT_WIDTH), *([_px(_CELL_WIDTH)] * (n_frozen - 1))]
        rules = []
        offset = 0
        for j, width in enumerate(widths):
            rules.append(
                f".pe-freeze-col{j}{{position:sticky;left:{offset}px;z-index:5;{bg}}}"
            )
            offset += width
        # Header row sticks to the top; corner header cells (also frozen) sit
        # above everything so they stay visible on both scroll axes.
        rules.append(f".pe-header{{position:sticky;top:0;z-index:6;{bg}}}")
        for j in range(n_frozen):
            rules.append(f".pe-header.pe-freeze-col{j}{{z-index:7;}}")
        return widgets.HTML("<style>" + "".join(rules) + "</style>")

    # ============ Value coercion ============

    def _coerce(self, value_str: str, dtype: Any) -> Any:
        """Convert an edited string back to the column's underlying type.

        Args:
            value_str: The raw string value from the cell widget.
            dtype: The pandas dtype of the target column.

        Returns:
            ``None`` for an empty string, otherwise the value coerced to an int,
            float, bool or left as a string according to ``dtype``.
        """
        if value_str == "":
            return None
        if pd.api.types.is_bool_dtype(dtype):
            return value_str.strip().lower() in ("true", "1", "yes")
        if pd.api.types.is_integer_dtype(dtype):
            return int(value_str)
        if pd.api.types.is_float_dtype(dtype):
            return float(value_str)
        return value_str

    def _original_str(self, rowid: int, col: str) -> str:
        """Return the original display string for a cell in the loaded frame.

        Args:
            rowid: The row identifier.
            col: The column name.

        Returns:
            The original value rendered as a string (``""`` for nulls).
        """
        assert self._df is not None
        mask = self._df[ROWID_COLUMN] == rowid
        value = self._df.loc[mask, col].iloc[0]
        return "" if pd.isna(value) else str(value)

    # ============ Event handlers ============

    def _on_refresh_tables(self, _btn: widgets.Button | None = None) -> None:
        """Reload the list of available tables into the dropdown."""
        self._refresh_tables()

    def _refresh_tables(self) -> None:
        """Populate the table dropdown from ``con.list_tables()``."""
        with self.status_out:
            try:
                tables = self.con.list_tables()
            except Exception as exc:
                self._log(f"Could not list tables: {exc}", error=True)
                return
        current = self.table_dropdown.value
        self._suppress_autoload = True
        try:
            self.table_dropdown.options = tables
            if current in tables:
                self.table_dropdown.value = current
            elif tables:
                self.table_dropdown.value = tables[0]
        finally:
            self._suppress_autoload = False

    def _update_table_info(self, table: str) -> int | None:
        """Show the row count, ``user_defined_id`` and product for the table.

        Args:
            table: The table whose metadata should be shown.

        Returns:
            The total row count, or ``None`` if it could not be determined.
        """
        from .query import QueryOperations

        try:
            info = QueryOperations(self.con._get_connection())._get_tag_info(table)
        except Exception:
            info = None

        try:
            n_rows: int | None = self.con.count(table_name=table)
            rows_str = f"{n_rows:,}"
        except Exception:
            n_rows = None
            rows_str = "?"

        uid = (info or {}).get("user_defined_id") or []
        product = (info or {}).get("product_name") or ""
        uid_str = ", ".join(str(c) for c in uid) if uid else "&mdash;"
        self.table_info_html.value = (
            "<span style='color:#555'>"
            f"<b>rows:</b> <code>{rows_str}</code>"
            f" &nbsp;&nbsp; <b>user_defined_id:</b> <code>{uid_str}</code>"
            f" &nbsp;&nbsp; <b>product:</b> <code>{product}</code></span>"
        )
        return n_rows

    def _on_table_change(self, _change: dict[str, Any]) -> None:
        """Autoload rows when the selected table changes.

        Clears the WHERE filter first, since a filter valid for one table may
        reference columns absent from another.

        Args:
            _change: The ipywidgets observe payload (unused).
        """
        if self._suppress_autoload or not self.table_dropdown.value:
            return
        self.where_text.value = ""
        self._on_load()

    def _on_load(self, _btn: widgets.Button | None = None) -> None:
        """Load rows for the selected table using the WHERE filter and limit."""
        table = self.table_dropdown.value
        if not table:
            self._log("Select a table first.", error=True)
            return
        self.confirm_box.layout.display = "none"
        self._pending_delete_where = None
        where = self.where_text.value.strip() or None
        limit = self.limit_int.value
        # Sorting forces a full sort before LIMIT; skip it unless requested.
        order_by = ROWID_COLUMN if self.sort_chk.value else None
        try:
            with self._busy(f"Loading '{table}'…"):
                total = self._update_table_info(table)
                # Reuse the total count when there is no filter to avoid a
                # second full-table COUNT.
                if where is None and total is not None:
                    match_count = total
                else:
                    match_count = self.con.count(table_name=table, where=where)
                df = self.con.view(
                    table_name=table,
                    where=where,
                    limit=limit,
                    order_by=order_by,
                )
        except Exception as exc:
            self._log(f"Load failed: {exc}", error=True)
            return
        self._where = where
        self._match_count = match_count
        self._render_grid(df)
        self._update_result_info(where, match_count, len(df), limit)
        self._log(f"Loaded {len(df)} row(s) from '{table}'.")

    def _update_result_info(
        self, where: str | None, match_count: int, loaded: int, limit: int
    ) -> None:
        """Show how many rows the WHERE filter matches vs how many are loaded.

        Args:
            where: The WHERE clause used (``None`` when no filter).
            match_count: Total rows matching the filter (ignoring the limit).
            loaded: Number of rows currently loaded into the grid.
            limit: The row limit that was applied.
        """
        if where is None:
            self.result_info_html.value = ""
            return
        truncated = match_count > loaded
        color = "#b00" if truncated else "#555"
        note = (
            f" &nbsp; <b style='color:#b00'>&#9888; showing first {loaded} of "
            f"{match_count:,} (limit {limit}) — increase the limit to load/act on all</b>"
            if truncated
            else ""
        )
        self.result_info_html.value = (
            f"<span style='color:{color}'>"
            f"<b>WHERE matches:</b> <code>{match_count:,}</code> row(s)</span>{note}"
        )

    def _on_save(self, _btn: widgets.Button | None = None) -> None:
        """Commit all edited cells, one ``edit()`` call per changed row."""
        table = self.table_dropdown.value
        if self._df is None or not self._cell_widgets:
            self._log("Nothing loaded to save.", error=True)
            return
        reason, comment = self.reason_dropdown.value, self.comment_text.value.strip()
        if not comment:
            self._log("A change comment is required before saving.", error=True)
            return

        edited_rows = 0
        with self._busy("Saving edits…"):
            for rowid, col_widgets in self._cell_widgets.items():
                changes: dict[str, Any] = {}
                for col, widget in col_widgets.items():
                    if widget.value != self._original_str(rowid, col):
                        changes[col] = self._coerce(widget.value, self._dtypes[col])
                if not changes:
                    continue
                try:
                    self.con.edit(
                        table_name=table,
                        rowid=rowid,
                        changes=changes,
                        change_event_reason=reason,
                        change_comment=comment,
                    )
                    edited_rows += 1
                except Exception as exc:
                    self._log(f"Edit of rowid {rowid} failed: {exc}", error=True)
                    return

        if edited_rows == 0:
            self._log("No cell changes detected.")
            return
        self._log(f"Saved edits to {edited_rows} row(s).")
        self._on_load()

    def _on_delete(self, _btn: widgets.Button | None = None) -> None:
        """Delete all selected rows via a single ``delete_row()`` call."""
        table = self.table_dropdown.value
        if not self._select_widgets:
            self._log("Nothing loaded to delete.", error=True)
            return
        selected = [rid for rid, chk in self._select_widgets.items() if chk.value]
        if not selected:
            self._log("No rows selected for deletion.", error=True)
            return
        reason, comment = self.reason_dropdown.value, self.comment_text.value.strip()
        if not comment:
            self._log("A change comment is required before deleting.", error=True)
            return

        rowid_list = ", ".join(str(rid) for rid in selected)
        where = f"{ROWID_COLUMN} IN ({rowid_list})"
        try:
            with self._busy(f"Deleting {len(selected)} row(s)…"):
                self.con.delete_row(
                    table_name=table,
                    where=where,
                    change_event_reason=reason,
                    change_comment=comment,
                )
        except Exception as exc:
            self._log(f"Delete failed: {exc}", error=True)
            return
        self._log(f"Deleted {len(selected)} row(s): {rowid_list}.")
        self._on_load()

    def _on_delete_all_request(self, _btn: widgets.Button | None = None) -> None:
        """Ask for confirmation to delete every row matching the WHERE filter.

        Uses the current WHERE box (not just the loaded rows), so it can remove
        more rows than are visible when the result exceeds the limit.

        Args:
            _btn: The clicked button (unused).
        """
        table = self.table_dropdown.value
        where = self.where_text.value.strip() or None
        if where is None:
            self._log(
                "Set a WHERE filter first — 'Delete ALL matching WHERE' removes "
                "every row matching the filter. Use e.g. '1=1' to target all rows.",
                error=True,
            )
            return
        if not self.comment_text.value.strip():
            self._log("A change comment is required before deleting.", error=True)
            return
        try:
            with self._busy("Counting matching rows…"):
                count = self.con.count(table_name=table, where=where)
        except Exception as exc:
            self._log(f"Could not count matching rows: {exc}", error=True)
            return
        if count == 0:
            self._log("No rows match the current WHERE filter.")
            return

        self._pending_delete_where = where
        self.confirm_label.value = (
            f"<span style='color:#b00'>&#9888; Permanently delete "
            f"<b>{count:,}</b> row(s) matching <code>{where}</code>? "
            f"This cannot be undone.</span>"
        )
        self.confirm_yes_btn.description = f"Yes, delete {count:,}"
        self.confirm_box.layout.display = "flex"

    def _on_cancel_delete_all(self, _btn: widgets.Button | None = None) -> None:
        """Dismiss the delete-all confirmation without deleting anything."""
        self.confirm_box.layout.display = "none"
        self._pending_delete_where = None
        self._log("Delete cancelled.")

    def _on_confirm_delete_all(self, _btn: widgets.Button | None = None) -> None:
        """Delete every row matching the confirmed WHERE filter."""
        self.confirm_box.layout.display = "none"
        where = self._pending_delete_where
        self._pending_delete_where = None
        if where is None:
            return
        table = self.table_dropdown.value
        reason, comment = self.reason_dropdown.value, self.comment_text.value.strip()
        if not comment:
            self._log("A change comment is required before deleting.", error=True)
            return
        try:
            with self._busy("Deleting all matching rows…"):
                self.con.delete_row(
                    table_name=table,
                    where=where,
                    change_event_reason=reason,
                    change_comment=comment,
                )
        except Exception as exc:
            self._log(f"Delete failed: {exc}", error=True)
            return
        self._log(f"Deleted all rows matching WHERE: {where}.")
        self._on_load()

    def _on_show_log(self, _btn: widgets.Button | None = None) -> None:
        """Display the raw ``commit_extra_info`` for the latest edit of the table.

        Args:
            _btn: The clicked button (unused).
        """
        table = self.table_dropdown.value
        if not table:
            self._log("Select a table first.", error=True)
            return
        try:
            with self._busy("Loading edit log…"):
                edits = self.con.get_edits(table_name=table)
        except Exception as exc:
            self._log(f"Could not load edit log: {exc}", error=True)
            return

        self.log_out.clear_output()
        with self.log_out:
            if edits.empty or "commit_extra_info" not in edits.columns:
                print("No edits found for this table.")
                return
            columns = [
                c for c in ("snapshot_time", "commit_extra_info") if c in edits.columns
            ]
            latest = edits
            if "snapshot_time" in edits.columns:
                latest = edits.sort_values("snapshot_time", ascending=False)
            latest = latest.head(1)[columns].reset_index(drop=True)
            with pd.option_context(
                "display.max_colwidth", None, "display.max_rows", None
            ):
                display(latest)  # type: ignore[no-untyped-call]

    def _on_select_all(self, change: dict[str, Any]) -> None:
        """Toggle every row-selection checkbox to match the 'Select all' box.

        Args:
            change: The ipywidgets observe payload; ``change['new']`` holds the
                new checked state.
        """
        for chk in self._select_widgets.values():
            chk.value = bool(change["new"])

    # ============ Helpers ============

    def _set_busy(self, busy: bool, message: str = "") -> None:
        """Show/hide the spinner and enable/disable action buttons.

        Args:
            busy: ``True`` to enter the busy state, ``False`` to leave it.
            message: Text shown next to the spinner while busy.
        """
        self.busy_html.value = (
            f"{_SPINNER_IMG}{message}" if busy else ""
        )
        for btn in (
            self.load_btn,
            self.save_btn,
            self.delete_btn,
            self.delete_all_btn,
            self.refresh_tables_btn,
            self.show_log_btn,
        ):
            btn.disabled = busy

    @contextmanager
    def _busy(self, message: str) -> Iterator[None]:
        """Context manager that shows the spinner for the wrapped operation.

        Args:
            message: Text shown next to the spinner while busy.

        Yields:
            None. The busy state is cleared when the block exits.
        """
        self._set_busy(True, message)
        try:
            yield
        finally:
            self._set_busy(False)

    def _log(self, message: str, error: bool = False) -> None:
        """Print a status message into the output area.

        Args:
            message: The text to show.
            error: When ``True``, render the message as an error.
        """
        with self.status_out:
            prefix = "❌ " if error else "✅ "
            print(prefix + message)
        if error:
            logger.error(message)
        else:
            logger.info(message)
