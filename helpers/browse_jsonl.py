"""Interactive TUI to browse a JSONL file sample by sample.

Each sample's top-level fields are rendered as collapsible sections so
you can step through the file and inspect the contents.
"""

import argparse
from pathlib import Path
from typing import Any

import orjson
from rich.markup import escape
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection


def load_jsonl(path: Path) -> list[dict]:
    """Parse a JSONL file at ``path`` into a list of dicts, skipping blank lines."""
    items: list[dict] = []
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(orjson.loads(line))
    return items


def render_scalar(value: Any) -> str:
    """Render a JSON scalar as a Rich markup string with type-appropriate styling."""
    if value is None:
        return "[dim italic]null[/]"
    if isinstance(value, bool):
        return f"[magenta]{value}[/]"
    if isinstance(value, (int, float)):
        return f"[cyan]{value}[/]"
    if isinstance(value, str):
        return escape(value) if value else "[dim italic](empty string)[/]"
    return escape(repr(value))


def render_list(value: list) -> str:
    """Render a list of JSON values as Rich markup, inlining scalars and bulleting nested structures."""
    if not value:
        return "[dim italic](empty list)[/]"
    if all(not isinstance(v, (dict, list)) for v in value):
        return ", ".join(render_scalar(v) for v in value)
    return "\n".join(f"• {render_scalar(v) if not isinstance(v, (dict, list)) else orjson.dumps(v).decode()}" for v in value)


def collect_keys(items: list[dict]) -> list[str]:
    """Return all top-level keys across items, preserving first-seen order."""
    seen: dict[str, None] = {}
    for item in items:
        for key in item.keys():
            seen.setdefault(key, None)
    return list(seen.keys())


class ColumnSelectScreen(ModalScreen[list[str] | None]):
    """Modal letting the user pick which top-level fields to display."""

    DEFAULT_CSS = """
    ColumnSelectScreen {
        align: center middle;
    }
    ColumnSelectScreen > Vertical {
        width: 70;
        height: auto;
        max-height: 90%;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    ColumnSelectScreen SelectionList {
        height: auto;
        max-height: 30;
        margin: 1 0;
    }
    ColumnSelectScreen #buttons {
        height: auto;
        align-horizontal: right;
    }
    ColumnSelectScreen Button {
        margin-left: 1;
    }
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
        Binding("a", "all", "All"),
        Binding("n", "none", "None"),
        Binding("enter", "confirm", "Confirm", priority=True),
    ]

    def __init__(self, keys: list[str], selected: list[str] | None = None) -> None:
        """Initialise the modal with all available ``keys`` and the currently ``selected`` subset."""
        super().__init__()
        self.keys = keys
        self._initial = set(keys if selected is None else selected)

    def compose(self) -> ComposeResult:
        """Build the modal layout: header label, selection list, and action buttons."""
        from textual.containers import Horizontal, Vertical

        with Vertical():
            yield Label("Select columns to display (space to toggle, a=all, n=none, enter=confirm):")
            yield SelectionList[str](
                *(Selection(k, k, initial_state=k in self._initial) for k in self.keys),
                id="columns",
            )
            with Horizontal(id="buttons"):
                yield Button("All", id="all", variant="default")
                yield Button("None", id="none", variant="default")
                yield Button("Confirm", id="confirm", variant="primary")

    def on_mount(self) -> None:
        """Focus the selection list when the modal appears."""
        self.query_one(SelectionList).focus()

    def action_all(self) -> None:
        """Select every column in the list."""
        self.query_one(SelectionList).select_all()

    def action_none(self) -> None:
        """Deselect every column in the list."""
        self.query_one(SelectionList).deselect_all()

    def action_confirm(self) -> None:
        """Dismiss the modal, returning the currently selected columns."""
        self.dismiss(list(self.query_one(SelectionList).selected))

    def action_cancel(self) -> None:
        """Dismiss the modal without applying any change."""
        self.dismiss(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Route button clicks to the matching action handler."""
        if event.button.id == "all":
            self.action_all()
        elif event.button.id == "none":
            self.action_none()
        elif event.button.id == "confirm":
            self.action_confirm()


class JumpScreen(ModalScreen[int | None]):
    """Modal dialog asking for a 1-based sample index."""

    DEFAULT_CSS = """
    JumpScreen {
        align: center middle;
    }
    JumpScreen > Vertical {
        width: 50;
        height: auto;
        padding: 1 2;
        background: $panel;
        border: thick $accent;
    }
    """

    BINDINGS = [Binding("escape", "cancel", "Cancel")]

    def __init__(self, total: int, current: int) -> None:
        """Configure the dialog for a dataset of ``total`` samples, pre-filled with the ``current`` index."""
        super().__init__()
        self.total = total
        self.current = current

    def compose(self) -> ComposeResult:
        """Build the modal layout: a prompt label and the numeric input."""
        from textual.containers import Vertical

        with Vertical():
            yield Label(f"Jump to sample (1–{self.total}):")
            yield Input(value=str(self.current + 1), id="jump-input")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Parse the submitted value and dismiss with a zero-based index, or ``None`` if invalid."""
        try:
            idx = int(event.value) - 1
        except ValueError:
            self.dismiss(None)
            return
        if 0 <= idx < self.total:
            self.dismiss(idx)
        else:
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Dismiss the modal without jumping."""
        self.dismiss(None)


class SampleView(VerticalScroll):
    """Scrollable list of field sections for a single sample."""


class BrowserApp(App):
    CSS = """
    SampleView {
        padding: 1 2;
    }
    .field-value {
        padding: 0 1 1 2;
    }
    Collapsible {
        margin-bottom: 1;
    }
    """

    BINDINGS = [
        Binding("n,right,space", "next", "Next"),
        Binding("p,left", "prev", "Prev"),
        Binding("home", "first", "First"),
        Binding("end", "last", "Last"),
        Binding("g", "jump", "Go to"),
        Binding("c", "toggle_all", "Collapse/expand all"),
        Binding("s", "select_columns", "Select columns"),
        Binding("ctrl+r", "refresh", "Reload file"),
        Binding("q", "quit", "Quit"),
    ]

    index: reactive[int] = reactive(0, init=False)

    def __init__(self, path: Path, items: list[dict]) -> None:
        """Initialise the app with the source ``path`` and the parsed ``items`` to browse."""
        super().__init__()
        self.path = path
        self.items = items
        self._all_collapsed = False
        self._all_keys = collect_keys(items)
        self._selected_keys: list[str] = list(self._all_keys)

    def compose(self) -> ComposeResult:
        """Build the app layout: header, scrollable sample view, footer."""
        yield Header(show_clock=False)
        yield SampleView(id="content")
        yield Footer()

    def on_mount(self) -> None:
        """Render the first sample and prompt for column selection on startup."""
        self.title = "JSONL Browser"
        self._render_sample()
        if self._all_keys:
            self.call_after_refresh(self.action_select_columns)

    def watch_index(self, _old: int, _new: int) -> None:
        """Re-render the sample view whenever the reactive ``index`` changes."""
        self._render_sample()

    def _render_sample(self) -> None:
        """Replace the sample view's contents with collapsible sections for the current sample."""
        view = self.query_one(SampleView)
        view.remove_children()
        if not self.items:
            view.mount(Static("[dim italic]File is empty.[/]"))
            self.sub_title = f"{self.path.name} — 0 samples"
            return

        item = self.items[self.index]
        self.sub_title = f"{self.path.name} — sample {self.index + 1} / {len(self.items)}"

        selected = set(self._selected_keys)
        shown = [(k, v) for k, v in item.items() if k in selected]
        if not shown:
            view.mount(Static("[dim italic]No columns selected. Press 's' to choose columns.[/]"))
            return

        for key, value in shown:
            view.mount(self._build_field(key, value))
        view.scroll_home(animate=False)

    def _build_field(self, key: str, value: Any) -> Collapsible:
        """Build a top-level :class:`Collapsible` section for one ``(key, value)`` field of a sample."""
        title = f"[bold cyan]{escape(key)}[/]"

        if isinstance(value, dict):
            if not value:
                children = [Static("[dim italic](empty object)[/]", classes="field-value")]
            else:
                children = [self._build_subfield(k, v) for k, v in value.items()]
        elif isinstance(value, list):
            children = [Static(render_list(value), classes="field-value")]
        else:
            children = [Static(render_scalar(value), classes="field-value")]

        return Collapsible(*children, title=title, collapsed=self._all_collapsed)

    def _build_subfield(self, key: str, value: Any) -> Static:
        """Render a nested ``(key, value)`` pair inside a top-level collapsible as a single :class:`Static`."""
        label = f"[bold yellow]{escape(key)}[/]"
        if isinstance(value, list):
            body = render_list(value)
        elif isinstance(value, dict):
            body = escape(orjson.dumps(value).decode())
        else:
            body = render_scalar(value)
        return Static(f"{label}\n{body}", classes="field-value")

    def action_next(self) -> None:
        """Advance to the next sample, clamped at the last item."""
        if self.index < len(self.items) - 1:
            self.index += 1

    def action_prev(self) -> None:
        """Step back to the previous sample, clamped at the first item."""
        if self.index > 0:
            self.index -= 1

    def action_first(self) -> None:
        """Jump to the first sample in the file."""
        self.index = 0

    def action_last(self) -> None:
        """Jump to the last sample in the file."""
        if self.items:
            self.index = len(self.items) - 1

    def action_jump(self) -> None:
        """Open the :class:`JumpScreen` modal to jump to a 1-based sample index."""
        if not self.items:
            return

        def _set(result: int | None) -> None:
            """Apply the modal's chosen index to ``self.index`` if one was returned."""
            if result is not None:
                self.index = result

        self.push_screen(JumpScreen(len(self.items), self.index), _set)

    def action_refresh(self) -> None:
        """Reload the JSONL file from disk, preserving column selection where possible."""
        try:
            self.items = load_jsonl(self.path)
        except (OSError, orjson.JSONDecodeError) as exc:
            self.notify(f"Reload failed: {exc}", severity="error")
            return

        self._all_keys = collect_keys(self.items)
        current = set(self._selected_keys)
        self._selected_keys = [k for k in self._all_keys if k in current] or list(self._all_keys)

        if self.items:
            self.index = min(self.index, len(self.items) - 1)
        else:
            self.index = 0
        self._render_sample()
        self.notify(f"Reloaded {len(self.items)} samples")

    def action_select_columns(self) -> None:
        """Open the :class:`ColumnSelectScreen` modal to pick which top-level fields are shown."""
        if not self._all_keys:
            return

        def _set(result: list[str] | None) -> None:
            """Apply the modal's chosen columns and re-render the current sample."""
            if result is None:
                return
            chosen = set(result)
            self._selected_keys = [k for k in self._all_keys if k in chosen]
            self._render_sample()

        self.push_screen(ColumnSelectScreen(self._all_keys, self._selected_keys), _set)

    def action_toggle_all(self) -> None:
        """Collapse or expand every :class:`Collapsible` section in the current view."""
        self._all_collapsed = not self._all_collapsed
        for c in self.query(Collapsible):
            c.collapsed = self._all_collapsed


def parse_args() -> argparse.Namespace:
    """Parse the CLI arguments — a single positional ``input`` JSONL path."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("input", type=Path, help="Path to the JSONL file to browse")
    return parser.parse_args()


def main() -> None:
    """Entry point — parse args, load the JSONL file, and run the TUI app."""
    args = parse_args()
    if not args.input.is_file():
        raise SystemExit(f"File not found: {args.input}")
    items = load_jsonl(args.input)
    BrowserApp(args.input, items).run()


if __name__ == "__main__":
    main()
