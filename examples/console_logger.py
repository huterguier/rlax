import atexit
import math
import threading
import time
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
from lox.logdict import logdict
from lox.loggers.logger import Logger, LoggerState
from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import ProgressBar
from rich.table import Table
from rich.text import Text

_VALUE_ALLOWANCE = 16
_DETAIL_ALLOWANCE = 24
_GUTTER = 3
_MAX_COLUMNS = 4
_CELL_PADDING = 3
_PANEL_CHROME = 4

_SPARK = "▁▂▃▄▅▆▇█"
_CELLS = 8
_HISTORY = 1024


class Row(NamedTuple):
    """One rendered metric: its name, what is shown beside it, and its statistic.

    The label carries no section indent; that belongs to the rendering.
    """

    label: str
    detail: str
    summary: str


class Bar(NamedTuple):
    """One progress bar: its key, how far it has gone, and its timing."""

    label: str
    completed: float
    total: float
    eta: float | None


Section = tuple[str, list[Row]]
"""A section name -- empty for ungrouped keys -- and the rows beneath it."""


class History:
    """A metric's whole run held in a fixed number of points.

    Values arrive at very different rates: a metric logged inside the training
    scan contributes a whole epoch at once, an eval metric a single point. Keeping
    the newest N of each would leave their sparklines spanning different amounts
    of training -- unreadable, since the lines sit in one column and invite
    comparison. So the buffer decimates rather than truncates: when it fills, every
    two points are averaged into one and the stride doubles. Every stored point
    then covers the same number of raw values, and the line goes on spanning the
    run from its start, just more coarsely.

    That bounds the memory at ``capacity`` points however long the run is, and the
    work is bounded too: values are binned by ``reshape``/``mean`` rather than one
    at a time, since a long run hands over millions of them in a single call and a
    Python loop over each would stall the render.
    """

    def __init__(self, capacity: int = _HISTORY):
        assert capacity % 2 == 0, f"capacity ({capacity}) must be even to halve"
        self.capacity = capacity
        self.points = np.empty(0)
        self.stride = 1
        self._bucket = np.empty(0)

    def extend(self, values: jax.Array) -> None:
        values = np.concatenate([self._bucket, np.ravel(np.asarray(values))])
        whole = values.size // self.stride
        if whole:
            binned = values[: whole * self.stride].reshape(whole, self.stride)
            self.points = np.concatenate([self.points, binned.mean(-1)])
        self._bucket = values[whole * self.stride :]
        while self.points.size >= self.capacity:
            self._halve()

    def _halve(self) -> None:
        """Averages the points pairwise, dropping the oldest if they do not pair.

        The odd one out has to be the oldest: dropping the newest would lose the
        value the panel is about to show.
        """
        points = self.points[self.points.size % 2 :]
        self.points = points.reshape(-1, 2).mean(-1)
        self.stride *= 2

    def __len__(self) -> int:
        return self.points.size


@jax.tree_util.register_dataclass
@dataclass
class ConsoleLoggerState(LoggerState):
    key: jax.Array
    id: jax.Array


def _sections(keys: list[str]) -> dict[str, list[str]]:
    """Groups keys by the part before their first ``/``.

    Keys without a ``/`` share the leading unnamed section, so they stay at the top
    rather than being scattered between the named ones.
    """
    sections: dict[str, list[str]] = {"": []}
    for key in keys:
        section = key.split("/")[0] if "/" in key else ""
        sections.setdefault(section, []).append(key)
    if not sections[""]:
        del sections[""]
    return sections


def _flatten(data: dict, prefix: str = "") -> dict:
    """Flattens nested log dicts into ``"outer/inner"`` keys."""
    flat = {}
    for key, value in data.items():
        if isinstance(value, dict):
            flat.update(_flatten(value, f"{prefix}{key}/"))
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def _runs(n: int) -> str:
    return f"{n} run" + ("s" if n != 1 else "")


def _number(value: float) -> str:
    """Formats a statistic for display.

    Large magnitudes use thousands separators rather than scientific notation, so
    that step counters stay readable; everything else keeps four significant
    digits, which suits both ordinary metrics and very small ones.
    """
    return f"{value:,.0f}" if abs(value) >= 1e4 else f"{value:.4g}"


def _duration(seconds: float) -> str:
    """Formats a duration as ``1h02m``, ``3m12s`` or ``45s``, whichever fits.

    Progress speed is rough enough that sub-second precision would be noise, so
    the smallest unit shown is seconds and everything below that is dropped.
    """
    seconds = max(0, round(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


def _signed(value: float) -> str:
    """Formats a change, so that a metric's direction reads at a glance."""
    return f"{'+' if value >= 0 else '-'}{_number(abs(value))}"


def _events(value: jax.Array) -> jax.Array:
    """Reduces a logged value to one number per event.

    ``lox.log`` gives every value a leading event axis, so whatever sits behind it
    -- nothing for a scalar, a vector, an image -- is what a single number has to
    stand in for, and a mean is the only thing that serves all of them. Doing this
    for every key, rather than only for the ones that need it, is what lets one
    rule describe the whole panel: the value shown is the newest event, and the
    sparkline is that same number over time.
    """
    value = jnp.abs(value) if jnp.iscomplexobj(value) else value
    value = jnp.atleast_1d(value)
    return jnp.mean(value.reshape(value.shape[0], -1), axis=-1)


def _event_shape(values: list[jax.Array]) -> str:
    """Names what a row's mean is taken over, or "" when it is taken over nothing.

    Only the shape behind the event axis matters here: a scalar per event needs no
    annotation, since its value is not standing in for anything. Trailing size-1
    axes are dropped, which is what ``vmap`` leaves behind when its lanes flatten
    into the event axis.
    """
    shapes = set()
    for value in values:
        shape = jnp.atleast_1d(value).shape[1:]
        while shape and shape[-1] == 1:
            shape = shape[:-1]
        shapes.add(shape)
    if shapes == {()}:
        return ""
    return "mixed shapes" if len(shapes) > 1 else str(shapes.pop())


def _sparkline(series: jax.Array, cells: int = _CELLS) -> str:
    """Draws a series as a line of block characters, oldest to newest.

    Longer series are binned by mean into ``cells`` buckets of as equal a size as
    divides them. Every value lands in a bucket, so the line spans exactly the
    window the change beside it is measured over -- dropping the remainder instead
    would quietly shorten the line while the number kept the full span. The scale
    is the series' own min-max, so the line shows shape and not magnitude; the
    numbers beside it carry magnitude.

    Non-finite values have no position on that scale, so a series containing one
    is left undrawn and its NaN surfaces in the value instead.
    """
    series = jnp.ravel(series)
    if series.size == 0 or not bool(jnp.all(jnp.isfinite(series))):
        return ""
    if series.size > cells:
        edges = jnp.linspace(0, series.size, cells + 1).round().astype(int)
        binned = jnp.stack(
            [
                jnp.mean(series[int(start) : int(stop)])
                for start, stop in zip(edges[:-1], edges[1:], strict=True)
            ]
        )
    else:
        binned = series
    low, high = float(jnp.min(binned)), float(jnp.max(binned))
    if high == low:
        return _SPARK[len(_SPARK) // 2] * binned.size
    scaled = (binned - low) / (high - low) * (len(_SPARK) - 1)
    return "".join(_SPARK[round(float(v))] for v in scaled)


class ConsoleLogger(Logger[ConsoleLoggerState]):
    """
    A logger that renders logs as a live-updating panel on stdout.

    Each call to :meth:`init` registers a new run. Keys are grouped into sections by
    the part before their first ``/``, and the sections are laid out in several
    columns when the terminal is wide enough.

    One rule describes every row: the bold value is the metric's **newest** value,
    and the dim cell beside it is a sparkline of that same number over time.

    "Newest" is well defined because the leading axis is chronological: ``lox.log``
    gives each value an event axis and ``spool`` flattens a scan into it in order,
    nesting outer-major. ``vmap`` interleaves its lanes within a step but leaves
    time as the major axis, so the tail is still the newest step. The one thing
    that does break the ordering is two ``lox.log`` call sites writing the same
    key, since those are concatenated as blocks.

    A value that is not a scalar -- a vector, an image -- is reduced to its mean,
    which is the only number that serves every shape. That is a real loss of
    information, so the shape it was taken over is named beside the line, where a
    scalar's row shows nothing because its value stands in for nothing. The
    sparkline goes on meaning what it means everywhere else: the history of the
    number shown.

    With several runs the summary is taken across runs: each run contributes its
    own latest value, and the ``±`` returns to report how much the runs disagree. Getting
    that number requires one :meth:`init` per run -- either in a Python loop or
    under ``vmap`` -- since runs sharing a single state are indistinguishable once
    logged.

    Each call replaces the values of the keys it logs, so the value describes the
    most recent call. The sparkline is the exception: values are appended to a
    per-key history, so the line spans the session rather than the last call. A key
    logged one scalar at a time -- an eval metric, once an epoch -- would otherwise
    have no line at all.
    """

    console: Console
    logss: dict[str, logdict]
    history: dict[str, dict[str, History]]
    live: Live | None

    def __init__(self, progress: dict[str, float] | None = None):
        """
        Args:
            progress: Maps a logged key to the total it counts towards, rendering
                it as a progress bar below the metrics instead of as a row. Bars
                only advance during a run under :meth:`tap`; under :meth:`spool`
                the logs arrive in one callback once the function has returned.
        """
        self.console = Console()
        self.logss = {}
        self.live = None
        self.progress = dict(progress or {})
        self.history = {}
        self._lock = threading.Lock()
        self._bar_starts: dict[str, float] = {}

    def init(self, key: jax.Array) -> ConsoleLoggerState:
        def callback(key):
            with self._lock:
                run_id = jnp.int32(len(self.logss.keys()))
                self.logss[str(run_id)] = logdict({})
                self._start()
            return run_id

        run_id = jax.experimental.io_callback(
            callback,
            jax.ShapeDtypeStruct((), jnp.int32),
            key=key,
        )

        return ConsoleLoggerState(key=key, id=run_id)

    def _start(self) -> None:
        """Starts the live display, reusing it across runs."""
        if self.live is None:
            self.live = Live(Text(""), console=self.console, refresh_per_second=4)
            self.live.start()
            atexit.register(self.close)

    def close(self) -> None:
        """Stops the live display and restores the terminal."""
        if self.live is not None:
            self.live.stop()
            self.live = None

    def callback(self, logger_state: ConsoleLoggerState, logs: logdict):
        """Records the logs and redraws the panel.

        Rendering is serialised: ``tap`` routes logs through an *unordered*
        ``jax.debug.callback``, so two lanes can arrive at once and one could
        register a run while the other iterates them to build the panel.
        """
        with self._lock:
            self._render(logger_state, logs)

    def _values(self, key: str) -> list[jax.Array]:
        """Each run's latest array for a key, skipping runs that lack it.

        Complex values are reduced to their magnitude: they have no mean the
        terminal can show, and leaving them would raise from ``float()`` later.
        """
        return [
            jnp.abs(run[key]) if jnp.iscomplexobj(run[key]) else run[key]
            for run in self.logss.values()
            if key in run
        ]

    def _record(self, run_id: str, logs: dict) -> None:
        """Appends a run's newest values to the history its sparkline is drawn from.

        The values of one call are appended whole rather than reduced to a point:
        the leading axis is already in chronological order, so a scan of 48 loops
        contributes 48 points of real trend that a per-call reduction would throw
        away. A key logged one scalar at a time contributes one, and both end up
        on the same footing.
        """
        history = self.history.setdefault(run_id, {})
        for key, value in logs.items():
            if key in self.progress:
                continue
            series = history.setdefault(key, History())
            series.extend(_events(value))

    def _series(self, key: str) -> jax.Array | None:
        """The key's history averaged across runs, or ``None`` if it has none.

        Runs are averaged over their common tail, since they advance independently
        and one may have recorded more than another by the time the panel is drawn.
        """
        histories = [
            h for run in self.logss if (h := self.history.get(run, {}).get(key))
        ]
        if not histories:
            return None
        length = min(len(h) for h in histories)
        return jnp.mean(
            jnp.stack([jnp.asarray(h.points[len(h) - length :]) for h in histories]),
            axis=0,
        )

    def _row(self, key: str, label: str, show_runs: bool) -> Row:
        """Builds one metric's row: its newest value, and where it has been.

        The value stands alone. A deviation taken within the call has no business
        being written ``value ± std``, since ``±`` claims an interval around a
        centre and the newest value is not the centre of anything; writing it
        honestly needs a label, and a labelled second number costs more attention
        than it returns when the sparkline beside it already shows whether the
        metric is jumpy. Across runs there is a genuine centre -- the runs' values
        do sit around their mean -- so there the ``±`` returns and says what it
        always said.
        """
        values = self._values(key)
        latest = jnp.stack([_events(value)[-1] for value in values])

        summary = _number(float(jnp.mean(latest)))
        if latest.size > 1:
            summary += f" ± {_number(float(jnp.std(latest)))}"

        parts = []
        series = self._series(key)
        if series is not None and (spark := _sparkline(series)):
            parts.append(spark)
        if shape := _event_shape(values):
            parts.append(shape)
        if show_runs:
            parts.append(_runs(len(values)))

        return Row(label=label, detail=" ".join(parts), summary=summary)

    def _layout(self) -> tuple[list[Section], str | None]:
        """The sections to draw, and the subtitle summarising them.

        The run count is normally the same on every row, so it is stated once in
        the subtitle rather than repeated down a column. When the counts disagree
        -- while runs are still reporting, or for a key only some of them log --
        there is no single count to state, so the subtitle is dropped and each row
        carries its own instead.
        """
        keys = {k for run in self.logss.values() for k in run} - set(self.progress)
        counts = {len(self._values(key)) for key in keys}
        show_runs = len(counts) > 1

        sections: list[Section] = []
        for section, section_keys in _sections(sorted(keys)).items():
            rows = [
                self._row(
                    key,
                    label=key.removeprefix(f"{section}/") if section else key,
                    show_runs=show_runs,
                )
                for key in section_keys
            ]
            sections.append((section, rows))

        subtitle = _runs(next(iter(counts))) if len(counts) == 1 else None
        return sections, subtitle

    def _progress_bars(self) -> list[Bar]:
        """How far each configured progress key has advanced, and its ETA.

        Progress is the maximum logged value rather than the last: the leading
        axis has no reliable order, and a counter only ever grows. Under ``vmap``
        the lanes advance in lockstep, so their mean is simply the shared position.

        The ETA is extrapolated from the average rate since the bar's first
        update -- wall-clock time elapsed divided into progress made -- rather
        than from a fixed step cost, since steps commonly vary in cost (e.g. an
        eval pass every N steps) and a moving-window rate would just add jitter
        for little benefit here.
        """
        bars = []
        for key, total in self.progress.items():
            values = self._values(key)
            if not values:
                continue
            completed = float(jnp.mean(jnp.stack([jnp.max(value) for value in values])))
            start = self._bar_starts.setdefault(key, time.monotonic())
            elapsed = time.monotonic() - start
            eta = (
                (total - completed) * elapsed / completed
                if completed > 0 and elapsed > 0
                else None
            )
            bars.append(Bar(label=key, completed=completed, total=total, eta=eta))
        return bars

    def _render(self, logger_state: ConsoleLoggerState, logs: logdict):
        """Records the logs, then redraws the panel from every run so far."""
        run_id = str(logger_state.id)
        self.logss.setdefault(run_id, logdict({}))
        self.logss[run_id] |= logdict(_flatten(logs))
        self._record(run_id, _flatten(logs))
        self._start()

        sections, subtitle = self._layout()
        grid = self._grid(sections)
        bars = self._bars()
        self.live.update(
            Panel(
                Group(grid, Text(""), bars) if bars.row_count else grid,
                box=box.ROUNDED,
                border_style="dim",
                subtitle=subtitle,
                subtitle_align="right",
            )
        )

    def _new_table(self) -> Table:
        """Builds an empty table of name, detail, value and a trailing spacer.

        The detail sits immediately behind the name rather than in a far column,
        but keeps a cell of its own so values stay aligned across rows. The spacer
        takes the slack, so a value stays beside its name instead of drifting to
        the far edge.
        """
        table = Table(box=None, expand=True, show_header=False, pad_edge=False)
        table.add_column(no_wrap=True)
        table.add_column(no_wrap=True, style="dim")
        table.add_column(justify="right", no_wrap=True, style="bold")
        table.add_column(ratio=1)
        return table

    def _columns(self, sections: list[Section]) -> int:
        """Chooses how many columns the sections are laid out in.

        The estimate deliberately uses only widths that do not move as values do
        -- key names, shapes, and a fixed allowance for the number -- so that a
        metric growing from ``5`` to ``1,234,567`` cannot make the whole layout
        flip between column counts on successive refreshes. ``_VALUE_ALLOWANCE``
        is that allowance, chosen as the widest realistic ``mean ± std``, and the
        two constants beside it cover the cell padding and the panel border.
        """
        rows = [row for _, section_rows in sections for row in section_rows]
        if not rows:
            return 1
        indents = {row: 2 if section else 0 for section, rs in sections for row in rs}
        width = (
            max(len(row.label) + indents[row] for row in rows)
            + max(_DETAIL_ALLOWANCE, max(len(row.detail) for row in rows))
            + _VALUE_ALLOWANCE
            + _CELL_PADDING
        )
        available = self.console.width - _PANEL_CHROME
        return max(1, min(_MAX_COLUMNS, len(sections), available // (width + _GUTTER)))

    def _pack(self, sections: list[Section], n: int) -> list[list[Section]]:
        """Splits the sections into ``n`` columns, in order.

        Filling column by column keeps sections where they would be in a single
        column, at the cost of columns that can differ in height.
        """
        heights = [len(rows) + (2 if section else 0) for section, rows in sections]
        target = max(1, math.ceil(sum(heights) / n))
        columns: list[list[Section]] = []
        current: list[Section] = []
        used = 0
        for section, height in zip(sections, heights, strict=True):
            if current and used + height > target and len(columns) < n - 1:
                columns.append(current)
                current, used = [], 0
            current.append(section)
            used += height
        columns.append(current)
        return columns

    def _grid(self, sections: list[Section]) -> Table:
        """Lays the sections out side by side."""
        n = self._columns(sections)
        grid = Table.grid(expand=True, padding=(0, _GUTTER))
        for _ in range(n):
            grid.add_column(ratio=1)

        tables = []
        for column in self._pack(sections, n):
            table = self._new_table()
            for i, (section, rows) in enumerate(column):
                if section:
                    if i:
                        table.add_row("", "", "", "")
                    table.add_row(f"[bold cyan]{section}[/bold cyan]", "", "", "")
                for row in rows:
                    label = f"  {row.label}" if section else row.label
                    table.add_row(label, row.detail, row.summary, "")
            tables.append(table)
        grid.add_row(*tables, *[""] * (n - len(tables)))
        return grid

    def _bars(self) -> Table:
        """Draws the progress bars.

        They live in their own table so the metric columns are not stretched to
        accommodate a full-width bar.
        """
        table = Table(box=None, expand=True, show_header=False, pad_edge=False)
        table.add_column(no_wrap=True)
        table.add_column(ratio=1)
        table.add_column(justify="right", style="dim", no_wrap=True)
        for bar in self._progress_bars():
            count = f"{bar.completed:,.0f}/{bar.total:,.0f}"
            if bar.completed >= bar.total:
                stat = count
            elif bar.eta is not None:
                stat = f"{count} · eta {_duration(bar.eta)}"
            else:
                stat = f"{count} · eta --"
            table.add_row(
                bar.label,
                ProgressBar(
                    total=bar.total,
                    completed=min(bar.completed, bar.total),
                    complete_style="cyan",
                    finished_style="green",
                ),
                stat,
            )
        return table


if __name__ == "__main__":
    """Draws a panel from a fabricated run, so layouts can be tried without training.

    The shapes mirror what ``Trainer`` sends once an epoch: metrics logged inside
    the training scan arrive as a whole epoch's worth of values, while an eval
    metric arrives as a single scalar.
    """
    import numpy as np

    num_epochs, num_loops, num_minibatches = 10, 48, 768
    rng = np.random.default_rng(0)
    logger = ConsoleLogger(progress={"step": num_epochs * num_loops})
    state = ConsoleLoggerState(key=jax.random.key(0), id=jnp.int32(0))

    for epoch in range(num_epochs):
        progress = np.linspace(epoch, epoch + 1, num_loops) / num_epochs
        fine = np.linspace(epoch, epoch + 1, num_minibatches) / num_epochs
        logger.callback(
            state,
            logdict(
                {
                    "step": jnp.asarray((epoch + 1) * num_loops)[None],
                    "return": jnp.asarray(
                        500 * progress**2 + rng.normal(0, 15, num_loops)
                    ),
                    "actor/loss": jnp.asarray(
                        -0.04 * (1 - fine) + rng.normal(0, 0.01, num_minibatches)
                    ),
                    "actor/entropy": jnp.asarray(
                        0.69 * np.exp(-2 * fine) + rng.normal(0, 0.02, num_minibatches)
                    ),
                    "critic/loss": jnp.asarray(
                        80 * np.exp(-3 * fine) + rng.normal(0, 3, num_minibatches)
                    ),
                    "critic/value": jnp.asarray(
                        480 * fine + rng.normal(0, 20, num_minibatches)
                    ),
                    # not a scalar per event: keeps the shape and mean +- std
                    "actor/logits": jnp.asarray(rng.normal(0, 1, (num_loops, 2))),
                    "eval/return": jnp.asarray(500 * ((epoch + 1) / num_epochs) ** 1.5)[
                        None
                    ],
                }
            ),
        )
        time.sleep(0.4)

    logger.close()
