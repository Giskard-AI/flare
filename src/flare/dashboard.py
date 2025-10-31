from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from rich import print
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    Table,
    TextColumn,
    TimeRemainingColumn,
)

from flare.stats import STATS

_progress = {}
_task_mapping = {"models": {}, "scorers": {}}
_layout = Layout()


def make_layout():
    # Divide the "screen" in to three parts
    _layout.split_column(
        Layout(name="header", size=4),
        Layout(
            name="main",
            ratio=1,
        ),
    )
    _layout["main"].split_row(Layout(name="stats"), Layout(name="logs", ratio=2))


def setup_stats(layout) -> tuple[dict[str, Any], dict[str, Any]]:
    generation_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(elapsed_when_finished=True),
    )
    for name in STATS["models"].keys():
        # Truncate model names to 30 characters for display
        display_name =  "..." + name[-50:] if len(name) > 50 else name
        _task_mapping["models"][name] = generation_progress.add_task(
            display_name, total=STATS["nb_samples"]
        )

    overall_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(elapsed_when_finished=True),
    )

    _task_mapping["all"] = overall_progress.add_task("All", total=STATS["total"])

    progress_table = Table.grid(expand=True)
    progress_table.add_row(
        Panel(
            overall_progress,
            title="Overall Progress",
            border_style="green",
            padding=(1, 1),
        ),
    )
    progress_table.add_row(
        Panel(
            generation_progress,
            title="[b]Generations",
            border_style="red",
            padding=(1, 1),
        ),
    )

    scorer_progress = Progress(
        "{task.description}",
        SpinnerColumn(),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(elapsed_when_finished=True),
    )
    for name in STATS["scorers"].keys():
        # Truncate scorer names to 30 characters for display
        display_name = name[:30] + "..." if len(name) > 30 else name
        _task_mapping["scorers"][name] = scorer_progress.add_task(
            display_name, total=STATS["samples_per_model"][name] * len(STATS["models"].keys())
        )

    progress_table.add_row(
        Panel(scorer_progress, title="[b]Scorers", border_style="blue", padding=(1, 2)),
    )

    layout["main"]["stats"].update(progress_table)
    _progress["all"] = overall_progress
    _progress["generation"] = generation_progress
    _progress["scorer"] = scorer_progress

    layout["header"].update(
        Panel(
            f"Run : {STATS["run_name"]}, started at {str(STATS["start"])}, running since {datetime.now() - STATS["start"]}",
            title="Summary",
            border_style="blue",
        )
    )


def tail(filename, n=50):
    with open(filename, encoding="utf-8") as f:
        return deque(f, n)


def update_display(log_file: Path):

    for name, stat in STATS["models"].items():
        _progress["generation"].update(
            _task_mapping["models"][name], completed=sum(stat.values())
        )
    for name, stat in STATS["scorers"].items():
        _progress["scorer"].update(
            _task_mapping["scorers"][name], completed=sum(stat.values())
        )
    _progress["all"].update(_task_mapping["all"], completed=STATS["current"])
    _layout["main"]["logs"].update(
        Panel(
            "".join(tail(str(log_file), 50)),
            title="[b]Log",
            border_style="green",
            padding=(1, 1),
        )
    )
    _layout["header"].update(
        Panel(
            f"""Run : {STATS["run_name"]}, started at {str(STATS["start"])}, running since {datetime.now() - STATS["start"]}""",
            title="Summary",
            border_style="blue",
        )
    )


def live_dashboard():
    make_layout()
    setup_stats(_layout)
    return Live(_layout, refresh_per_second=10, screen=True)


def print_dashboard():
    print(_layout)
