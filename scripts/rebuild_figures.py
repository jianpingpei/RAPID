"""Render the paper's measured figures (Fig 5, 8, 9, 10, 11) from data/curves.

The input curves are frozen derivatives.  ``dynamic_load`` is the per-cycle
dynamic-load trace from the archived dynamic experiment, and
``controller_response`` is the C4 controller-response trace used for Figure 9.
The sweep files already store the reported five-dataset means.  Every figure is
written as vector PDF plus a 300-dpi PNG.

Usage:
    python scripts/rebuild_figures.py [--output-dir outputs/figures]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

import paper_style as style  # noqa: E402
from methods import METHODS, NAMES  # noqa: E402

DATA = ROOT / "data" / "curves"
DEFAULT_OUTPUT = ROOT / "outputs" / "figures"

# Compact marks used by the network and dynamic-load panels.
METHOD_STYLES = {
    "fixed": ("#0072B2", "s"),
    "cost": ("#56B4E9", "v"),
    "specedge": ("#009E73", "^"),
    "opt_tree": ("#E69F00", "D"),
    "svip": ("#CC79A7", "s"),
    "edgellm": ("#666666", "X"),
    "pearl": ("#999999", "+"),
    "sequential": ("#333333", "x"),
    "rapid": (style.ORANGE, "o"),
}

# Panel order of the dynamic-load figure: RAPID first, as in the manuscript.
DYNAMIC_ORDER = [
    "rapid",
    "fixed",
    "cost",
    "specedge",
    "opt_tree",
    "svip",
    "edgellm",
    "pearl",
    "sequential",
]
DYNAMIC_NAMES = {
    "rapid": "RAPID",
    "fixed": "Fixed Budget",
    "cost": "Cost-Model",
    "specedge": "SpecEdge",
    "opt_tree": "OPT-Tree",
    "svip": "SVIP",
    "edgellm": "EdgeLLM",
    "pearl": "PEARL",
    "sequential": "Sequential",
}
NAME_TO_SLUG = {NAMES[slug]: slug for slug in METHODS}


def read(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save(figure: Figure, stem: str, output: Path) -> None:
    style.save(figure, stem, output)
    plt.close(figure)


def figure5_calibration(output: Path) -> None:
    rows = read("figure5_calibration.csv")
    if len(rows) != 4:
        raise AssertionError(f"figure5_calibration: expected 4 points, got {len(rows)}")
    workers = [int(row["worker_count"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(3.4, 1.65), layout="constrained")
    panels = [
        ("gpu_util_mean_pct", "GPU utilization (%)", style.BLUE),
        ("tps", "Throughput (tok/s)", style.ORANGE),
    ]
    for axis, (key, label, color) in zip(axes, panels, strict=True):
        values = [float(row[key]) for row in rows]
        axis.plot(workers, values, marker="o", color=color)
        axis.set(xlabel="Co-located workers", ylabel=label, xticks=workers)
        axis.set_ylim(bottom=0)
        style.clean_axis(axis)
    save(figure, "fig5_calibration", output)
    print("  fig5_calibration: 4 workers x (GPU utilization, throughput)")


def _averaged_series(
    rows: list[dict[str, str]], key: str, index: str, value: str
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Average a time series over runs, indexed by method."""
    grouped: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[row[key]][int(row[index])].append(float(row[value]))
    series = {}
    for name, by_index in grouped.items():
        indices = sorted(by_index)
        if indices != list(range(indices[-1] + 1)):
            raise AssertionError(f"{key}={name}: cycles are not contiguous")
        series[name] = (
            np.array(indices, dtype=float),
            np.array([float(np.mean(by_index[i])) for i in indices]),
        )
    return series


def figure8_dynamic_load(output: Path) -> None:
    rows = read("dynamic_load_timeseries.csv")
    if len(rows) != 909:
        raise AssertionError(f"dynamic_load_timeseries: expected 909 rows, got {len(rows)}")
    if {row["method"] for row in rows} != set(METHODS):
        raise AssertionError("dynamic_load_timeseries: method coverage differs")
    rate = _averaged_series(rows, "method", "cycle", "rate_tok_s")
    steps = _averaged_series(rows, "method", "cycle", "steps_per_cycle")
    series = {}
    for method in METHODS:
        rate_x, rate_y = rate[method]
        steps_x, steps_y = steps[method]
        if not np.array_equal(rate_x, steps_x):
            raise AssertionError(f"{method}: rate/steps cycle grids differ")
        keep = rate_x <= 100
        series[method] = (
            rate_x[keep],
            style.gaussian(rate_y[keep], 5),
            style.rolling_mean(steps_y[keep], 11),
        )

    figure = plt.figure(figsize=(7.4, 5.25))
    outer = figure.add_gridspec(
        3, 3, left=0.075, right=0.99, bottom=0.075, top=0.915, hspace=0.29, wspace=0.15
    )
    for index, method in enumerate(DYNAMIC_ORDER):
        row, column = divmod(index, 3)
        cell = outer[row, column].subgridspec(2, 1, height_ratios=[1.8, 1], hspace=0.13)
        top = figure.add_subplot(cell[0])
        bottom = figure.add_subplot(cell[1], sharex=top)
        rate_x, rate_y, steps_y = series[method]
        rapid_x, rapid_rate, rapid_steps = series["rapid"]
        for panel, axis in enumerate((top, bottom)):
            style.clean_axis(axis)
            axis.grid(axis="y", color=style.GRID, linewidth=0.4, alpha=0.6)
            axis.axvspan(30, 70, color="#EBECEE", zorder=0)
            axis.axvline(30, color="#C5C8CC", linewidth=0.5, linestyle=":")
            axis.axvline(70, color="#C5C8CC", linewidth=0.5, linestyle=":")
            reference = rapid_rate if panel == 0 else rapid_steps
            if method != "rapid":
                axis.plot(
                    rapid_x,
                    reference,
                    color=style.ORANGE,
                    linewidth=1.0,
                    alpha=0.75,
                    linestyle=(0, (3, 2)),
                    zorder=2,
                )
            axis.plot(
                rate_x,
                rate_y if panel == 0 else steps_y,
                color=style.ORANGE if method == "rapid" else style.BLUE,
                linewidth=1.35,
                zorder=3,
            )
            axis.set_xlim(0, 100)
            axis.tick_params(labelsize=6.5, length=2, pad=1.5)
            axis.spines["left"].set_linewidth(0.55)
            axis.spines["bottom"].set_linewidth(0.55)
            if column:
                axis.tick_params(labelleft=False)
        top.set(ylim=(0, 45), yticks=[0, 20, 40])
        bottom.set(ylim=(0, 6), yticks=[0, 3, 6], xticks=[0, 30, 70, 100])
        top.tick_params(labelbottom=False)
        top.set_title(
            f"({chr(97 + index)})  {DYNAMIC_NAMES[method]}",
            loc="left",
            fontsize=8.8,
            pad=3,
            color=style.ORANGE if method == "rapid" else "#202A35",
        )
        if column == 0:
            top.set_ylabel("Rate (tok/s)", fontsize=7)
            bottom.set_ylabel("Steps / cycle", fontsize=7)
        if row < 2:
            bottom.tick_params(labelbottom=False)
    figure.legend(
        handles=[
            Line2D([], [], color=style.BLUE, linewidth=1.4, label="Panel method"),
            Line2D(
                [],
                [],
                color=style.ORANGE,
                linewidth=1.2,
                linestyle="--",
                label="RAPID reference",
            ),
            Patch(facecolor="#EBECEE", label="YOLO active"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.995),
        ncol=3,
        fontsize=8,
        handlelength=2.4,
        columnspacing=2.0,
    )
    figure.text(0.53, 0.018, "Global non-prefill cycle", ha="center", fontsize=8.5)
    save(figure, "fig8_dynamic_load", output)
    print("  fig8_dynamic_load: 9 method panels (rate + steps/cycle, YOLO window 30-70)")


def _controller_series(
    rows: list[dict[str, str]], variant: str, field: str
) -> tuple[np.ndarray, np.ndarray]:
    by_time: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        if row["variant"] == variant and row["dataset"] == "c4":
            by_time[float(row["time_s"])].append(float(row[field]))
    times = sorted(by_time)
    if not times:
        raise AssertionError(f"controller_response: no C4 rows for {variant}")
    return np.array(times), np.array([float(np.mean(by_time[t])) for t in times])


def figure9_controller_response(output: Path) -> None:
    rows = read("controller_response_timeseries.csv")
    if len(rows) != 1926:
        raise AssertionError(f"controller_response: expected 1926 rows, got {len(rows)} rows")
    variants = {
        "single_ref": ("Single-reference", style.ORANGE, "-"),
        "two_timescale": ("Two-timescale", style.BLUE, (0, (4, 2.5))),
    }
    if {row["variant"] for row in rows} != set(variants):
        raise AssertionError("controller_response: variant coverage differs")
    fields = [
        ("reference_ms", "Reference\n(ms/token)", (0, 35), [10, 20, 30]),
        ("local_latency_ms", "Local latency\n(ms/token)", (0, 52), [10, 20, 30, 40, 50]),
        ("esr_rho", r"ESR $\rho$", (0, 0.9), [0.0, 0.3, 0.6]),
    ]
    figure, axes = plt.subplots(3, 1, figsize=(3.25, 3.95), sharex=True,
                                layout="constrained")
    for axis, (field, label, ylim, yticks) in zip(axes, fields, strict=True):
        for variant, (_, color, linestyle) in variants.items():
            times, values = _controller_series(rows, variant, field)
            # The data file is the measured figure curve; no additional smoothing.
            axis.plot(times, values, color=color, linewidth=1.4, linestyle=linestyle)
        axis.axvspan(0, 120, color=style.EVENT_BAND, zorder=0)
        axis.axvline(0, color="#C5C8CC", linewidth=0.6, linestyle=":")
        axis.set(ylim=ylim, yticks=yticks, ylabel=label,
                 xlim=(-40, 120), xticks=[-40, 0, 40, 80, 120])
        style.clean_axis(axis)
    axes[-1].set_xlabel("Time from slowdown onset (s)")
    for axis, label in zip(axes, ("(a)", "(b)", "(c)"), strict=True):
        axis.text(-0.02, 1.03, label, transform=axis.transAxes, fontsize=8.6,
                  fontweight="bold", ha="left", va="bottom")
    figure.legend(
        handles=[
            Line2D([], [], color=color, linewidth=1.4, linestyle=linestyle, label=name)
            for name, color, linestyle in variants.values()
        ],
        loc="outside upper center", ncol=2, fontsize=7.4,
    )
    save(figure, "fig9_controller_response", output)
    print("  fig9_controller_response: 3 stacked panels on the C4 slowdown trace")


def figure10_network_sweeps(output: Path) -> None:
    sweeps = [
        ("figure10_delay_sweep.csv", "Additional delay (ms)", 36),
        ("figure10_jitter_sweep.csv", "Jitter setting (ms)", 36),
        ("figure10_event_sweep.csv", "500-ms event probability (%)", 45),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), layout="constrained")
    for axis, (name, xlabel, expected) in zip(axes, sweeps, strict=True):
        rows = read(name)
        if len(rows) != expected:
            raise AssertionError(f"{name}: expected {expected} rows, got {len(rows)}")
        if {row["method_name"] for row in rows} != set(NAMES.values()):
            raise AssertionError(f"{name}: method coverage differs from the nine methods")
        x_values = sorted({float(row["x"]) for row in rows})
        for method in METHODS:
            display = NAMES[method]
            points = sorted(
                (float(row["x"]), float(row["tps"]))
                for row in rows
                if row["method_name"] == display
            )
            color, marker = METHOD_STYLES[method]
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                label=display,
                color=color,
                marker=marker,
                linewidth=1.5 if method == "rapid" else 0.9,
                linestyle="-" if method == "rapid" else "--",
                markersize=3.2,
            )
        axis.set(xlabel=xlabel, ylabel="Throughput (tok/s)", xticks=x_values)
        style.clean_axis(axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="outside upper center",
        ncol=5,
        fontsize=7.2,
        columnspacing=1.6,
        handlelength=2.0,
    )
    save(figure, "fig10_network_sweeps", output)
    print("  fig10_network_sweeps: delay (4) / jitter (4) / events (5) x 9 methods")


def figure11_parameter_sweeps(output: Path) -> None:
    rows = read("figure11_parameter_sweeps.csv")
    if len(rows) != 25:
        raise AssertionError(f"figure11_parameter_sweeps: expected 25 rows, got {len(rows)}")
    grouped: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row["panel"]][row["series"]].append(
            (float(row["x"]), float(row["tps"]))
        )
    for panel in grouped:
        for series in grouped[panel]:
            grouped[panel][series].sort()

    figure, axes = plt.subplots(2, 2, figsize=(7.0, 4.1), layout="constrained")

    axis = axes[0, 0]
    fixed = grouped["fixed_budget"]["Fixed Budget"]
    axis.plot(
        [point[0] for point in fixed],
        [point[1] for point in fixed],
        color=style.BLUE,
        marker="s",
        label="Fixed Budget",
    )
    rapid = {point[1] for point in grouped["fixed_budget"]["RAPID"]}
    if len(rapid) != 1:
        raise AssertionError("fixed-budget panel: RAPID reference is not constant")
    axis.axhline(rapid.pop(), color=style.ORANGE, linestyle="--", label="RAPID")
    axis.set(
        xlabel="Fixed draft budget",
        ylabel="Throughput (tok/s)",
        xticks=[point[0] for point in fixed],
    )
    axis.legend(fontsize=7.2)
    style.clean_axis(axis)
    style.panel_label(axis, "(a)")

    axis = axes[0, 1]
    target = grouped["target_acceptance"]["Mean"]
    axis.plot(
        [point[0] for point in target],
        [point[1] for point in target],
        color=style.BLUE,
        marker="o",
    )
    axis.set(
        xlabel=r"Target acceptance $A^*$",
        ylabel="Throughput (tok/s)",
        xticks=[point[0] for point in target],
    )
    axis.set_ylim(24, 28)
    style.clean_axis(axis)
    style.panel_label(axis, "(b)")

    axis = axes[1, 0]
    alpha = grouped["esr_alpha"]["Mean"]
    axis.plot(
        [point[0] for point in alpha],
        [point[1] for point in alpha],
        color=style.BLUE,
        marker="o",
        label="Mean",
    )
    axis.set(
        xlabel=r"ESR coefficient $\alpha$",
        ylabel="Throughput (tok/s)",
        xticks=[point[0] for point in alpha],
    )
    axis.set_ylim(24, 28)
    axis.legend(fontsize=7.2)
    style.clean_axis(axis)
    style.panel_label(axis, "(c)")

    axis = axes[1, 1]
    gains = grouped["pid_gain"]["PID"]
    axis.plot(
        [point[0] for point in gains],
        [point[1] for point in gains],
        color=style.ORANGE,
        marker="o",
        label="PID",
    )
    pure_p = {point[1] for point in grouped["pid_gain"]["P only"]}
    if len(pure_p) != 1:
        raise AssertionError("PID-gain panel: P-only reference is not constant")
    axis.axhline(pure_p.pop(), color=style.BLUE, linestyle="--", label="P only")
    axis.set(
        xlabel=r"Proportional gain $K_P$",
        ylabel="Throughput (tok/s)",
        xscale="log",
        xticks=[point[0] for point in gains],
    )
    axis.set_xticklabels([f"{int(value)}" for value, _ in gains])
    axis.set_ylim(22, 26)
    axis.legend(fontsize=7.2, ncol=2)
    style.clean_axis(axis)
    style.panel_label(axis, "(d)")

    save(figure, "fig11_parameter_sweeps", output)
    print("  fig11_parameter_sweeps: budget / target / alpha / gain panels")


def main() -> None:
    global DATA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    args = parser.parse_args()

    DATA = args.data_dir
    output = args.output_dir
    style.configure()
    print(f"rendering measured figures from {DATA} into {output}")
    figure5_calibration(output)
    figure8_dynamic_load(output)
    figure9_controller_response(output)
    figure10_network_sweeps(output)
    figure11_parameter_sweeps(output)
    written = sorted(path.name for path in output.iterdir() if path.suffix in (".pdf", ".png"))
    print(f"wrote {len(written)} figure files (5 PDF + 5 PNG): {', '.join(written)}")


if __name__ == "__main__":
    main()
