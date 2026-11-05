"""Rebuild the paper's tables 3-12 from the frozen experimental data.

``data/runs.csv`` stores every measured cell together with three repeated runs.
Every value written here is the arithmetic mean of those three runs, in the
same ordering the manuscript uses.  The script only reads ``data/`` and writes
CSV plus booktabs LaTeX under the requested output directory (default
``outputs/tables``, which is git-ignored).

Usage:
    python scripts/rebuild_tables.py [--output outputs/tables]
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from statistics import mean

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from methods import (  # noqa: E402
    DATASET_NAMES,
    DATASET_TABLE_NAMES,
    DATASETS,
    METHODS,
    NAMES,
    PAIR_NAMES,
    PAIRS,
)

DEFAULT_DATA = ROOT / "data"
DEFAULT_OUTPUT = ROOT / "outputs" / "tables"

Cells = dict[tuple[str, str, str], list[dict[str, str]]]

EXPECTED_RUNS = 1653
EXPECTED_CELLS = 551
EXPECTED_CONFIGS = 117
RUNS_PER_CELL = 3

# experiment -> (number of cells, metric)
EXPERIMENTS = {
    "table3_throughput": (180, "aggregate_throughput"),
    "table4_latency": (180, "per_token_latency"),
    "table5_energy": (36, "energy_per_token"),
    "table6_request_latency": (36, None),
    "table7_ablation": (8, None),
    "table8_contention": (36, "aggregate_throughput"),
    "table9_controller_variants": (9, "aggregate_throughput"),
    "table10_esr": (30, None),
    "table11_waste": (12, None),
    "table12_pdfr": (24, "pdfr"),
}

REQUEST_LATENCY_METRICS = {
    "mean": "request_latency_mean",
    "p50": "request_latency_p50",
    "p95": "request_latency_p95",
    "p99": "request_latency_p99",
}
ESR_COLUMNS = [
    ("single_ref|rho_median", "esr_rho_median", 2),
    ("single_ref|eta_mean", "esr_eta_mean", 2),
    ("single_ref|rounds_gt_03_pct", "rounds_rho_gt_0_3", 1),
    ("two_timescale|rho_median", "esr_rho_median", 2),
    ("two_timescale|eta_mean", "esr_eta_mean", 2),
    ("two_timescale|rounds_gt_03_pct", "rounds_rho_gt_0_3", 1),
]
ABLATION_ROWS = [
    ("Fixed 16", "NPDG", "Fixed 16|NPDG"),
    ("Fixed 16", "IPDG", "Fixed 16|IPDG"),
    ("CAAC", "NPDG", "CAAC|NPDG"),
    ("CAAC", "IPDG", "CAAC|IPDG"),
]
CONTROLLER_ROWS = [
    ("full_caac", "Full CAAC"),
    ("latency_only", "Latency only"),
    ("acceptance_only", "Acceptance only"),
]
WASTE_ROWS = [
    ("None", "NPDG", "None|NPDG"),
    ("None", "IPDG", "None|IPDG"),
    ("Compute", "NPDG", "Compute|NPDG"),
    ("Compute", "IPDG", "Compute|IPDG"),
]
WASTE_COLUMNS = [
    ("all_pct", "all_discarded_drafting", "All (\\%)"),
    ("after_resp_pct", "post_response_drafting", "After resp. (\\%)"),
    ("after_share_pct", "post_response_share", "After share (\\%)"),
]
PDFR_COLUMNS = [
    ("npdg|base", "npdg_base", "NPDG Base"),
    ("npdg|instr", "npdg_instr", "NPDG Instr."),
    ("npdg|specbench", "npdg_specbench", "NPDG SpecBench"),
    ("ipdg|base", "ipdg_base", "IPDG Base"),
    ("ipdg|instr", "ipdg_instr", "IPDG Instr."),
    ("ipdg|specbench", "ipdg_specbench", "IPDG SpecBench"),
]
CONTENTION_COLUMNS = ["none", "compute", "yolo", "bandwidth"]
CONTENTION_NAMES = ["None", "Compute", "YOLO", "Bandwidth"]
PHASE_COLUMNS = ["no_load", "phased_compute", "phased_yolo"]
PHASE_NAMES = ["No load", "Phased compute", "Phased YOLO"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_cells(path: Path) -> Cells:
    """Group the run rows into cells keyed by (experiment, row, column)."""
    rows = read_rows(path)
    if len(rows) != EXPECTED_RUNS:
        raise AssertionError(f"{path}: expected {EXPECTED_RUNS} run rows, got {len(rows)}")
    cells: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row["experiment"], row["row"], row["column"])
        cells.setdefault(key, []).append(row)
    if len(cells) != EXPECTED_CELLS:
        raise AssertionError(f"{path}: expected {EXPECTED_CELLS} cells, got {len(cells)}")
    for key, group in cells.items():
        runs = sorted(int(row["run_id"]) for row in group)
        if len(group) != RUNS_PER_CELL or runs != [1, 2, 3]:
            raise AssertionError(f"cell {key}: expected three runs 1/2/3, got {runs}")
        metrics = {row["metric"] for row in group}
        if len(metrics) != 1:
            raise AssertionError(f"cell {key}: mixed metrics {metrics}")
    return cells


def cell(cells: Cells,
         experiment: str, row: str, column: str) -> float:
    return mean(float(item["value"]) for item in cells[(experiment, row, column)])


def validate(cells: Cells) -> None:
    """Check the frozen coverage: 9 methods, 4 pairs, 5 datasets, 551 cells."""
    for experiment, (expected_cells, metric) in EXPERIMENTS.items():
        found_cells = [key for key in cells if key[0] == experiment]
        if len(found_cells) != expected_cells:
            raise AssertionError(
                f"{experiment}: expected {expected_cells} cells, got "
                f"{len(found_cells)}"
            )
        if metric is not None:
            metrics = {cells[key][0]["metric"] for key in found_cells}
            if metrics != {metric}:
                raise AssertionError(f"{experiment}: unexpected metrics {metrics}")

    dataset_labels = [DATASET_NAMES[slug] for slug in DATASETS]
    table34 = {(f"{pair}|{label}", method) for pair in PAIRS
               for label in dataset_labels for method in METHODS}
    for experiment in ("table3_throughput", "table4_latency"):
        found34 = {(row, column) for (exp, row, column) in cells if exp == experiment}
        if found34 != table34:
            raise AssertionError(f"{experiment}: coverage differs from 4x5x9")
    expectations = {
        "table5_energy": {(pair, method) for pair in PAIRS for method in METHODS},
        "table6_request_latency": {
            (method, column) for method in METHODS for column in REQUEST_LATENCY_METRICS
        },
        "table8_contention": {
            (method, column) for method in METHODS for column in CONTENTION_COLUMNS
        },
    }
    for experiment, expected in expectations.items():
        found = {(row, column) for (exp, row, column) in cells if exp == experiment}
        if found != expected:
            raise AssertionError(f"{experiment}: unexpected row/column coverage")
    expectations = {
        "table7_ablation": {(label, column) for _, _, label in ABLATION_ROWS
                            for column in ("tok_per_s", "J_per_token")},
        "table9_controller_variants": {(row, column) for row, _ in CONTROLLER_ROWS
                                       for column in PHASE_COLUMNS},
        "table10_esr": {(DATASET_TABLE_NAMES[slug], column)
                        for slug in DATASETS for column, _, _ in ESR_COLUMNS},
        "table11_waste": {(label, column) for _, _, label in WASTE_ROWS
                          for column, _, _ in WASTE_COLUMNS},
        "table12_pdfr": {(pair, column) for pair in PAIRS
                         for column, _, _ in PDFR_COLUMNS},
    }
    for experiment, expected in expectations.items():
        found = {(row, column) for (exp, row, column) in cells if exp == experiment}
        if found != expected:
            raise AssertionError(f"{experiment}: unexpected row/column coverage")


def validate_configs(path: Path) -> None:
    configs = read_rows(path)
    if len(configs) != EXPECTED_CONFIGS:
        raise AssertionError(f"{path}: expected {EXPECTED_CONFIGS} rows")
    if len({config["config_id"] for config in configs}) != EXPECTED_CONFIGS:
        raise AssertionError(f"{path}: duplicate config_id")
    if {config["method"] for config in configs} != set(METHODS):
        raise AssertionError(f"{path}: method coverage differs from the nine methods")
    if {config["pair"] for config in configs} != set(PAIRS):
        raise AssertionError(f"{path}: pair coverage differs from the four pairs")
    if {config["target_hardware"] for config in configs} != {"NVIDIA A800 80GB"}:
        raise AssertionError(f"{path}: target hardware is not the measured A800")
    if {config["edge_hardware"] for config in configs} != {"Jetson Orin NX 16GB"}:
        raise AssertionError(f"{path}: edge hardware is not the measured Jetson")


def escape(text: str) -> str:
    return text.replace("\\", r"\textbackslash{}").replace("_", r"\_").replace(
        "&", r"\&"
    ).replace("%", r"\%")


def write_table(
    output: Path,
    name: str,
    csv_header: Sequence[str],
    csv_rows: Iterable[Sequence[object]],
    *,
    caption: str,
    label: str,
    tex_header: Sequence[str],
    tex_rows: Iterable[str],
    wide: bool = True,
    column_spec: str | None = None,
) -> None:
    """Write one ``<name>.csv`` / ``<name>.tex`` pair and report it."""
    rows = [list(row) for row in csv_rows]
    with (output / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(csv_header)
        writer.writerows(rows)

    environment = "table*" if wide else "table"
    spec = column_spec or ("l" + "r" * (len(tex_header) - 1))
    lines = [
        rf"\begin{{{environment}}}[t]",
        r"\centering",
        r"\footnotesize",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if wide:
        lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}" + spec + "@{}}")
    else:
        lines.append(r"\begin{tabular}{@{}" + spec + "@{}}")
    lines += [r"\toprule", " & ".join(tex_header) + r" \\", r"\midrule"]
    lines += list(tex_rows)
    lines += [
        r"\bottomrule",
        r"\end{tabular*}" if wide else r"\end{tabular}",
        rf"\end{{{environment}}}",
    ]
    (output / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{name}: {len(rows)} rows -> {output / (name + '.csv')}")


def formatted(values: Sequence[float], digits: int, *, minimize: bool = False,
              rank: bool = False) -> list[str]:
    """Format values, optionally marking the best and second-best in a row."""
    rounded = [round(value, digits) for value in values]
    rank_values = sorted(set(rounded), reverse=not minimize)
    best = rank_values[0]
    second = rank_values[1] if len(rank_values) > 1 else None
    texts = []
    for value in rounded:
        text = f"{value:.{digits}f}"
        if rank and value == best:
            text = r"\textbf{" + text + "}"
        elif rank and value == second:
            text = r"\underline{" + text + "}"
        texts.append(text)
    return texts


def table03(cells: Cells, output: Path) -> None:
    header = ["pair", "dataset", *METHODS]
    rows, tex_rows = [], []
    for pair in PAIRS:
        for slug in DATASETS:
            label = DATASET_NAMES[slug]
            values = [cell(cells, "table3_throughput", f"{pair}|{label}", method)
                      for method in METHODS]
            rows.append([pair, slug, *(round(value, 2) for value in values)])
            marked = formatted(values, 2, rank=True)
            tex_rows.append(
                f"{escape(PAIR_NAMES[pair])} & {escape(DATASET_NAMES[slug])} & "
                + " & ".join(marked) + r" \\"
            )
    write_table(
        output,
        "table03_throughput",
        header,
        rows,
        caption=(
            "Aggregate throughput (tok/s) across four target/draft model pairs and "
            "five datasets. Bold and underline mark the highest and second-highest "
            "value in each row."
        ),
        label="tab:end_to_end",
        tex_header=["Target/Draft", "Dataset", *[NAMES[m] for m in METHODS]],
        tex_rows=tex_rows,
        column_spec="ll" + "r" * len(METHODS),
    )


def table04(cells: Cells, output: Path) -> None:
    header = ["pair", "dataset", *METHODS]
    rows, tex_rows = [], []
    for pair in PAIRS:
        for slug in DATASETS:
            label = DATASET_NAMES[slug]
            values = [cell(cells, "table4_latency", f"{pair}|{label}", method)
                      for method in METHODS]
            rows.append([pair, slug, *(round(value, 1) for value in values)])
            marked = formatted(values, 1, minimize=True, rank=True)
            tex_rows.append(
                f"{escape(PAIR_NAMES[pair])} & {escape(DATASET_NAMES[slug])} & "
                + " & ".join(marked) + r" \\"
            )
    write_table(
        output,
        "table04_latency",
        header,
        rows,
        caption=(
            "Per-token decoding latency without contention (ms/token) across four "
            "target/draft model pairs and five datasets. Bold and underline mark the "
            "lowest and second-lowest value in each row."
        ),
        label="restored:tab:end_to_end_latency",
        tex_header=["Target/Draft", "Dataset", *[NAMES[m] for m in METHODS]],
        tex_rows=tex_rows,
        column_spec="ll" + "r" * len(METHODS),
    )


def table05(cells: Cells, output: Path) -> None:
    rows, tex_rows = [], []
    for pair in PAIRS:
        values = [cell(cells, "table5_energy", pair, method) for method in METHODS]
        rows.append([pair, *(round(value, 2) for value in values)])
        marked = formatted(values, 2, minimize=True, rank=True)
        tex_rows.append(f"{escape(PAIR_NAMES[pair])} & " + " & ".join(marked) + r" \\")
    write_table(
        output,
        "table05_energy",
        ["pair", *METHODS],
        rows,
        caption=(
            "Energy per generated token (J/token), arithmetic means over five "
            "datasets. The boundary sums Jetson VDD\\_IN and target GPU board energy "
            "once over the shared request window. Bold and underline mark the lowest "
            "and second-lowest values. This is not wall-plug energy."
        ),
        label="tab:energy",
        tex_header=["Target/Draft", *[NAMES[m] for m in METHODS]],
        tex_rows=tex_rows,
        column_spec="l" + "r" * len(METHODS),
    )


def table06(cells: Cells, output: Path) -> None:
    best = [
        min(round(cell(cells, "table6_request_latency", method, metric), 2)
            for method in METHODS)
        for metric in REQUEST_LATENCY_METRICS
    ]
    rows, tex_rows = [], []
    for method in METHODS:
        rounded = [
            round(cell(cells, "table6_request_latency", method, metric), 2)
            for metric in REQUEST_LATENCY_METRICS
        ]
        rows.append([method, *rounded])
        texts = [
            r"\textbf{" + f"{value:.2f}" + "}" if value == best[index] else f"{value:.2f}"
            for index, value in enumerate(rounded)
        ]
        tex_rows.append(f"{escape(NAMES[method])} & " + " & ".join(texts) + r" \\")
    write_table(
        output,
        "table06_request_latency",
        ["method", "mean", "p50", "p95", "p99"],
        rows,
        caption=(
            "Request-level end-to-end latency (s) for Qwen3-14B/0.6B with three "
            "clients. Each value aggregates 25 requests per method from the archived "
            "client clocks. Every request includes prefill, waiting, and all "
            "decoding rounds. Bold marks the lowest value in each column."
        ),
        label="tab:request_latency",
        tex_header=["Method", "Mean", "P50", "P95", "P99"],
        tex_rows=tex_rows,
        wide=False,
    )


def table07(cells: Cells, output: Path) -> None:
    rows, tex_rows = [], []
    for budget, execution, label in ABLATION_ROWS:
        throughput = cell(cells, "table7_ablation", label, "tok_per_s")
        energy = cell(cells, "table7_ablation", label, "J_per_token")
        rows.append([budget, execution, round(throughput, 2), round(energy, 2)])
        tex_rows.append(
            f"{escape(budget)} & {escape(execution)} & {throughput:.2f} & {energy:.2f}"
            + r" \\"
        )
    write_table(
        output,
        "table07_ablation",
        ["budget", "execution", "tok_per_s", "j_per_token"],
        rows,
        caption=(
            "CAAC/IPDG ablation with three clients sharing the edge device on "
            "Qwen3-14B/0.6B. Values are five-dataset arithmetic means under the "
            "supplementary protocol."
        ),
        label="tab:ablation_2x2",
        tex_header=["Budget", "Execution", "tok/s", "J/token"],
        tex_rows=tex_rows,
        wide=False,
    )


def table08(cells: Cells, output: Path) -> None:
    rows, tex_rows = [], []
    for method in METHODS:
        values = [cell(cells, "table8_contention", method, column)
                  for column in CONTENTION_COLUMNS]
        rows.append([method, *(round(value, 2) for value in values)])
        tex_rows.append(
            f"{escape(NAMES[method])} & " + " & ".join(f"{value:.2f}" for value in values)
            + r" \\"
        )
    write_table(
        output,
        "table08_contention",
        ["method", *CONTENTION_COLUMNS],
        rows,
        caption=(
            "Throughput (tok/s) with Qwen3-14B/0.6B and three clients, averaged over "
            "five datasets. Load is static within a condition."
        ),
        label="tab:contention_type",
        tex_header=["Method", *CONTENTION_NAMES],
        tex_rows=tex_rows,
        wide=False,
    )


def table09(cells: Cells, output: Path) -> None:
    rows, tex_rows = [], []
    for key, display in CONTROLLER_ROWS:
        values = [cell(cells, "table9_controller_variants", key, column)
                  for column in PHASE_COLUMNS]
        rows.append([key, *(round(value, 2) for value in values)])
        tex_rows.append(
            f"{escape(display)} & " + " & ".join(f"{value:.2f}" for value in values)
            + r" \\"
        )
    write_table(
        output,
        "table09_controller_variants",
        ["controller", *PHASE_COLUMNS],
        rows,
        caption=(
            "Controller variants (tok/s), Qwen3-14B/0.6B and three clients. Values "
            "average five datasets. The no-load column is static. The compute and YOLO "
            "columns use phased workloads and include normal, loaded, and "
            "recovery intervals in the shared window."
        ),
        label="tab:caac_variants",
        tex_header=["Controller", *PHASE_NAMES],
        tex_rows=tex_rows,
        wide=False,
    )


def table10(cells: Cells, output: Path) -> None:
    rows, tex_rows = [], []
    tex_header = ["Dataset"]
    for column, _, _ in ESR_COLUMNS:
        prefix, field = column.split("|")
        name = "Single-ref" if prefix == "single_ref" else "Two-timescale"
        label = {"rho_median": r"$\rho$ median",
                 "eta_mean": r"$\eta$ mean",
                 "rounds_gt_03_pct": r"rounds $\rho>0.3$ (\%)"}[field]
        tex_header.append(f"{name} {label}")
    for slug in ("c4", "mtbench", "oasst", "wikitext", "specbench"):
        label = DATASET_TABLE_NAMES[slug]
        values = [cell(cells, "table10_esr", label, column) for column, _, _ in ESR_COLUMNS]
        rows.append([slug, *(round(value, digits) for (_, _, digits), value
                             in zip(ESR_COLUMNS, values, strict=False))])
        texts = [
            f"{value:.{digits}f}"
            for (_, _, digits), value in zip(ESR_COLUMNS, values, strict=False)
        ]
        tex_rows.append(f"{escape(label)} & " + " & ".join(texts) + r" \\")
    write_table(
        output,
        "table10_esr",
        ["dataset", *[column.replace("|", "_") for column, _, _ in ESR_COLUMNS]],
        rows,
        caption=(
            "ESR response to a sustained slowdown, averaged over three runs with "
            "Qwen3-14B/0.6B."
        ),
        label="restored:tab:esr_control",
        tex_header=tex_header,
        tex_rows=tex_rows,
        column_spec="l" + "r" * len(ESR_COLUMNS),
    )


def table11(cells: Cells, output: Path) -> None:
    rows, tex_rows = [], []
    for contention, execution, label in WASTE_ROWS:
        values = [cell(cells, "table11_waste", label, column)
                  for column, _, _ in WASTE_COLUMNS]
        rows.append([contention, execution,
                     *(round(value, 1) for value in values)])
        tex_rows.append(
            f"{escape(contention)} & {escape(execution)} & "
            + " & ".join(f"{value:.1f}" for value in values) + r" \\"
        )
    write_table(
        output,
        "table11_waste",
        ["contention", "execution", *[column for column, _, _ in WASTE_COLUMNS]],
        rows,
        caption=(
            "Discarded drafting with Qwen3-14B/0.6B and three clients. All and After "
            "resp. are percentages of ordinary plus proactive drafting time; After "
            "share is the post-response fraction of discarded drafting. Each entry is "
            "a five-dataset arithmetic mean."
        ),
        label="restored:tab:waste",
        tex_header=["Contention", "Execution", *[name for _, _, name in WASTE_COLUMNS]],
        tex_rows=tex_rows,
        wide=False,
    )


def table12(cells: Cells, output: Path) -> None:
    rows, tex_rows = [], []
    # The manuscript lists the PDFR pairs in this order.
    for pair in ["qwen14b_17b", "qwen32b_17b", "qwen14b_06b", "llama8b_1b"]:
        values = [cell(cells, "table12_pdfr", pair, column)
                  for column, _, _ in PDFR_COLUMNS]
        rows.append([pair, *(round(value, 2) for value in values)])
        tex_rows.append(
            f"{escape(PAIR_NAMES[pair])} & " + " & ".join(f"{value:.2f}" for value in values)
            + r" \\"
        )
    write_table(
        output,
        "table12_pdfr",
        ["pair", *[column.replace("|", "_") for column, _, _ in PDFR_COLUMNS]],
        rows,
        caption=(
            "Three-run mean PDFR, computed as $100\\%\\cdot(1-N_{\\mathrm{retained}}/"
            "N_{\\mathrm{eligible}})$, for NPDG and IPDG. Lower is better."
        ),
        label="restored:tab:ipdg_pdfr",
        tex_header=["Target/Draft", *[name for _, _, name in PDFR_COLUMNS]],
        tex_rows=tex_rows,
        column_spec="l" + "r" * len(PDFR_COLUMNS),
    )


def report_spot_checks(cells: Cells) -> None:
    """Assert a few paper values so the table rebuild cannot drift silently."""
    checks = [
        ("table3 qwen14b_06b C4 RAPID", cell(cells, "table3_throughput",
                                             "qwen14b_06b|C4", "rapid"), 25.52, 2),
        ("table3 qwen14b_17b MTBench RAPID", cell(cells, "table3_throughput",
                                                  "qwen14b_17b|MTBench", "rapid"), 15.11, 2),
        ("table4 qwen14b_06b C4 RAPID", cell(cells, "table4_latency",
                                             "qwen14b_06b|C4", "rapid"), 90.4, 1),
        ("table4 qwen32b_17b SpecBench RAPID", cell(cells, "table4_latency",
                                                    "qwen32b_17b|SpecBench", "rapid"), 93.9, 1),
        ("table5 qwen14b_06b RAPID", cell(cells, "table5_energy", "qwen14b_06b",
                                          "rapid"), 7.00, 2),
        ("table6 RAPID mean", cell(cells, "table6_request_latency", "rapid",
                                   "mean"), 11.84, 2),
        ("table7 CAAC|IPDG tok/s", cell(cells, "table7_ablation", "CAAC|IPDG",
                                        "tok_per_s"), 24.57, 2),
        ("table8 RAPID no load", cell(cells, "table8_contention", "rapid",
                                      "none"), 24.57, 2),
        ("table9 full CAAC no load", cell(cells, "table9_controller_variants",
                                          "full_caac", "no_load"), 24.57, 2),
        ("table10 C4 two-timescale rho", cell(cells, "table10_esr", "C4",
                                              "two_timescale|rho_median"), 0.19, 2),
        ("table11 None|NPDG all", cell(cells, "table11_waste", "None|NPDG",
                                       "all_pct"), 39.9, 1),
        ("table12 qwen14b_06b NPDG base", cell(cells, "table12_pdfr", "qwen14b_06b",
                                               "npdg|base"), 47.43, 2),
    ]
    for name, value, expected, digits in checks:
        if round(value, digits) != expected:
            raise AssertionError(f"{name}: mean {value:.4f} rounds to "
                                 f"{round(value, digits)}, expected {expected}")
    print("\nspot checks (three-run means) match the manuscript:")
    for name, value, expected, digits in checks:
        print(f"  {name:38s} {value:10.4f} -> {round(value, digits)} (paper {expected})")

    print("\nRAPID five-dataset means (printed in the manuscript):")
    print("  throughput (tok/s):", end=" ")
    means = [mean(cell(cells, "table3_throughput", f"{pair}|{DATASET_NAMES[slug]}",
                       "rapid") for slug in DATASETS) for pair in PAIRS]
    print(" / ".join(f"{value:.2f}" for value in means))
    print("  latency (ms/token):", end=" ")
    means = [mean(cell(cells, "table4_latency", f"{pair}|{DATASET_NAMES[slug]}",
                       "rapid") for slug in DATASETS) for pair in PAIRS]
    print(" / ".join(f"{value:.1f}" for value in means))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    args = parser.parse_args()

    output = args.output
    data_dir = args.data_dir
    if data_dir.resolve() in output.resolve().parents:
        parser.error("refusing to write tables inside the frozen data/ directory")
    output.mkdir(parents=True, exist_ok=True)

    cells = load_cells(data_dir / "runs.csv")
    validate(cells)
    validate_configs(data_dir / "configs.csv")
    print(f"validated {EXPECTED_RUNS} runs / {EXPECTED_CELLS} cells / "
          f"{EXPECTED_CONFIGS} configurations")

    table03(cells, output)
    table04(cells, output)
    table05(cells, output)
    table06(cells, output)
    table07(cells, output)
    table08(cells, output)
    table09(cells, output)
    table10(cells, output)
    table11(cells, output)
    table12(cells, output)
    report_spot_checks(cells)
    print(f"\nwrote csv + booktabs tex for 10 tables under {output}")



if __name__ == "__main__":
    main()
