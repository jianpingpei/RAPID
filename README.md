# RAPID: Resource-Aware Proactive and Interruptible Speculative Decoding

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24127252.svg)](https://doi.org/10.5281/zenodo.24127252)

RAPID is a method for edge-cloud speculative decoding of large language models
under changing edge load. A small edge model drafts candidate tokens and a large
cloud target model verifies them in parallel. Proactive draft generation overlaps
edge drafting with network transmission and cloud verification, and RAPID keeps
that overlap productive in two ways:

- **CAAC (Congestion-Aware Adaptive Controller)** sets the draft length before
  each round from measured local token latency and recent acceptance. A physical
  path turns the local slowdown into a bounded penalty; a logical path converts
  the acceptance error into a length proposal through a PID law; a fusion stage
  combines them and quantizes the result.
- **IPDG (Interruptible Proactive Draft Generation)** builds the proactive draft
  one tree-expansion step at a time and checks cloud feedback after each
  completed step. A matching result retains and remaps the proactive branch; a
  root mismatch stops expansion at the next boundary, restores the saved
  tree/KV-cache state, and discards the invalid branch.

The method is evaluated against eight baselines (Fixed Budget, Cost-Model,
SpecEdge, OPT-Tree, SVIP, EdgeLLM, PEARL, Sequential) on four target/draft model
pairs and five datasets, with three repeated runs per condition.

This repository is the minimal publication artifact for the manuscript
*RAPID: Resource-Aware Proactive and Interruptible Speculative Decoding for
Edge-Cloud Large Language Model Inference* (Software: Practice and Experience).
It contains the reference implementation of the RAPID control logic, the
experiment data behind the paper's tables and measurement figures, and the
scripts that rebuild those tables and figures from the data.

## Contents

| Path | Contents |
|---|---|
| `src/rapid/` | Framework-free reference implementation: CAAC (`caac.py`), IPDG (`ipdg.py`), and the round workflow (`round.py`). Python standard library only. |
| `data/runs.csv` | Per-run measurements behind Tables 3-12: 1,653 rows = 551 measured cells x 3 runs. |
| `data/configs.csv` | The 117 measurement configurations (model pair, clients, method, protocol, hardware, precision, controller settings). |
| `data/curves/` | The seven curve files behind Figures 5, 8, 9, 10 and 11. |
| `scripts/rebuild_tables.py` | Rebuilds Tables 3-12 (CSV + booktabs LaTeX) from `data/runs.csv`. |
| `scripts/rebuild_figures.py` | Rebuilds Figures 5, 8, 9, 10 and 11 (PDF + PNG) from `data/curves/`. |
| `scripts/methods.py`, `scripts/paper_style.py` | Shared method/pair/dataset labels and figure style used by the two scripts. |

## Curve provenance

`data/curves/dynamic_load_timeseries.csv` holds the per-cycle dynamic-load
trace for Figure 8: one measured trace for each of the nine methods (101 cycles
per method, 909 rows). `data/curves/controller_response_timeseries.csv` holds
the C4 controller-response trace for Figure 9 (two controller variants, three
runs, 321 time points per run, 1,926 rows). The remaining five curve files are
the measured sweep derivatives behind Figures 5, 10 and 11.

## Quick start

Requirements: Python 3.10 or newer, `numpy` and `matplotlib` for the figure
script.

```bash
python -m pip install numpy matplotlib

# Tables 3-12: CSV plus booktabs LaTeX under outputs/tables/
python scripts/rebuild_tables.py

# Figures 5, 8, 9, 10 and 11: PDF plus 300-dpi PNG under outputs/figures/
python scripts/rebuild_figures.py
```

`rebuild_tables.py` validates the data before writing: 1,653 runs, 551 cells,
exactly three runs per cell, nine methods, four model pairs, five datasets, 117
configurations and ten experiments. Every regenerated value is the arithmetic
mean of the three runs of its cell. `rebuild_figures.py` reads the released
curve files and writes one PDF and one PNG per figure.

Expected spot values after rebuilding (also printed by the script):

```text
Table 3, RAPID C4, Qwen3-14B/0.6B      25.52 tok/s
Table 4, RAPID C4, Qwen3-14B/0.6B      90.4 ms/token
Table 5, RAPID Qwen3-14B/0.6B          7.00 J/token
RAPID five-dataset mean throughput    24.57 / 13.78 / 20.33 / 12.10 tok/s
RAPID five-dataset mean latency       88.0 / 111.5 / 113.4 / 127.3 ms/token
```

## Data

### `data/runs.csv`

Long format, one row per run of one paper-table cell. Columns:

| Column | Meaning |
|---|---|
| `experiment` | Paper table: `table3_throughput`, `table4_latency`, `table5_energy`, `table6_request_latency`, `table7_ablation`, `table8_contention`, `table9_controller_variants`, `table10_esr`, `table11_waste`, `table12_pdfr` |
| `config_id` | Configuration key; joins `data/configs.csv` |
| `row`, `column` | Position of the cell in the paper table layout |
| `metric`, `unit` | Metric name and unit of `value` |
| `run_id` | Run index, 1-3 |
| `value` | Measured value for this run and cell |
| `numerator`, `denominator` | Count components behind the value |
| `numerator_unit`, `denominator_unit` | Units of the count components |

The 1,653 rows are 551 measured cells x 3 runs. Rates and percentages satisfy
`value = numerator / denominator` or `value = 100 * numerator / denominator`;
distribution statistics keep the sample size in `denominator`. Reported values
are the three-run mean. Tables that aggregate datasets use the unweighted
arithmetic mean of the per-dataset values.

### `data/configs.csv`

117 configurations and 27 columns: `config_id`, model pair and models, client
count, method slug, protocol, load, planning/execution variants, hardware,
precision, link latency, KV-cache size, temperature, seed, fixed budget, target
acceptance, ESR coefficient, PID gains, and per-method planner parameters. All
configurations use a Jetson Orin NX 16 GB edge client and an NVIDIA A800 80 GB
target verifier over a WAN link with about 30 ms average round-trip latency,
FP16 draft and target weights, a 4,096-token KV cache, temperature 0, seed 42, a
fixed budget of 16 tokens, `A* = 0.75`, `alpha = 0.5` and
`(Kp, Ki, Kd) = (40, 0.8, 8)`.

Method slugs in the data are `fixed`, `cost` (Cost-Model, Online Speculative
Decoding), `specedge`, `opt_tree`, `svip`, `edgellm`, `pearl`, `sequential` and
`rapid`.

### `data/curves/`

| File | Rows | Paper object |
|---|---:|---|
| `figure5_calibration.csv` | 4 | Figure 5: worker count, mean GPU utilization, throughput |
| `dynamic_load_timeseries.csv` | 909 | Figure 8: output rate and ordinary expansion steps per cycle under a shared YOLO workload |
| `controller_response_timeseries.csv` | 1,926 | Figure 9: controller reference, local latency and ESR traces around a slowdown onset |
| `figure10_delay_sweep.csv` | 36 | Figure 10 (left): added-delay sweep, nine methods |
| `figure10_jitter_sweep.csv` | 36 | Figure 10 (middle): jitter sweep |
| `figure10_event_sweep.csv` | 45 | Figure 10 (right): 500-ms response-delay-event probability sweep |
| `figure11_parameter_sweeps.csv` | 25 | Figure 11: fixed-budget, target-acceptance, ESR-coefficient and PID-gain sweeps |

## Reference implementation

`src/rapid/` contains the algorithm-level implementation of the paper's control
logic, without model or framework dependencies:

- `caac.py`: the Congestion-Aware Adaptive Controller (physical path, logical
  path and fusion) with a typed configuration and decision object.
- `ipdg.py`: the interruptible expansion loop with boundary checks and mismatch
  stop.
- `round.py`: the round-level composition of CAAC and IPDG over runtime-supplied
  callbacks.

The reference package can be inspected without installing anything:

```python
import sys
sys.path.insert(0, "src")
from rapid import CAACConfig, CAACController, run_round

controller = CAACController(CAACConfig())
```

## Measurement environment

The paper measurements were collected on an NVIDIA Jetson Orin NX 16 GB edge
device and an NVIDIA A800 80 GB cloud verifier connected by a WAN link with a
measured average round-trip latency of about 30 ms. Draft and target weights are
FP16, the KV-cache capacity is 4,096 tokens, the temperature is 0 and the seed
is 42. Benchmark datasets are C4, WikiText, MT-Bench, OASST and SpecBench; model
pairs are Qwen3-14B/Qwen3-0.6B, Qwen3-14B/Qwen3-1.7B,
Llama-3.1-8B/Llama-3.2-1B, and Qwen3-32B/Qwen3-1.7B. Model weights, benchmark
prompt corpora and SGLang are external assets and are not redistributed here.

## Citation, license and attribution

Citation metadata is in [CITATION.cff](CITATION.cff). The archived artifact is
available in Zenodo at [10.5281/zenodo.24127252](https://doi.org/10.5281/zenodo.24127252)
(version v0.1.2). All material in this repository — the reference implementation,
the experiment data and the rebuild scripts — is distributed under the MIT
License; see [LICENSE](LICENSE). External model weights and benchmark prompt
corpora are not redistributed here. [NOTICE](NOTICE) records the external-asset
boundary and the relationship to the SpecEdge research prototype from early
development. RAPID itself is an independent method; SpecEdge is one of the eight
comparison baselines.
