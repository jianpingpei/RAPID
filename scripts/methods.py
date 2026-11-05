"""Shared method, model-pair and dataset labels for the publication scripts.

The slugs are the frozen identifiers stored in ``data/runs.csv`` and
``data/configs.csv``; the display names are the ones used in the manuscript
tables and figures.  ``scripts/rebuild_tables.py`` and
``scripts/rebuild_figures.py`` import this module so that ordering and labels
stay identical everywhere.
"""

METHODS = [
    "fixed",
    "cost",
    "specedge",
    "opt_tree",
    "svip",
    "edgellm",
    "pearl",
    "sequential",
    "rapid",
]

# Names used inside compact tables and figure legends.
NAMES = {
    "fixed": "Fixed",
    "cost": "Cost",
    "specedge": "SpecEdge",
    "opt_tree": "OPT-Tree",
    "svip": "SVIP",
    "edgellm": "EdgeLLM",
    "pearl": "PEARL",
    "sequential": "Sequential",
    "rapid": "RAPID",
}

# Full names used in the paper's comparison section.
FULL_NAMES = {
    "fixed": "Fixed Budget",
    "cost": "Cost-Model (Online Speculative Decoding)",
    "specedge": "SpecEdge",
    "opt_tree": "OPT-Tree",
    "svip": "SVIP",
    "edgellm": "EdgeLLM",
    "pearl": "PEARL",
    "sequential": "Sequential",
    "rapid": "RAPID",
}

PAIRS = ["qwen14b_06b", "qwen14b_17b", "llama8b_1b", "qwen32b_17b"]

PAIR_NAMES = {
    "qwen14b_06b": "Qwen3-14B/0.6B",
    "qwen14b_17b": "Qwen3-14B/1.7B",
    "llama8b_1b": "Llama-3.1-8B/3.2-1B",
    "qwen32b_17b": "Qwen3-32B/1.7B",
}

DATASETS = ["c4", "wikitext", "mtbench", "oasst", "specbench"]

DATASET_NAMES = {
    "c4": "C4",
    "wikitext": "WikiText",
    "mtbench": "MTBench",
    "oasst": "OASST",
    "specbench": "SpecBench",
}

# Some tables spell the datasets slightly differently in the manuscript.
DATASET_TABLE_NAMES = {
    "c4": "C4",
    "wikitext": "WikiText",
    "mtbench": "MT-Bench",
    "oasst": "OASST",
    "specbench": "SpecBench",
}

MODELS = {
    "qwen14b_06b": {"target": "Qwen3-14B", "draft": "Qwen3-0.6B"},
    "qwen14b_17b": {"target": "Qwen3-14B", "draft": "Qwen3-1.7B"},
    "llama8b_1b": {"target": "Llama-3.1-8B-Instruct", "draft": "Llama-3.2-1B-Instruct"},
    "qwen32b_17b": {"target": "Qwen3-32B", "draft": "Qwen3-1.7B"},
}
