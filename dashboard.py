#!/usr/bin/env python3
"""Interactive comparison dashboard for saved CommBench P2P-vs-staged sweep runs."""

from __future__ import annotations

import os
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import plotly.graph_objects as go
import streamlit as st

import cache_store

REPO_ROOT = Path(__file__).resolve().parent


def resolve_results_root() -> Path:
    """Results dir: $COMMBENCH_RESULTS_DIR, then .storage-config, then ./results."""
    env_override = os.environ.get("COMMBENCH_RESULTS_DIR")
    if env_override:
        return Path(env_override).expanduser()
    config_file = REPO_ROOT / ".storage-config"
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                return Path(line).expanduser()
    return REPO_ROOT / "results"


RESULTS_ROOT = resolve_results_root()
RESULT_FILE_GLOB = "commbench_sweep_*.txt"
SECTION_HEADER = re.compile(r"^---- size\s+(\d+) bytes : (P2P \(IPC, Level-Zero\)|P2P PULL \(IPC, Level-Zero\)|P2P \(raw Level-Zero, dedicated copy engine\)|staged, no pipeline \(GPU->host->GPU, P2P disabled\)|staged pipelined \(GPU->host->GPU, P2P disabled\)) ----")
STAT_LINE = re.compile(r"^(minTime|medTime|avgTime):\s+([\d.eE+-]+) us,\s+([\d.eE+-]+) GB/s")
LEG_LABEL = {
    "P2P (IPC, Level-Zero)": "p2p_sycl",
    "P2P PULL (IPC, Level-Zero)": "p2p_pull",
    "P2P (raw Level-Zero, dedicated copy engine)": "p2p",
    "staged pipelined (GPU->host->GPU, P2P disabled)": "no_p2p",
    "staged, no pipeline (GPU->host->GPU, P2P disabled)": "no_p2p_no_pipeline",
}
DEFAULT_TRACES = ["p2p"]  # raw Level-Zero P2P -- the real/best number; others are opt-in context
# Okabe-Ito palette: distinguishable under deuteranopia/protanopia/tritanopia.
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"]
DASH_STYLES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]
MARKER_SYMBOLS = ["circle", "square", "diamond", "triangle-up", "x", "star", "triangle-down", "pentagon"]


def read_environment(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        for line in path.read_text(errors="replace").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                values[key] = value
    except OSError:
        pass
    return values


def parse_result(path: Path) -> list[dict]:
    rows = []
    size = None
    leg = None
    try:
        content = path.read_text(errors="replace")
    except OSError:
        return rows
    for line in content.splitlines():
        line = line.strip()
        header_match = SECTION_HEADER.match(line)
        if header_match:
            size = float(header_match.group(1))
            leg = LEG_LABEL[header_match.group(2)]
            continue
        stat_match = STAT_LINE.match(line)
        if stat_match and size is not None:
            stat, time_us, gbps = stat_match.groups()
            rows.append({
                "size": size,
                "leg": leg,
                "stat": stat.replace("Time", "").lower(),  # min/med/avg
                "latency_us": float(time_us),
                "bw_gbps": float(gbps),
            })
    return rows


@st.cache_data(show_spinner=False)
def discover_runs() -> list[dict]:
    runs = []
    if not RESULTS_ROOT.exists():
        return runs
    for result_file in sorted(RESULTS_ROOT.glob(f"*/{RESULT_FILE_GLOB}")):
        rows = parse_result(result_file)
        if not rows:
            continue
        run_dir = result_file.parent
        environment = read_environment(run_dir / "environment.txt")
        config = environment.get("config_name") or "unlabelled"
        runs.append({
            "id": run_dir.name,
            "config": config,
            "rows": rows,
            "environment": environment,
            "file": result_file.name,
        })
    return runs


@st.cache_data(show_spinner=False)
def discover_skipped_results() -> list[str]:
    skipped = []
    if not RESULTS_ROOT.exists():
        return skipped
    for result_file in sorted(RESULTS_ROOT.glob(f"*/{RESULT_FILE_GLOB}")):
        if not parse_result(result_file):
            skipped.append(str(result_file))
    return skipped


FIRST_SCAN_NOTICE = ("No cached data found yet. The first scan has to walk the results directory over the network "
    "share and can take a minute or more depending on how many runs are present.")
REFRESH_NOTICE = "Rescanning RESULTS_ROOT over the network share -- this can take a minute or more."


def get_runs(force_refresh: bool = False) -> tuple[list[dict], dict | None, bool]:
    """Load runs from cache or re-scan. Returns (runs, meta, was_cached)."""
    runs, meta = (None, None) if force_refresh else cache_store.load_runs(RESULTS_ROOT)
    if runs is not None:
        return runs, meta, True
    banner = st.empty()
    banner.warning(REFRESH_NOTICE if force_refresh else FIRST_SCAN_NOTICE)
    t0 = time.time()
    if force_refresh:
        discover_runs.clear()  # bypass st.cache_data's own stale in-memory result too
    with st.spinner("Scanning RESULTS_ROOT..."):
        runs = discover_runs()
    meta = cache_store.make_meta(t0)
    cache_store.save_runs(RESULTS_ROOT, runs, meta)
    banner.empty()
    return runs, meta, False


def aggregate_runs(runs: list[dict], metric: str, stat: str) -> dict[str, dict[str, list[dict]]]:
    """configs -> leg -> sorted [{size, value}] medians across runs, for a given stat (min/med/avg)."""
    grouped: dict[str, dict[str, dict[float, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run in runs:
        for row in run["rows"]:
            if row["stat"] != stat:
                continue
            grouped[run["config"]][row["leg"]][row["size"]].append(row[metric])
    return {
        config: {
            leg: [{"size": size, "value": median(values)} for size, values in sorted(by_size.items())]
            for leg, by_size in by_leg.items()
        }
        for config, by_leg in grouped.items()
    }


def chart(title: str, metric: str, stat: str, unit: str, y_log: bool, runs: list[dict], aggregated: dict[str, dict[str, list[dict]]], traces: set[str]) -> go.Figure:
    figure = go.Figure()
    series_keys = sorted({f"{config} ({leg})" for config in aggregated for leg in aggregated[config]})
    styles = {
        key: {
            "color": PALETTE[index % len(PALETTE)],
            "dash": DASH_STYLES[index % len(DASH_STYLES)],
            "symbol": MARKER_SYMBOLS[index % len(MARKER_SYMBOLS)],
        }
        for index, key in enumerate(series_keys)
    }
    for run in runs:
        by_leg: dict[str, list[dict]] = defaultdict(list)
        for row in run["rows"]:
            if row["stat"] == stat and row["leg"] in traces:
                by_leg[row["leg"]].append({"size": row["size"], "value": row[metric]})
        for leg, rows in by_leg.items():
            rows = sorted(rows, key=lambda r: r["size"])
            style = styles[f"{run['config']} ({leg})"]
            figure.add_trace(go.Scatter(
                x=[row["size"] for row in rows], y=[row["value"] for row in rows],
                mode="lines", line={"color": style["color"], "width": 1, "dash": "dot"}, opacity=0.28,
                hovertemplate=f"{run['config']} {leg} | {run['id']}<br>%{{x:,.0f}} B<br>%{{y:.3f}} {unit}<extra></extra>",
                showlegend=False,
            ))
    for config, by_leg in aggregated.items():
        for leg, rows in by_leg.items():
            if leg not in traces:
                continue
            key = f"{config} ({leg})"
            style = styles[key]
            figure.add_trace(go.Scatter(
                x=[row["size"] for row in rows], y=[row["value"] for row in rows], mode="lines+markers",
                name=f"{key} median",
                line={"color": style["color"], "width": 3, "dash": style["dash"]},
                marker={"size": 8, "symbol": style["symbol"], "line": {"width": 1, "color": "white"}},
                hovertemplate=f"{key} median<br>%{{x:,.0f}} B<br>%{{y:.3f}} {unit}<extra></extra>",
            ))
    figure.update_layout(title=title, height=430, margin={"l": 8, "r": 8, "t": 46, "b": 92},
                         legend={"orientation": "h", "y": -0.24, "yanchor": "top"}, template="plotly_white")
    figure.update_xaxes(title="Message size (bytes)", type="log", tickformat="~s")
    figure.update_yaxes(title=f"{unit}", type="log" if y_log else "linear", rangemode="tozero")
    return figure


def metric_at(rows: list[dict], size: float) -> float | None:
    return next((row["value"] for row in rows if row["size"] == size), None)


def main() -> None:
    st.set_page_config(page_title="CommBench PCIe P2P Results", page_icon="C", layout="wide")
    st.title("CommBench P2P (IPC) vs Staged (P2P-disabled) Results")

    force_refresh = st.sidebar.button("Refresh data", help="Re-scan RESULTS_ROOT instead of using the cached scan")
    all_runs, meta, was_cached = get_runs(force_refresh)

    if meta:
        age = f"{time.time() - meta['generated_at']:.0f}s ago" if was_cached else "just now"
        st.sidebar.caption(f"Scan cached {age} (took {meta['duration_seconds']}s)")

    skipped_results = discover_skipped_results()
    if force_refresh:
        discover_skipped_results.clear()
        skipped_results = discover_skipped_results()
    if not all_runs:
        st.info("No parseable CommBench result files found under results/.")
        return

    if skipped_results:
        st.warning("Skipped result files with no parseable CommBench data: " + ", ".join(skipped_results))

    configs = sorted({run["config"] for run in all_runs})
    selected_configs = st.sidebar.multiselect("Configurations", configs, default=configs)
    stat = st.sidebar.radio("Statistic", ["min", "med", "avg"], index=0, help="min/med/avg across iterations")
    selected_runs = [run for run in all_runs if run["config"] in selected_configs]
    if not selected_runs:
        st.warning("Select at least one configuration.")
        return

    available_legs = sorted({row["leg"] for run in selected_runs for row in run["rows"]})
    default_legs = [leg for leg in DEFAULT_TRACES if leg in available_legs] or available_legs
    # Key includes the option set so a changed/renamed trace list resets the widget to the
    # new default instead of silently carrying over a stale selection from an older session.
    traces_key = "traces_" + ",".join(available_legs)
    selected_legs = set(st.sidebar.multiselect("Traces", available_legs, default=default_legs, key=traces_key,
                                                help="p2p is the real/best P2P number; the rest are optional context"))
    if not selected_legs:
        st.warning("Select at least one trace.")
        return

    bw_aggregated = aggregate_runs(selected_runs, "bw_gbps", stat)
    latency_aggregated = aggregate_runs(selected_runs, "latency_us", stat)
    st.caption(f"{len(selected_runs)} run(s) across {len(configs)} configuration(s), stat={stat}. "
               "Thin lines are individual runs; solid lines are per-size medians across runs. "
               "'p2p' = genuine GPU<->GPU P2P (IPC), issued via raw Level-Zero on the GPU's "
               "dedicated copy engine, bypassing SYCL's abstraction tax -- this is the real/best P2P "
               "number, shown by default; "
               "'p2p_sycl' = same P2P transfer via plain SYCL queue.memcpy() -- capped lower by a "
               "software artifact, kept as a before/after reference; "
               "'p2p_pull' = receiver pulls instead of sender pushing -- broken on this driver, always "
               "skips timing; "
               "'no_p2p' = GPU->host->GPU baseline with P2P disabled, chunked and pipelined (raw "
               "Level-Zero) so neighboring chunks' device<->host copies and MPI hop overlap -- the "
               "best-effort 'P2P disabled' ceiling; "
               "'no_p2p_no_pipeline' = same baseline but as one single full-buffer copy/send/copy with "
               "no overlap -- naive worst-case reference.")
    left, right = st.columns(2)
    with left:
        st.plotly_chart(chart("P2P bandwidth", "bw_gbps", stat, "GB/s", False, selected_runs, bw_aggregated, selected_legs), use_container_width=True)
    with right:
        st.plotly_chart(chart("P2P latency", "latency_us", stat, "us", True, selected_runs, latency_aggregated, selected_legs), use_container_width=True)

    summary = []
    for config in sorted(bw_aggregated):
        matching_runs = [run for run in selected_runs if run["config"] == config]
        environment = matching_runs[0]["environment"]
        p2p_bw = bw_aggregated[config].get("p2p", [])
        p2p_sycl_bw = bw_aggregated[config].get("p2p_sycl", [])
        p2p_pull_bw = bw_aggregated[config].get("p2p_pull", [])
        no_p2p_bw = bw_aggregated[config].get("no_p2p", [])
        no_p2p_no_pipeline_bw = bw_aggregated[config].get("no_p2p_no_pipeline", [])
        p2p_latency = latency_aggregated.get(config, {}).get("p2p", [])
        summary.append({
            "Configuration": config,
            "Runs": len(matching_runs),
            "4 KB latency p2p (us)": metric_at(p2p_latency, 4096),
            "256 MB BW p2p -- best (GB/s)": metric_at(p2p_bw, 256 * 1024 * 1024),
            "256 MB BW p2p_sycl (GB/s)": metric_at(p2p_sycl_bw, 256 * 1024 * 1024),
            "256 MB BW p2p_pull (GB/s)": metric_at(p2p_pull_bw, 256 * 1024 * 1024),
            "256 MB BW no_p2p (GB/s)": metric_at(no_p2p_bw, 256 * 1024 * 1024),
            "256 MB BW no_p2p_no_pipeline (GB/s)": metric_at(no_p2p_no_pipeline_bw, 256 * 1024 * 1024),
        })
    st.subheader(f"{stat.title()} Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    with st.expander("Captured run metadata"):
        for run in selected_runs:
            st.markdown(f"**{run['id']}** ({run['file']})")
            st.code("\n".join(f"{key}={value}" for key, value in run["environment"].items()) or "No environment metadata found.")


if __name__ == "__main__":
    main()
