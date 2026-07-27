"""Interactive Streamlit viewer for GRPO conductor traces.

Run from the repository root with::

    streamlit run trace_viewer.py
"""

from __future__ import annotations

import json
import sys
from html import escape
from collections import Counter
from pathlib import Path
from typing import Any

import altair as alt
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from theo_conductor.trace_analysis import (
    PLAN_METRIC_NAMES,
    TraceDataset,
    TraceQuery,
    TraceRecord,
    error_category,
    workflow_to_graphviz,
)
from theo_conductor.benchmark import oracle_routing_breakdown


ROOT = Path(__file__).resolve().parent
DEFAULT_MEGASCIENCE_DIR = ROOT / "outputs/megascience-small-models"
TRACE_FILENAME = "plans-and-worker-outputs-rank-0.jsonl"
PAGE_SIZE = 80
MEMORY_CHART_MAX_ROWS = 4_000
GPU_ROLES = {
    0: "DeepSeek TP0",
    1: "DeepSeek TP1",
    2: "Gemma TP0",
    3: "Gemma TP1",
    4: "Qwen Coder TP0",
    5: "Qwen Coder TP1",
    6: "Conductor primary",
    7: "Conductor secondary",
}
REWARD_COLORS = {0.0: "#c94848", 0.2: "#e87817", 0.5: "#f2c94c", 1.0: "#318260"}
DIFFICULTY_COLORS = {"easy": "#318260", "medium": "#f2c94c", "hard": "#c94848"}
ERROR_STYLES = (
    ("🔴", "#e63946"),
    ("🟠", "#f77f00"),
    ("🟡", "#e9c46a"),
    ("🟢", "#43aa8b"),
    ("💚", "#2a9d8f"),
    ("🔵", "#277da1"),
    ("🟦", "#6c5ce7"),
    ("🟣", "#9b5de5"),
    ("🩷", "#f15bb5"),
    ("🟤", "#8d6e63"),
    ("⚫", "#577590"),
    ("🩵", "#00b4d8"),
)


st.set_page_config(page_title="Theo trace viewer", page_icon="◈", layout="wide")


def reward_label(value: Any) -> str:
    try:
        reward = float(value)
    except (TypeError, ValueError):
        return str(value)
    return {
        0.0: "0.0 · malformed",
        0.2: "0.2 · invalid workflow",
        0.5: "0.5 · valid plan",
        1.0: "1.0 · correct",
    }.get(reward, str(value))


def reward_key(value: Any) -> str:
    try:
        return {0.0: "r0", 0.2: "r02", 0.5: "r05", 1.0: "r1"}.get(float(value), "other")
    except (TypeError, ValueError):
        return "other"


def reward_icon(value: Any) -> str:
    try:
        return {0.0: "🔴", 0.2: "🟠", 0.5: "🟡", 1.0: "🟢"}.get(float(value), "⚪")
    except (TypeError, ValueError):
        return "⚪"


def error_style_map(records: list[TraceRecord]) -> dict[str, tuple[str, str]]:
    categories = sorted({record.error_category for record in records if record.data.get("error")})
    return {category: ERROR_STYLES[index % len(ERROR_STYLES)] for index, category in enumerate(categories)}


@st.cache_data(show_spinner=False)
def load_path(path_text: str, modified_ns: int) -> TraceDataset:
    del modified_ns  # Included in the cache key so changed traces are reloaded.
    return TraceDataset.load(Path(path_text))


@st.cache_data(show_spinner=False)
def load_upload(raw: bytes, name: str) -> TraceDataset:
    records: list[TraceRecord] = []
    malformed: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if not isinstance(data, dict):
                raise ValueError("record is not a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            malformed.append({"source": name, "line": line_number, "error": str(exc)})
            continue
        records.append(
            TraceRecord(
                data=data,
                record_id=f"0:{line_number}",
                source=name,
                line=line_number,
                error_category=error_category(data.get("error")),
            )
        )
    if not records:
        raise ValueError("No JSON records were found.")
    return TraceDataset(records, malformed_lines=malformed)


@st.cache_data(show_spinner=False)
def load_megascience(summary_path: str, results_path: str, modified_ns: tuple[int, int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load the benchmark aggregate and per-question records."""
    del modified_ns
    with Path(summary_path).open(encoding="utf-8") as handle:
        summary = json.load(handle)
    records: list[dict[str, Any]] = []
    with Path(results_path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid benchmark record on line {line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Benchmark record on line {line_number} is not a JSON object.")
            records.append(record)
    return summary, records


@st.cache_data(show_spinner=False)
def load_memory_telemetry(
    gpu_path: str,
    process_path: str | None,
    modified_ns: tuple[int, int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize whole-GPU and per-process memory telemetry."""
    del modified_ns
    gpu = pd.read_csv(gpu_path, skipinitialspace=True)
    gpu.columns = [str(column).strip() for column in gpu.columns]
    required_gpu = {
        "timestamp",
        "gpu_index",
        "gpu_uuid",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
    }
    missing_gpu = required_gpu - set(gpu.columns)
    if missing_gpu:
        raise ValueError(f"gpu-memory.csv is missing columns: {', '.join(sorted(missing_gpu))}")
    gpu["timestamp"] = pd.to_datetime(gpu["timestamp"], utc=True, errors="coerce")
    for column in (
        "gpu_index",
        "memory_used_mib",
        "memory_total_mib",
        "utilization_gpu_percent",
    ):
        gpu[column] = pd.to_numeric(gpu[column], errors="coerce")
    gpu["gpu_uuid"] = gpu["gpu_uuid"].astype(str).str.strip()
    gpu = gpu.dropna(subset=["timestamp", "gpu_index", "memory_used_mib"]).copy()
    gpu["gpu_index"] = gpu["gpu_index"].astype(int)
    start = gpu["timestamp"].min()
    gpu["elapsed_minutes"] = (gpu["timestamp"] - start).dt.total_seconds() / 60
    gpu["used_gib"] = gpu["memory_used_mib"] / 1024
    gpu["total_gib"] = gpu["memory_total_mib"] / 1024
    gpu["gpu"] = gpu["gpu_index"].map(
        lambda index: f"GPU {index} · {GPU_ROLES.get(index, 'unassigned')}"
    )

    if process_path is None:
        return gpu, pd.DataFrame()
    process = pd.read_csv(process_path, skipinitialspace=True)
    process.columns = [str(column).strip() for column in process.columns]
    required_process = {
        "timestamp",
        "gpu_uuid",
        "pid",
        "process_name",
        "used_memory_mib",
    }
    missing_process = required_process - set(process.columns)
    if missing_process:
        raise ValueError(
            f"{Path(process_path).name} is missing columns: {', '.join(sorted(missing_process))}"
        )
    process["timestamp"] = pd.to_datetime(process["timestamp"], utc=True, errors="coerce")
    process["pid"] = pd.to_numeric(process["pid"], errors="coerce")
    process["used_memory_mib"] = pd.to_numeric(process["used_memory_mib"], errors="coerce")
    process["gpu_uuid"] = process["gpu_uuid"].astype(str).str.strip()
    process["process_name"] = process["process_name"].astype(str).str.strip()
    process = process.dropna(subset=["timestamp", "pid", "used_memory_mib"]).copy()
    uuid_to_gpu = (
        gpu[["gpu_uuid", "gpu_index"]]
        .drop_duplicates("gpu_uuid")
        .set_index("gpu_uuid")["gpu_index"]
    )
    process["gpu_index"] = process["gpu_uuid"].map(uuid_to_gpu)
    process = process.dropna(subset=["gpu_index"]).copy()
    process["gpu_index"] = process["gpu_index"].astype(int)
    process["pid"] = process["pid"].astype(int)
    process["elapsed_minutes"] = (process["timestamp"] - start).dt.total_seconds() / 60
    process["used_gib"] = process["used_memory_mib"] / 1024
    process["process"] = process.apply(
        lambda row: (
            f"GPU {row['gpu_index']} · {row['process_name']} · PID {row['pid']}"
        ),
        axis=1,
    )
    return gpu, process


def downsample_peaks(
    frame: pd.DataFrame,
    *,
    series_column: str,
    value_column: str,
    max_rows: int = MEMORY_CHART_MAX_ROWS,
) -> pd.DataFrame:
    """Bound chart size while retaining the peak sample in each time bucket."""
    if len(frame) <= max_rows or frame.empty:
        return frame
    ordered = frame.sort_values([series_column, "timestamp"]).copy()
    series_count = max(int(ordered[series_column].nunique()), 1)
    points_per_series = max(max_rows // series_count, 2)
    positions = ordered.groupby(series_column, sort=False).cumcount()
    sizes = ordered.groupby(series_column, sort=False)[series_column].transform("size")
    strides = ((sizes + points_per_series - 1) // points_per_series).clip(lower=1)
    ordered["_bucket"] = positions // strides
    peak_indices = ordered.groupby(
        [series_column, "_bucket"], sort=False
    )[value_column].idxmax()
    return ordered.loc[peak_indices].drop(columns="_bucket").sort_values("timestamp")


def gpu_pressure_statistics(
    gpu: pd.DataFrame,
) -> tuple[dict[str, float | str], pd.DataFrame]:
    """Summarize utilization duty cycle, memory pressure, and device imbalance."""
    valid = gpu.dropna(
        subset=[
            "timestamp",
            "gpu_index",
            "used_gib",
            "total_gib",
            "utilization_gpu_percent",
        ]
    ).copy()
    valid = valid[valid["total_gib"] > 0]
    if valid.empty:
        return {}, pd.DataFrame()

    valid["memory_fraction"] = valid["used_gib"] / valid["total_gib"]
    utilization_by_time = valid.pivot_table(
        index="timestamp",
        columns="gpu_index",
        values="utilization_gpu_percent",
        aggfunc="mean",
    )
    active_times = utilization_by_time.max(axis=1) > 10
    active_timestamps = set(utilization_by_time.index[active_times])
    active = valid[valid["timestamp"].isin(active_timestamps)]
    if active.empty:
        active = valid
    active_spread = utilization_by_time.loc[active_times].max(axis=1) - utilization_by_time.loc[
        active_times
    ].min(axis=1)
    if active_spread.empty:
        active_spread = utilization_by_time.max(axis=1) - utilization_by_time.min(axis=1)

    active_mean = float(active["utilization_gpu_percent"].mean())
    active_busy_share = float((active["utilization_gpu_percent"] >= 90).mean())
    overall_idle_share = float((valid["utilization_gpu_percent"] <= 10).mean())
    peak_memory_fraction = float(valid["memory_fraction"].max())
    memory_pressure_share = float((valid["memory_fraction"] >= 0.9).mean())
    p95_spread = float(active_spread.quantile(0.95))

    if memory_pressure_share >= 0.2 or peak_memory_fraction >= 0.98:
        if active_mean >= 70 and active_busy_share >= 0.4:
            assessment = (
                "High compute saturation and high memory pressure: the run may be constrained by "
                "both accelerator throughput and memory capacity."
            )
        else:
            assessment = (
                "Memory-capacity pressure is stronger than the compute-saturation signal."
            )
    elif active_mean >= 70 and active_busy_share >= 0.4:
        assessment = (
            "Strong compute-bound signal: GPUs remain highly utilized during active windows "
            "without sustained near-capacity memory use."
        )
    elif overall_idle_share >= 0.4 or active_mean < 40:
        assessment = (
            "Weak compute-bound signal: substantial low-utilization time suggests input, CPU, "
            "communication, synchronization, or scheduling stalls."
        )
    elif p95_spread >= 50:
        assessment = (
            "Uneven compute utilization: one or more GPUs frequently wait while others are busy."
        )
    else:
        assessment = (
            "Mixed utilization signal: this telemetry alone does not identify a dominant bottleneck."
        )

    per_gpu = (
        valid.groupby(["gpu_index", "gpu"], as_index=False)
        .agg(
            mean_utilization=("utilization_gpu_percent", "mean"),
            p95_utilization=("utilization_gpu_percent", lambda values: values.quantile(0.95)),
            busy_share=("utilization_gpu_percent", lambda values: (values >= 90).mean()),
            idle_share=("utilization_gpu_percent", lambda values: (values <= 10).mean()),
            peak_used_gib=("used_gib", "max"),
            capacity_gib=("total_gib", "max"),
            peak_memory_fraction=("memory_fraction", "max"),
        )
        .sort_values("gpu_index")
    )
    return (
        {
            "mean_utilization": float(valid["utilization_gpu_percent"].mean()),
            "active_mean_utilization": active_mean,
            "p95_utilization": float(valid["utilization_gpu_percent"].quantile(0.95)),
            "active_busy_share": active_busy_share,
            "overall_idle_share": overall_idle_share,
            "peak_memory_fraction": peak_memory_fraction,
            "memory_pressure_share": memory_pressure_share,
            "p95_utilization_spread": p95_spread,
            "minimum_headroom_gib": float((valid["total_gib"] - valid["used_gib"]).min()),
            "assessment": assessment,
        },
        per_gpu,
    )


def selected_dataset() -> tuple[TraceDataset, str]:
    traces: dict[int, Path] = {}
    for output_dir in (ROOT / "outputs").glob("grpo-*"):
        job_id = output_dir.name.removeprefix("grpo-")
        path = output_dir / "traces" / TRACE_FILENAME
        if job_id.isdigit() and path.is_file():
            traces[int(job_id)] = path

    if not traces:
        st.info(f"No SLURM traces found at outputs/grpo-<SLURM ID>/traces/{TRACE_FILENAME}.")
        st.stop()

    job_ids = sorted(traces, reverse=True)
    latest_job_id = job_ids[0]
    job_id = st.sidebar.selectbox(
        "SLURM ID",
        job_ids,
        format_func=lambda value: f"{value} (latest)" if value == latest_job_id else str(value),
    )
    path = traces[job_id]
    return load_path(str(path), path.stat().st_mtime_ns), str(path.relative_to(ROOT))


def pie_chart(
    counts: Counter[str], denominator: int, *, category_colors: dict[str, str] | None = None
) -> alt.Chart:
    rows = [
        {"name": name, "count": count, "percent": count / denominator}
        for name, count in counts.most_common()
    ]
    color_encoding: alt.Color = alt.Color("name:N", title=None)
    if category_colors:
        domain = [name for name, _ in counts.most_common()]
        fallback = ("#697b8c", "#8b5fbf", "#287396")
        color_encoding = alt.Color(
            "name:N",
            title=None,
            scale=alt.Scale(
                domain=domain,
                range=[category_colors.get(name.casefold(), fallback[index % len(fallback)]) for index, name in enumerate(domain)],
            ),
        )
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_arc(innerRadius=42)
        .encode(
            theta=alt.Theta("count:Q"),
            color=color_encoding,
            tooltip=[alt.Tooltip("name:N"), alt.Tooltip("count:Q"), alt.Tooltip("percent:Q", format=".1%")],
        )
        .properties(height=230)
    )


def render_overview(dataset: TraceDataset, error_styles: dict[str, tuple[str, str]]) -> None:
    summary = dataset.summary()
    records = dataset.records
    token_info = summary["completion_tokens"]
    values = (
        (f'{summary["records"]:,}', "Trace records"),
        (f'{summary["mean_reward"]:.3f}' if summary["mean_reward"] is not None else "—", "Mean reward"),
        (f'{summary["parsed_plans"]:,}', "Parsed plans"),
        (f'{summary["worker_runs"]:,}', "Worker runs"),
        (f'{summary["unique_questions"]:,}', "Unique questions"),
    )
    columns = st.columns(5)
    for column, (value, label) in zip(columns, values, strict=True):
        column.metric(label, value)

    st.subheader("Conductor plan statistics")
    plans = [record.data["plan"] for record in records if isinstance(record.data.get("plan"), dict)]
    steps = [step for plan in plans for step in plan.get("workflow", []) if isinstance(step, dict)]
    multi_step = sum(len(plan.get("workflow", [])) > 1 for plan in plans)
    models = Counter(
        str(step["model_id"] if step.get("model_id") is not None else "(missing)")
        for step in steps
    )
    plan_values = (
        (f"{len(steps) / len(plans):.2f}" if plans else "—", "Mean planned steps"),
        (f"{multi_step / len(plans):.1%}" if plans else "—", "Multi-step plans"),
        (f'{token_info["mean"]:,.1f}' if token_info else "—", "Average conductor tokens thus far"),
    )
    columns = st.columns(3)
    for column, (value, label) in zip(columns, plan_values, strict=True):
        column.metric(label, value)

    metric_labels = {
        "num_steps": "num_steps",
        "num_edges": "num_edges",
        "critical_path_steps": "critical_path_steps",
        "critical_path_runtime": "critical_path_runtime (s)",
        "total_worker_runtime": "total_worker_runtime (s)",
        "maximum_width": "maximum_width",
        "work_span_parallelism": "work_span_parallelism",
        "observed_wall_time": "observed_wall_time (s)",
        "observed_peak_concurrency": "observed_peak_concurrency",
        "realized_parallelism": "realized_parallelism",
        "parallelism_utilization": "parallelism_utilization",
    }
    plan_metrics = summary.get("plan_metrics") or []
    plan_statistics = summary.get("plan_statistics") or {}
    statistics_rows = [
        {
            "Metric": metric_labels[name],
            "Mean": plan_statistics.get(name),
            "Plans with data": sum(row.get(name) is not None for row in plan_metrics),
        }
        for name in PLAN_METRIC_NAMES
    ]
    st.dataframe(
        pd.DataFrame(statistics_rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Mean": st.column_config.NumberColumn(format="%.3f"),
            "Plans with data": st.column_config.NumberColumn(format="%d"),
        },
    )
    st.caption(
        "Critical-path and total-worker runtimes require latency for every planned step. "
        "Observed metrics are populated by newly recorded runner traces; legacy traces remain blank."
    )

    left, right = st.columns(2)
    difficulties = Counter(str(plan.get("difficulty") or "(missing)") for plan in plans)
    with left:
        st.markdown("**Difficulty**")
        if difficulties:
            st.altair_chart(
                pie_chart(difficulties, len(plans), category_colors=DIFFICULTY_COLORS),
                width="stretch",
            )
        else:
            st.caption("No parsed plans.")
    with right:
        st.markdown("**Worker model assignments**")
        if models:
            st.altair_chart(pie_chart(models, len(steps)), width="stretch")
        else:
            st.caption("No planned worker calls.")
    st.caption("Model assignments count planned workflow steps, not worker executions.")

    st.markdown("### LLM performance")
    conductor_rows = [
        {**row, "role": "conductor", "latency_basis": row.get("latency_basis", "generation batch")}
        for row in summary.get("conductor_performance", [])
    ]
    judge_rows = [
        {**row, "role": "judge", "latency_basis": row.get("latency_basis", "request")}
        for row in summary.get("judge_performance", [])
    ]
    worker_rows = [
        {**row, "role": "worker", "latency_basis": "request"}
        for row in summary["worker_performance"]
    ]
    performance_rows = conductor_rows + judge_rows + worker_rows
    performance = pd.DataFrame(
        [
            {
                "LLM": row["model_id"],
                "Role": row["role"],
                "Runs": row["runs"],
                "Latency basis": row["latency_basis"],
                "Mean latency (s)": (
                    row["mean_latency_ms"] / 1000 if row["mean_latency_ms"] is not None else None
                ),
                "P95 latency (s)": (
                    row["p95_latency_ms"] / 1000 if row["p95_latency_ms"] is not None else None
                ),
                "Mean prompt tokens": row["mean_prompt_tokens"],
                "Mean output tokens": row["mean_completion_tokens"],
                "Mean total tokens": row["mean_total_tokens"],
                "Output tokens/s": row["mean_output_tokens_per_second"],
            }
            for row in performance_rows
        ]
    )
    if performance.empty:
        st.caption("No conductor or worker performance data is available.")
    else:
        st.dataframe(
            performance,
            width="stretch",
            hide_index=True,
            column_config={
                "Runs": st.column_config.NumberColumn(format="%d"),
                "Mean latency (s)": st.column_config.NumberColumn(format="%.2f"),
                "P95 latency (s)": st.column_config.NumberColumn(format="%.2f"),
                "Mean prompt tokens": st.column_config.NumberColumn(format="%.1f"),
                "Mean output tokens": st.column_config.NumberColumn(format="%.1f"),
                "Mean total tokens": st.column_config.NumberColumn(format="%.1f"),
                "Output tokens/s": st.column_config.NumberColumn(format="%.1f"),
            },
        )
        st.caption(
            "Worker throughput is the mean per-request completion-token rate. Conductor throughput is "
            "the mean batched generation rate and its latency is measured once per generation batch. "
            "Judge throughput is measured per successful request."
        )
    availability_notes = []
    if not any(row.get("mean_output_tokens_per_second") is not None for row in conductor_rows):
        availability_notes.append(
            "Conductor tok/s is not available for this trace because conductor generation latency was not recorded."
        )
    if not judge_rows:
        availability_notes.append(
            "Kimi K2.6 judging performance is not available for this trace."
        )
    elif not any(row.get("mean_output_tokens_per_second") is not None for row in judge_rows):
        availability_notes.append(
            "Kimi K2.6 judge tok/s is not available because token usage or request latency was not recorded."
        )
    if availability_notes:
        st.info(" ".join(availability_notes))

    st.subheader("Runtime bottlenecks")
    timing = summary.get("worker_timing")
    if not timing:
        st.caption("No worker latency data is available.")
    else:
        interval_share = timing.get("workflow_interval_share")
        headline = st.columns(4)
        metrics = (
            (
                f"{float(timing['estimated_workflow_latency_ms']) / 3_600_000:.2f} h",
                "Estimated workflow time",
            ),
            (
                f"{float(interval_share):.1%}" if interval_share is not None else "—",
                "Share of observed batch time",
            ),
            (
                f"{float(timing['total_call_latency_ms']) / 3_600_000:.2f} h",
                "Total worker call-seconds",
            ),
            (
                f"{float(timing['parallelism_savings_ms']) / 3_600_000:.2f} h",
                "In-workflow parallelism saved",
            ),
        )
        for column, (value, label) in zip(headline, metrics, strict=True):
            column.metric(label, value)

        model_timing = pd.DataFrame(
            [
                {
                    "Worker LLM": row["model_id"],
                    "Critical-path hours": row["critical_path_latency_ms"] / 3_600_000,
                    "Critical-path share": row["critical_path_share"],
                    "Total call hours": row["total_call_latency_ms"] / 3_600_000,
                    "Calls": row["calls"],
                    "Bottleneck layers": row["bottleneck_layers"],
                }
                for row in timing["models"]
            ]
        )
        if not model_timing.empty:
            bars = (
                alt.Chart(model_timing)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Critical-path hours:Q", title="Estimated critical-path hours"),
                    y=alt.Y("Worker LLM:N", title=None, sort="-x"),
                    color=alt.Color("Worker LLM:N", legend=None),
                    tooltip=[
                        "Worker LLM:N",
                        alt.Tooltip("Critical-path hours:Q", format=".2f"),
                        alt.Tooltip("Critical-path share:Q", format=".1%"),
                        alt.Tooltip("Calls:Q", format=",d"),
                        alt.Tooltip("Bottleneck layers:Q", format=",d"),
                    ],
                )
                .properties(height=max(120, 42 * len(model_timing)))
            )
            table_column, chart_column = st.columns((1.15, 1))
            with table_column:
                st.dataframe(
                    model_timing,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Critical-path hours": st.column_config.NumberColumn(format="%.2f"),
                        "Critical-path share": st.column_config.ProgressColumn(
                            format="percent", min_value=0, max_value=1
                        ),
                        "Total call hours": st.column_config.NumberColumn(format="%.2f"),
                        "Calls": st.column_config.NumberColumn(format="%d"),
                        "Bottleneck layers": st.column_config.NumberColumn(format="%d"),
                    },
                )
            with chart_column:
                st.altair_chart(bars, width="stretch")
        observed = int(timing.get("observed_batch_intervals") or 0)
        st.caption(
            "Workflow time is estimated as the slowest call in each concurrent execution layer. "
            "Total call-seconds sum all requests and can exceed wall time. "
            + (
                f"The observed-time share compares this estimate with {observed} intervals between "
                "successive batch trace writes; the remainder includes conductor generation, judging, "
                "training, and logging."
                if observed
                else "Observed-time share requires timestamps from at least two completed batches."
            )
        )

    st.subheader("Reward outcomes")
    reward_counts = Counter(float(record.data.get("reward", 0)) for record in records)
    reward_rows = [
        {
            "reward": reward_label(reward),
            "count": reward_counts.get(reward, 0),
            "fraction": reward_counts.get(reward, 0) / len(records),
        }
        for reward in sorted(set(REWARD_COLORS) | set(reward_counts))
    ]
    domain = [reward_label(value) for value in sorted(REWARD_COLORS)]
    colors = [REWARD_COLORS[value] for value in sorted(REWARD_COLORS)]
    reward_chart = (
        alt.Chart(pd.DataFrame(reward_rows))
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            x=alt.X("count:Q", title="Records"),
            y=alt.Y("reward:N", title=None, sort=domain),
            color=alt.Color("reward:N", scale=alt.Scale(domain=domain, range=colors), legend=None),
            tooltip=["reward:N", "count:Q", alt.Tooltip("fraction:Q", format=".1%")],
        )
        .properties(height=180)
    )
    st.altair_chart(reward_chart, width="stretch")

    failures_tab, batches_tab = st.tabs(("Failure reasons", "Reward mix by batch"))
    failures = Counter(
        record.error_category
        for record in records
        if record.data.get("error") and float(record.data.get("reward", 0)) in (0.0, 0.2)
    )
    with failures_tab:
        if failures:
            failure_rows = [
                {
                    "reason": reason,
                    "label": f"{error_styles[reason][0]}  {reason}",
                    "count": count,
                }
                for reason, count in failures.most_common()
            ]
            failure_data = pd.DataFrame(failure_rows)
            failure_bars = (
                alt.Chart(failure_data)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("count:Q", title="Records", axis=alt.Axis(tickMinStep=1)),
                    y=alt.Y(
                        "label:N",
                        title=None,
                        sort="-x",
                        axis=alt.Axis(labelLimit=520),
                    ),
                    color=alt.Color(
                        "reason:N",
                        scale=alt.Scale(
                            domain=list(error_styles),
                            range=[style[1] for style in error_styles.values()],
                        ),
                        legend=None,
                    ),
                    tooltip=[alt.Tooltip("reason:N", title="Reason"), alt.Tooltip("count:Q", title="Records")],
                )
            )
            failure_labels = alt.Chart(failure_data).mark_text(
                align="left", baseline="middle", dx=5, color="#60707c"
            ).encode(
                x=alt.X("count:Q"),
                y=alt.Y("label:N", sort="-x"),
                text=alt.Text("count:Q"),
            )
            chart_height = min(420, max(150, len(failure_data) * 36))
            st.altair_chart((failure_bars + failure_labels).properties(height=chart_height), width="stretch")
        else:
            st.success("No 0.0 or 0.2 failures.")
    with batches_tab:
        st.caption("Each column shows the reward composition of one batch.")
        batch_rows = [
            {"batch": record.data.get("batch"), "reward": reward_label(record.data.get("reward")), "count": 1}
            for record in records
        ]
        if batch_rows:
            batch_chart = (
                alt.Chart(pd.DataFrame(batch_rows))
                .mark_bar()
                .encode(
                    x=alt.X("batch:O", title="Batch", axis=alt.Axis(labelOverlap=True)),
                    y=alt.Y("count:Q", aggregate="sum", stack="normalize", title="Reward composition"),
                    color=alt.Color("reward:N", scale=alt.Scale(domain=domain, range=colors), title="Reward"),
                    tooltip=["batch:O", "reward:N", alt.Tooltip("count:Q", aggregate="sum")],
                )
                .properties(height=190)
            )
            st.altair_chart(batch_chart, width="stretch")


def render_memory_telemetry(source_name: str) -> None:
    source_path = Path(source_name)
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    run_dir = source_path.parent.parent
    gpu_path = run_dir / "gpu-memory.csv"
    process_path = next(
        (
            candidate
            for candidate in (
                run_dir / "gpu-process-memory.csv",
                run_dir / "gpo-process-memory.csv",
                run_dir / "grpo-process-memory.csv",
            )
            if candidate.is_file()
        ),
        None,
    )
    if not gpu_path.is_file():
        return

    st.subheader("GPU memory telemetry")
    try:
        gpu, process = load_memory_telemetry(
            str(gpu_path),
            str(process_path) if process_path is not None else None,
            (
                gpu_path.stat().st_mtime_ns,
                process_path.stat().st_mtime_ns if process_path is not None else 0,
            ),
        )
    except (OSError, UnicodeError, ValueError, pd.errors.ParserError) as exc:
        st.warning(f"Could not load memory telemetry: {exc}")
        return
    if gpu.empty:
        st.caption("gpu-memory.csv contains no usable samples.")
        return

    duration_minutes = float(gpu["elapsed_minutes"].max())
    headline = st.columns(4)
    headline[0].metric("Telemetry duration", f"{duration_minutes / 60:.2f} h")
    headline[1].metric("GPUs observed", f"{gpu['gpu_index'].nunique():,}")
    headline[2].metric("Peak device memory", f"{gpu['used_gib'].max():.1f} GiB")
    headline[3].metric(
        "Peak GPU utilization",
        f"{gpu['utilization_gpu_percent'].max():.0f}%",
    )
    pressure, per_gpu = gpu_pressure_statistics(gpu)
    if pressure:
        diagnostic_values = (
            (
                f"{float(pressure['active_mean_utilization']):.1f}%",
                "Active-window mean utilization",
            ),
            (
                f"{float(pressure['p95_utilization']):.1f}%",
                "P95 GPU utilization",
            ),
            (
                f"{float(pressure['active_busy_share']):.1%}",
                "Active samples ≥90% utilized",
            ),
            (
                f"{float(pressure['overall_idle_share']):.1%}",
                "All samples ≤10% utilized",
            ),
            (
                f"{float(pressure['peak_memory_fraction']):.1%}",
                "Peak memory capacity used",
            ),
            (
                f"{float(pressure['memory_pressure_share']):.1%}",
                "Samples ≥90% memory",
            ),
            (
                f"{float(pressure['minimum_headroom_gib']):.1f} GiB",
                "Minimum memory headroom",
            ),
            (
                f"{float(pressure['p95_utilization_spread']):.1f} pp",
                "P95 cross-GPU utilization spread",
            ),
        )
        for offset in range(0, len(diagnostic_values), 4):
            columns = st.columns(4)
            for column, (value, label) in zip(
                columns,
                diagnostic_values[offset : offset + 4],
                strict=True,
            ):
                column.metric(label, value)
        st.info(f"Compute-bound assessment: {pressure['assessment']}")
        st.caption(
            "Active windows are timestamps where at least one GPU exceeds 10% utilization. "
            "This is a telemetry heuristic—not a profiler result; confirming the bottleneck "
            "requires SM occupancy/kernel timing plus CPU, I/O, and interconnect measurements."
        )

    gpu_tab, process_tab = st.tabs(
        ("gpu-memory.csv", process_path.name if process_path is not None else "process memory")
    )
    with gpu_tab:
        memory_data = downsample_peaks(
            gpu,
            series_column="gpu",
            value_column="used_gib",
        )
        memory_chart = (
            alt.Chart(memory_data)
            .mark_line()
            .encode(
                x=alt.X("elapsed_minutes:Q", title="Elapsed time (minutes)"),
                y=alt.Y("used_gib:Q", title="Memory used (GiB)", scale=alt.Scale(zero=True)),
                color=alt.Color("gpu:N", title="Device"),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("gpu:N", title="Device"),
                    alt.Tooltip("used_gib:Q", title="Used GiB", format=".2f"),
                    alt.Tooltip("total_gib:Q", title="Total GiB", format=".2f"),
                ],
            )
            .properties(height=350, title="Device memory usage")
            .interactive(bind_y=False)
        )
        st.altair_chart(memory_chart, width="stretch")

        utilization_data = downsample_peaks(
            gpu,
            series_column="gpu",
            value_column="utilization_gpu_percent",
        )
        utilization_chart = (
            alt.Chart(utilization_data)
            .mark_line()
            .encode(
                x=alt.X("elapsed_minutes:Q", title="Elapsed time (minutes)"),
                y=alt.Y(
                    "utilization_gpu_percent:Q",
                    title="GPU utilization (%)",
                    scale=alt.Scale(domain=[0, 110]),
                ),
                color=alt.Color("gpu:N", title="Device"),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("gpu:N", title="Device"),
                    alt.Tooltip(
                        "utilization_gpu_percent:Q",
                        title="Utilization",
                        format=".0f",
                    ),
                ],
            )
            .properties(height=260, title="GPU utilization")
            .interactive(bind_y=False)
        )
        st.altair_chart(utilization_chart, width="stretch")
        if not per_gpu.empty:
            per_gpu_display = per_gpu.rename(
                columns={
                    "gpu_index": "GPU",
                    "gpu": "Role",
                    "mean_utilization": "Mean utilization",
                    "p95_utilization": "P95 utilization",
                    "busy_share": "Samples ≥90%",
                    "idle_share": "Samples ≤10%",
                    "peak_used_gib": "Peak GiB",
                    "capacity_gib": "Capacity GiB",
                    "peak_memory_fraction": "Peak memory share",
                }
            )
            st.dataframe(
                per_gpu_display,
                width="stretch",
                hide_index=True,
                column_config={
                    "GPU": st.column_config.NumberColumn(format="%d"),
                    "Mean utilization": st.column_config.NumberColumn(format="%.1f%%"),
                    "P95 utilization": st.column_config.NumberColumn(format="%.1f%%"),
                    "Samples ≥90%": st.column_config.ProgressColumn(
                        format="percent",
                        min_value=0,
                        max_value=1,
                    ),
                    "Samples ≤10%": st.column_config.ProgressColumn(
                        format="percent",
                        min_value=0,
                        max_value=1,
                    ),
                    "Peak GiB": st.column_config.NumberColumn(format="%.2f"),
                    "Capacity GiB": st.column_config.NumberColumn(format="%.2f"),
                    "Peak memory share": st.column_config.ProgressColumn(
                        format="percent",
                        min_value=0,
                        max_value=1,
                    ),
                },
            )
        st.caption(
            f"{len(gpu):,} samples loaded from {gpu_path.name}; charts retain peak samples "
            "when downsampling long runs."
        )

    with process_tab:
        if process_path is None:
            st.info(
                "Per-process memory telemetry is not available for this run. Expected "
                "gpu-process-memory.csv (also accepts gpo-process-memory.csv)."
            )
        elif process.empty:
            st.caption(f"{process_path.name} contains no usable samples.")
        else:
            process_data = downsample_peaks(
                process,
                series_column="process",
                value_column="used_gib",
            )
            process_chart = (
                alt.Chart(process_data)
                .mark_line()
                .encode(
                    x=alt.X("elapsed_minutes:Q", title="Elapsed time (minutes)"),
                    y=alt.Y(
                        "used_gib:Q",
                        title="Process GPU memory (GiB)",
                        scale=alt.Scale(zero=True),
                    ),
                    color=alt.Color("process:N", title="GPU process"),
                    tooltip=[
                        alt.Tooltip("timestamp:T", title="Time"),
                        alt.Tooltip("process:N", title="Process"),
                        alt.Tooltip("used_gib:Q", title="Used GiB", format=".2f"),
                    ],
                )
                .properties(height=410, title="Per-process GPU memory")
                .interactive(bind_y=False)
            )
            st.altair_chart(process_chart, width="stretch")
            peaks = (
                process.groupby(
                    ["gpu_index", "pid", "process_name"],
                    as_index=False,
                )["used_gib"]
                .max()
                .rename(
                    columns={
                        "gpu_index": "GPU",
                        "pid": "PID",
                        "process_name": "Process",
                        "used_gib": "Peak GiB",
                    }
                )
                .sort_values("Peak GiB", ascending=False)
            )
            st.dataframe(
                peaks,
                width="stretch",
                hide_index=True,
                column_config={
                    "GPU": st.column_config.NumberColumn(format="%d"),
                    "PID": st.column_config.NumberColumn(format="%d"),
                    "Peak GiB": st.column_config.NumberColumn(format="%.2f"),
                },
            )
            st.caption(
                f"{len(process):,} samples loaded from {process_path.name}; UUIDs are mapped "
                "to GPU indices using gpu-memory.csv."
            )


def render_plan(plan: Any) -> None:
    if not isinstance(plan, dict):
        st.code("No parsed plan is available for this record.")
        return
    workflow = plan.get("workflow", [])
    st.caption(
        f'Task type: {plan.get("task_type", "")} · Difficulty: {plan.get("difficulty", "")} '
        f'· {len(workflow)} step(s)'
    )
    dot = workflow_to_graphviz(plan)
    st.graphviz_chart(dot, width="stretch")
    with st.expander("Graphviz DOT source"):
        st.code(dot, language="dot")
    for index, step in enumerate(workflow, 1):
        st.markdown(f'**{index}. {step.get("step_id", "")}** · `{step.get("model_id", "")}`')
        st.write(step.get("instruction", ""))
        accesses = " ".join(f'`{value}`' for value in step.get("access_list", []))
        if accesses:
            st.markdown(f"Access: {accesses}")


def render_record(record: TraceRecord, error_styles: dict[str, tuple[str, str]]) -> None:
    data = record.data
    title = str(data.get("question") or "Question unavailable")
    reward = data.get("reward")
    status_icon = error_styles[record.error_category][0] if data.get("error") else reward_icon(reward)
    with st.expander(
        f'**{status_icon}  {reward_label(reward)}**  ·  Batch {data.get("batch")} / Sample {data.get("sample")}  —  {title}'
    ):
        meta = [f'batch {data.get("batch")}', f'sample {data.get("sample")}']
        plan = data.get("plan") or {}
        if plan.get("task_type"):
            meta.append(str(plan["task_type"]))
        if plan.get("difficulty"):
            meta.append(str(plan["difficulty"]))
        if record.completion_tokens is not None:
            meta.append(f'{record.completion_tokens} conductor tokens{" ★" if record.completion_saturated else ""}')
        chips = "".join(f'<span class="trace-chip">{escape(item)}</span>' for item in meta)
        st.markdown(f'<div class="trace-meta">{chips}</div>', unsafe_allow_html=True)
        if data.get("error"):
            reason = f'<strong>{escape(record.error_category)}</strong><br>{escape(str(data["error"]))}'
        else:
            status = (
                "Workflow completed; the extracted final answer is available below."
                if data.get("final_answer") else "Valid workflow recorded without an execution error."
            )
            reason = f"<strong>{escape(status)}</strong>"
        st.markdown(
            f'<div class="trace-reason {reward_key(reward)}">{reason}</div>',
            unsafe_allow_html=True,
        )
        plan_tab, workers_tab, answers_tab, raw_tab = st.tabs(
            ("Parsed plan", f'Worker outputs ({len(data.get("worker_outputs") or {})})', "Answers", "Raw completion")
        )
        with plan_tab:
            render_plan(data.get("plan"))
        with workers_tab:
            outputs = data.get("worker_outputs") or {}
            if not outputs:
                st.code("No worker outputs were recorded. Parsing or validation likely failed before execution.")
            for step_id, output in outputs.items():
                st.markdown(f'**{step_id}** · `{output.get("model_id", "unknown model")}`')
                usage = output.get("usage") or {}
                metadata = []
                if output.get("latency_ms") is not None:
                    metadata.append(f'{float(output["latency_ms"]):.0f} ms')
                if usage.get("total_tokens") is not None:
                    metadata.append(f'{int(usage["total_tokens"]):,} total tokens')
                if metadata:
                    st.caption(" · ".join(metadata))
                st.code(output.get("text") or json.dumps(output, indent=2, ensure_ascii=False))
        with answers_tab:
            st.markdown("**Final answer**")
            st.code(str(data.get("final_answer") or "(none)"))
            st.markdown("**Gold answer**")
            st.code(str(data.get("gold_answer") or "(none)"))
        with raw_tab:
            st.code(str(data.get("conductor_completion") or "(none)"))


def render_trace_analysis_page() -> None:
    st.title("`theo-conductor` trace analysis")
    st.caption("Inspect reward cohorts, validation failures, conductor plans, and worker responses.")

    try:
        dataset, source_name = selected_dataset()
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        st.stop()

    st.caption(f"{source_name} · {len(dataset.records):,} records")
    if dataset.malformed_lines:
        lines = ", ".join(str(issue["line"]) for issue in dataset.malformed_lines)
        st.warning(f"Skipped {len(dataset.malformed_lines)} malformed JSONL line(s): {lines}")

    error_styles = error_style_map(dataset.records)
    render_overview(dataset, error_styles)
    render_memory_telemetry(source_name)

    st.subheader("Trace records")
    reward_values = sorted({float(record.data.get("reward", 0)) for record in dataset.records})
    categories = sorted({record.error_category for record in dataset.records})
    filter_columns = st.columns((1, 2))
    selected_rewards = filter_columns[0].multiselect(
        "Rewards", reward_values, format_func=reward_label, placeholder="All rewards"
    )
    selected_categories = filter_columns[1].multiselect("Reasons", categories, placeholder="All reasons")

    matches = dataset.query(
        TraceQuery(rewards=set(selected_rewards), categories=set(selected_categories))
    )
    pages = max(1, (len(matches) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = int(st.number_input("Page", min_value=1, max_value=pages, value=1, step=1))
    start = (page - 1) * PAGE_SIZE
    shown = matches[start : start + PAGE_SIZE]
    st.caption(f"Showing {start + 1 if shown else 0}–{start + len(shown)} of {len(matches):,} matching records")
    for trace_record in shown:
        render_record(trace_record, error_styles)


def _percent(value: Any) -> str:
    return f"{float(value):.1%}" if value is not None else "—"


def render_megascience_record(record: dict[str, Any]) -> None:
    outcome = "Request failed" if record.get("error") else ("Correct" if record.get("correct") else "Incorrect")
    icon = "🟢" if record.get("correct") else ("🔴" if record.get("error") else "🟠")
    question = str(record.get("question") or "Question unavailable")
    with st.expander(
        f'{icon} **{outcome}** · {record.get("display_name") or record.get("model_id")} · '
        f'{record.get("subject") or "unknown"} — {question}'
    ):
        metadata = []
        if record.get("example_id"):
            metadata.append(str(record["example_id"]))
        if record.get("latency_ms") is not None:
            metadata.append(f'{float(record["latency_ms"]) / 1000:.2f} s')
        if record.get("total_tokens") is not None:
            metadata.append(f'{int(record["total_tokens"]):,} tokens')
        if metadata:
            st.caption(" · ".join(metadata))
        st.markdown("**Question**")
        st.markdown(question)
        answer_columns = st.columns(2)
        with answer_columns[0]:
            st.markdown("**Extracted answer**")
            st.markdown(str(record.get("extracted_answer") or "_No `FINAL:` answer extracted._"))
        with answer_columns[1]:
            st.markdown("**Reference answer**")
            st.markdown(str(record.get("reference_answer") or record.get("gold_answer") or "—"))
        if record.get("judge_reason"):
            st.info(f'Kimi judge: {record["judge_reason"]}')
        if record.get("error"):
            st.error(str(record["error"]))
        with st.expander("Full model response"):
            st.markdown(str(record.get("response") or "_No response._"))
        with st.expander("Full gold answer"):
            st.markdown(str(record.get("gold_answer") or "—"))


def render_megascience_page() -> None:
    st.title("Small models on MegaScience")
    st.caption("Compare the local worker models on the shared deterministic MegaScience validation set.")

    summary_path = DEFAULT_MEGASCIENCE_DIR / "summary.json"
    results_path = DEFAULT_MEGASCIENCE_DIR / "results.jsonl"
    try:
        summary, records = load_megascience(
            str(summary_path),
            str(results_path),
            (summary_path.stat().st_mtime_ns, results_path.stat().st_mtime_ns),
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError) as exc:
        st.error(f"Could not load the MegaScience benchmark: {exc}")
        st.stop()

    models = summary.get("models") or {}
    expected = int(summary.get("evaluated_samples") or 0) * len(models)
    correct = sum(bool(record.get("correct")) for record in records)
    request_failures = sum(record.get("error") is not None for record in records)
    extraction_failures = sum(
        record.get("error") is None and record.get("extracted_answer") is None for record in records
    )
    oracle = oracle_routing_breakdown(records)
    oracle_correct = oracle["solved_questions"]
    oracle_accuracy = (
        oracle_correct / oracle["questions"] if oracle["questions"] else None
    )

    headline = st.columns(6)
    for column, (value, label) in zip(
        headline,
        (
            (f"{len(models)}", "Models"),
            (f'{summary.get("evaluated_samples", 0):,}', "Questions / model"),
            (f"{len(records):,} / {expected:,}", "Completed calls"),
            (_percent(correct / len(records) if records else None), "Overall accuracy"),
            (_percent(oracle_accuracy), "Oracle success rate"),
            (f"{request_failures:,}", "Request failures"),
        ),
        strict=True,
    ):
        column.metric(label, value)
    st.caption(
        f'{summary.get("dataset", "MegaScience")} · {summary.get("split", "validation")} · '
        f'seed {summary.get("seed", "—")} · temperature {summary.get("temperature", "—")} · '
        f'{summary.get("max_tokens", "—")} max output tokens'
    )

    comparison_rows = []
    subject_rows = []
    for model_id, metrics in models.items():
        ci = metrics.get("accuracy_95_ci") or [None, None]
        name = metrics.get("display_name") or model_id
        comparison_rows.append(
            {
                "Model": name,
                "Accuracy": metrics.get("accuracy"),
                "95% CI low": ci[0],
                "95% CI high": ci[1],
                "Correct": metrics.get("correct"),
                "Questions": metrics.get("questions"),
                "Missing FINAL": metrics.get("answer_extraction_failures"),
                "Mean latency (s)": float(metrics.get("mean_latency_ms") or 0) / 1000,
                "Mean tokens": metrics.get("mean_total_tokens"),
            }
        )
        for subject, values in (metrics.get("by_subject") or {}).items():
            subject_rows.append(
                {"Model": name, "Subject": subject, "Accuracy": values.get("accuracy"), "Questions": values.get("questions")}
            )

    st.subheader("Model comparison")
    comparison = pd.DataFrame(comparison_rows)
    if not comparison.empty:
        st.dataframe(
            comparison,
            width="stretch",
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1),
                "95% CI low": st.column_config.NumberColumn(format="%.1%%"),
                "95% CI high": st.column_config.NumberColumn(format="%.1%%"),
                "Mean latency (s)": st.column_config.NumberColumn(format="%.2f"),
                "Mean tokens": st.column_config.NumberColumn(format="%.0f"),
            },
        )
        resource_rows = comparison.melt(
            id_vars=["Model"],
            value_vars=["Mean latency (s)", "Mean tokens"],
            var_name="Metric",
            value_name="Value",
        )
        latency_tab, subject_tab = st.tabs(("Cost per answer", "Accuracy by subject"))
        with latency_tab:
            st.altair_chart(
                alt.Chart(resource_rows)
                .mark_bar(cornerRadiusEnd=4)
                .encode(
                    x=alt.X("Value:Q", title=None),
                    y=alt.Y("Model:N", title=None, sort="-x"),
                    color=alt.Color("Model:N", legend=None),
                    tooltip=["Model:N", "Metric:N", alt.Tooltip("Value:Q", format=",.2f")],
                    column=alt.Column("Metric:N", title=None, spacing=30),
                )
                .properties(height=170)
                .resolve_scale(x="independent"),
                width="stretch",
            )
        with subject_tab:
            if subject_rows:
                st.altair_chart(
                    alt.Chart(pd.DataFrame(subject_rows))
                    .mark_rect(cornerRadius=3)
                    .encode(
                        x=alt.X("Subject:N", title=None),
                        y=alt.Y("Model:N", title=None),
                        color=alt.Color("Accuracy:Q", scale=alt.Scale(domain=[0, 1], scheme="redyellowgreen")),
                        tooltip=["Model:N", "Subject:N", alt.Tooltip("Accuracy:Q", format=".1%"), "Questions:Q"],
                    )
                    .properties(height=170),
                    width="stretch",
                )

    st.subheader("Oracle model choices")
    oracle_rows = pd.DataFrame(
        [
            {
                "Model": row["display_name"],
                "Oracle selection credit": row["oracle_selection_credit"],
                "Share": row["oracle_selection_share"],
            }
            for row in oracle["models"]
        ]
    )
    if oracle_rows.empty:
        st.caption("No question was answered correctly by any model.")
    else:
        table_column, chart_column = st.columns((1, 1))
        with table_column:
            st.dataframe(
                oracle_rows,
                width="stretch",
                hide_index=True,
                column_config={
                    "Oracle selection credit": st.column_config.NumberColumn(format="%.2f"),
                    "Share": st.column_config.ProgressColumn(
                        format="percent", min_value=0, max_value=1
                    ),
                },
            )
        with chart_column:
            st.altair_chart(
                alt.Chart(oracle_rows)
                .mark_arc(innerRadius=42)
                .encode(
                    theta=alt.Theta("Oracle selection credit:Q"),
                    color=alt.Color("Model:N", title=None),
                    tooltip=[
                        "Model:N",
                        alt.Tooltip("Oracle selection credit:Q", format=".2f"),
                        alt.Tooltip("Share:Q", format=".1%"),
                    ],
                )
                .properties(height=230),
                width="stretch",
            )
        st.caption(
            f'Based on {oracle["solved_questions"]:,} oracle-solvable questions. '
            f'When multiple models are correct, the question is split evenly between them '
            f'({oracle["tied_questions"]:,} tied questions).'
        )

    if correct == 0 and records:
        st.warning(
            "The saved evaluator marked every answer incorrect. Use the answer browser below to compare extracted and reference answers; correctness reflects the stored benchmark labels."
        )

    st.subheader("Answer browser")
    all_models = sorted({str(record.get("display_name") or record.get("model_id")) for record in records})
    all_subjects = sorted({str(record.get("subject") or "unknown") for record in records})
    filters = st.columns((2, 1, 1, 2))
    selected_models = filters[0].multiselect("Models", all_models, placeholder="All models")
    selected_subjects = filters[1].multiselect("Subjects", all_subjects, placeholder="All subjects")
    selected_outcome = filters[2].selectbox("Outcome", ("All", "Correct", "Incorrect", "Missing FINAL", "Request failed"))
    search = filters[3].text_input("Search questions and answers")

    def matches(record: dict[str, Any]) -> bool:
        name = str(record.get("display_name") or record.get("model_id"))
        subject = str(record.get("subject") or "unknown")
        if selected_models and name not in selected_models:
            return False
        if selected_subjects and subject not in selected_subjects:
            return False
        outcomes = {
            "Correct": bool(record.get("correct")),
            "Incorrect": not record.get("correct") and record.get("error") is None,
            "Missing FINAL": record.get("error") is None and record.get("extracted_answer") is None,
            "Request failed": record.get("error") is not None,
        }
        if selected_outcome != "All" and not outcomes[selected_outcome]:
            return False
        if search:
            haystack = " ".join(str(record.get(key) or "") for key in ("question", "response", "extracted_answer", "reference_answer"))
            if search.casefold() not in haystack.casefold():
                return False
        return True

    matching_records = [record for record in records if matches(record)]
    pages = max(1, (len(matching_records) + PAGE_SIZE - 1) // PAGE_SIZE)
    page = int(st.number_input("Answer page", min_value=1, max_value=pages, value=1, step=1))
    start = (page - 1) * PAGE_SIZE
    shown = matching_records[start : start + PAGE_SIZE]
    st.caption(
        f"Showing {start + 1 if shown else 0}–{start + len(shown)} of {len(matching_records):,} answers · "
        f"{extraction_failures:,} missing FINAL answers overall"
    )
    for record in shown:
        render_megascience_record(record)


page_name = st.sidebar.radio("Viewer page", ("Trace analysis", "MegaScience · small models"))
if page_name == "MegaScience · small models":
    render_megascience_page()
else:
    render_trace_analysis_page()
