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
from theo_conductor.trace_analysis import TraceDataset, TraceQuery, TraceRecord, error_category


ROOT = Path(__file__).resolve().parent
DEFAULT_TRACE = ROOT / "outputs/grpo-11352/traces/plans-and-worker-outputs-rank-0.jsonl"
PAGE_SIZE = 80
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


st.set_page_config(page_title="GRPO trace analysis", page_icon="◈", layout="wide")


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


def selected_dataset() -> tuple[TraceDataset, str]:
    query_trace = st.query_params.get("trace")
    default_value = str(query_trace) if query_trace else str(DEFAULT_TRACE.relative_to(ROOT))
    source = st.sidebar.radio("Trace source", ("Repository path", "SLURM job", "Upload JSONL"))
    if source == "Upload JSONL":
        upload = st.sidebar.file_uploader("Trace file", type=("jsonl", "json"))
        if upload is None:
            st.info("Upload a JSONL trace to begin.")
            st.stop()
        return load_upload(upload.getvalue(), upload.name), upload.name

    if source == "SLURM job":
        job_id = st.sidebar.text_input("SLURM job ID", placeholder="11352")
        if not job_id:
            st.info("Enter a SLURM job ID in the sidebar.")
            st.stop()
        if not job_id[0].isdigit() or any(char not in "0123456789_-" for char in job_id):
            raise ValueError("Job IDs must begin with a number and contain only numbers, '_' or '-'.")
        relative = Path(f"outputs/grpo-{job_id}/traces/plans-and-worker-outputs-rank-0.jsonl")
    else:
        relative = Path(st.sidebar.text_input("Trace path", value=default_value))

    path = relative if relative.is_absolute() else ROOT / relative
    path = path.resolve()
    try:
        path.relative_to(ROOT)
    except ValueError as exc:
        raise ValueError("Repository trace paths must stay inside the repository.") from exc
    if not path.is_file():
        raise FileNotFoundError(f"Trace not found: {path}")
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
        (f'{token_info["max"]:,}' if token_info else "—", "Max conductor tokens"),
    )
    columns = st.columns(6)
    for column, (value, label) in zip(columns, values, strict=True):
        column.metric(label, value)

    st.subheader("Conductor plan statistics")
    plans = [record.data["plan"] for record in records if isinstance(record.data.get("plan"), dict)]
    steps = [step for plan in plans for step in plan.get("workflow", []) if isinstance(step, dict)]
    multi_step = sum(len(plan.get("workflow", [])) > 1 for plan in plans)
    models = Counter(str(step.get("model_id") or "(missing)") for step in steps)
    plan_values = (
        (f"{len(steps) / len(plans):.2f}" if plans else "—", "Mean planned steps"),
        (f"{multi_step / len(plans):.1%}" if plans else "—", "Multi-step plans"),
        (f'{token_info["mean"]:,.1f}' if token_info else "—", "Mean conductor tokens"),
    )
    columns = st.columns(3)
    for column, (value, label) in zip(columns, plan_values, strict=True):
        column.metric(label, value)

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


def render_plan(plan: Any) -> None:
    if not isinstance(plan, dict):
        st.code("No parsed plan is available for this record.")
        return
    workflow = plan.get("workflow", [])
    st.caption(
        f'Task type: {plan.get("task_type", "")} · Difficulty: {plan.get("difficulty", "")} '
        f'· {len(workflow)} step(s)'
    )
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

st.subheader("Trace records")
reward_values = sorted({float(record.data.get("reward", 0)) for record in dataset.records})
categories = sorted({record.error_category for record in dataset.records})
filter_columns = st.columns((1, 2))
selected_rewards = filter_columns[0].multiselect(
    "Rewards", reward_values, format_func=reward_label, placeholder="All rewards"
)
selected_categories = filter_columns[1].multiselect("Reasons", categories, placeholder="All reasons")

matches = dataset.query(
    TraceQuery(
        rewards=set(selected_rewards),
        categories=set(selected_categories),
    )
)
pages = max(1, (len(matches) + PAGE_SIZE - 1) // PAGE_SIZE)
page = int(st.number_input("Page", min_value=1, max_value=pages, value=1, step=1))
start = (page - 1) * PAGE_SIZE
shown = matches[start : start + PAGE_SIZE]
st.caption(f"Showing {start + 1 if shown else 0}–{start + len(shown)} of {len(matches):,} matching records")
for trace_record in shown:
    render_record(trace_record, error_styles)
