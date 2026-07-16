#!/usr/bin/env python3
"""Render GPU memory telemetry as a dependency-free standalone SVG."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path


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
COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4f46e5",
    "#65a30d",
]


@dataclass(frozen=True)
class GpuSample:
    timestamp: datetime
    used_gib: float
    total_gib: float


@dataclass(frozen=True)
class ProcessSample:
    timestamp: datetime
    used_gib: float


def parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.strip().replace("Z", "+00:00"))


def read_gpu_samples(path: Path) -> tuple[dict[int, list[GpuSample]], dict[str, int]]:
    samples: dict[int, list[GpuSample]] = defaultdict(list)
    uuid_to_index: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gpu = int(row["gpu_index"].strip())
            uuid_to_index[row["gpu_uuid"].strip()] = gpu
            samples[gpu].append(
                GpuSample(
                    timestamp=parse_timestamp(row["timestamp"]),
                    used_gib=float(row["memory_used_mib"].strip()) / 1024,
                    total_gib=float(row["memory_total_mib"].strip()) / 1024,
                )
            )
    return dict(samples), uuid_to_index


def read_process_samples(
    path: Path, uuid_to_index: dict[str, int]
) -> dict[tuple[int, int, str], list[ProcessSample]]:
    samples: dict[tuple[int, int, str], list[ProcessSample]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            uuid = row["gpu_uuid"].strip()
            if uuid not in uuid_to_index:
                continue
            key = (uuid_to_index[uuid], int(row["pid"].strip()), row["process_name"].strip())
            samples[key].append(
                ProcessSample(
                    timestamp=parse_timestamp(row["timestamp"]),
                    used_gib=float(row["used_memory_mib"].strip()) / 1024,
                )
            )
    return dict(samples)


def line_path(points: list[tuple[float, float]]) -> str:
    if not points:
        return ""
    return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)


def render_svg(
    gpu_samples: dict[int, list[GpuSample]],
    process_samples: dict[tuple[int, int, str], list[ProcessSample]],
    output: Path,
    title: str,
) -> None:
    all_gpu_samples = [sample for values in gpu_samples.values() for sample in values]
    if not all_gpu_samples:
        raise ValueError("GPU telemetry contains no samples")

    start = min(sample.timestamp for sample in all_gpu_samples)
    end = max(sample.timestamp for sample in all_gpu_samples)
    duration = max((end - start).total_seconds(), 1)
    gpu_ids = sorted(gpu_samples)

    width = 1500
    left, right = 205, 70
    plot_width = width - left - right
    row_height, row_gap = 112, 24
    top = 100
    summary_top = top + len(gpu_ids) * (row_height + row_gap) + 35
    height = summary_top + 230

    def x_for(timestamp: datetime) -> float:
        return left + ((timestamp - start).total_seconds() / duration) * plot_width

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;fill:#dbeafe}",
        ".title{font-size:24px;font-weight:700}.subtitle{font-size:13px;fill:#94a3b8}",
        ".label{font-size:13px;font-weight:600}.small{font-size:11px;fill:#94a3b8}",
        ".grid{stroke:#334155;stroke-width:1}.total{fill:none;stroke:#f8fafc;stroke-width:2.4}",
        "</style>",
        '<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<text x="32" y="40" class="title">{escape(title)}</text>',
        f'<text x="32" y="66" class="subtitle">{start.isoformat()} to {end.isoformat()} · {duration / 60:.1f} minutes · white = total device use · colors = CUDA processes</text>',
    ]

    process_keys = sorted(process_samples, key=lambda key: (key[1], key[0], key[2]))
    process_colors = {key: COLORS[index % len(COLORS)] for index, key in enumerate(process_keys)}

    for row_index, gpu in enumerate(gpu_ids):
        y_top = top + row_index * (row_height + row_gap)
        values = gpu_samples[gpu]
        capacity = max(sample.total_gib for sample in values)

        def y_for(gib: float) -> float:
            return y_top + row_height - min(gib / capacity, 1) * row_height

        peak = max(values, key=lambda sample: sample.used_gib)
        final = values[-1]
        role = GPU_ROLES.get(gpu, "")
        parts.extend(
            [
                f'<rect x="{left}" y="{y_top}" width="{plot_width}" height="{row_height}" fill="#111c31" stroke="#334155"/>',
                f'<text x="32" y="{y_top + 25}" class="label">GPU {gpu} · {escape(role)}</text>',
                f'<text x="32" y="{y_top + 48}" class="small">peak {peak.used_gib:.1f} GiB ({peak.used_gib / capacity:.0%})</text>',
                f'<text x="32" y="{y_top + 66}" class="small">final {final.used_gib:.1f} GiB</text>',
            ]
        )
        for fraction in (0.25, 0.5, 0.75):
            y = y_for(capacity * fraction)
            parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" class="grid"/>')
        parts.append(
            f'<path d="{line_path([(x_for(sample.timestamp), y_for(sample.used_gib)) for sample in values])}" class="total"/>'
        )

        for key, samples in process_samples.items():
            if key[0] != gpu:
                continue
            points = [(x_for(sample.timestamp), y_for(sample.used_gib)) for sample in samples]
            parts.append(
                f'<path d="{line_path(points)}" fill="none" stroke="{process_colors[key]}" stroke-width="1.5" opacity="0.9"/>'
            )

    for minute in range(0, int(duration // 60) + 1, max(1, int(duration // 360) or 1)):
        x = left + (minute * 60 / duration) * plot_width
        parts.append(f'<text x="{x:.1f}" y="{summary_top - 18}" text-anchor="middle" class="small">+{minute}m</text>')

    parts.append(f'<text x="32" y="{summary_top + 10}" class="label">Process legend (peak memory on any GPU)</text>')
    legend_y = summary_top + 38
    for index, key in enumerate(process_keys):
        gpu, pid, name = key
        peak = max(sample.used_gib for sample in process_samples[key])
        column = index % 3
        row = index // 3
        x = 32 + column * 480
        y = legend_y + row * 24
        parts.extend(
            [
                f'<line x1="{x}" y1="{y - 4}" x2="{x + 24}" y2="{y - 4}" stroke="{process_colors[key]}" stroke-width="3"/>',
                f'<text x="{x + 32}" y="{y}" class="small">PID {pid} · {escape(name)} · peak {peak:.1f} GiB</text>',
            ]
        )

    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def print_summary(
    gpu_samples: dict[int, list[GpuSample]],
    process_samples: dict[tuple[int, int, str], list[ProcessSample]],
) -> None:
    print("GPU  role                  peak GiB   peak %   final GiB")
    for gpu, values in sorted(gpu_samples.items()):
        peak = max(sample.used_gib for sample in values)
        capacity = max(sample.total_gib for sample in values)
        print(f"{gpu:>3}  {GPU_ROLES.get(gpu, ''):<20} {peak:>8.1f} {peak / capacity:>8.1%} {values[-1].used_gib:>10.1f}")

    print("\nProcesses by peak GPU memory:")
    ranked = sorted(
        process_samples.items(), key=lambda item: max(sample.used_gib for sample in item[1]), reverse=True
    )
    for (gpu, pid, name), values in ranked:
        peak = max(sample.used_gib for sample in values)
        print(f"  GPU {gpu}  PID {pid:<8} {name:<22} {peak:>7.1f} GiB")

    if 6 in gpu_samples and 7 in gpu_samples:
        gpu6_final = gpu_samples[6][-1].used_gib
        gpu7_final = gpu_samples[7][-1].used_gib
        capacity = max(sample.total_gib for sample in gpu_samples[6])
        if gpu6_final > capacity * 0.9 and gpu7_final < capacity * 0.1:
            print("\nANOMALY: conductor memory is concentrated on GPU 6 while GPU 7 is nearly idle at the end.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing gpu-memory.csv and gpu-process-memory.csv")
    parser.add_argument("--output", type=Path, help="Output SVG path (default: RUN_DIR/gpu-memory.svg)")
    parser.add_argument("--title", help="Plot title")
    args = parser.parse_args()

    gpu_csv = args.run_dir / "gpu-memory.csv"
    process_csv = args.run_dir / "gpu-process-memory.csv"
    output = args.output or args.run_dir / "gpu-memory.svg"
    gpu_samples, uuid_to_index = read_gpu_samples(gpu_csv)
    process_samples = read_process_samples(process_csv, uuid_to_index)
    render_svg(gpu_samples, process_samples, output, args.title or f"GPU memory · {args.run_dir.name}")
    print_summary(gpu_samples, process_samples)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
