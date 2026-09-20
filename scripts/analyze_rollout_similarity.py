#!/usr/bin/env python3
"""Analyze within-question similarity for pregenerated conductor rollouts.

The analysis deliberately keeps topology/model routing and instruction semantics
as separate measurements.  It also simulates smaller rollout budgets by drawing
subsets from the observed 64 rollouts and measuring how much of the full set they
cover.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
WORD_RE = re.compile(r"\w+")


def percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else math.nan


def fmt(value: float, digits: int = 3) -> str:
    return "n/a" if math.isnan(value) else f"{value:.{digits}f}"


def ratio_similarity(a: int, b: int) -> float:
    if a == b == 0:
        return 1.0
    return min(a, b) / max(a, b)


def normalized_text(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


@dataclass
class Workflow:
    rollout: int
    dataset_index: int
    dataset_id: str
    question: str
    plan: dict[str, Any] | None
    error_type: str | None
    error: str | None

    @property
    def valid(self) -> bool:
        return self.plan is not None


@dataclass
class Graph:
    models: tuple[str, ...]
    instructions: tuple[str, ...]
    edges: frozenset[tuple[int, int]]
    layers: tuple[int, ...]
    indegrees: tuple[int, ...]
    outdegrees: tuple[int, ...]
    question_access: tuple[bool, ...]
    needs_tools: tuple[bool, ...]

    @property
    def size(self) -> int:
        return len(self.models)

    @property
    def topology_signature(self) -> tuple[Any, ...]:
        return (
            self.size,
            tuple(sorted(self.edges)),
            self.layers,
            self.question_access,
            self.needs_tools,
        )

    @property
    def route_signature(self) -> tuple[Any, ...]:
        return self.topology_signature + (self.models,)


def plan_graph(plan: dict[str, Any]) -> Graph:
    steps = plan["workflow"]
    ids = {str(step["step_id"]): i for i, step in enumerate(steps)}
    edges: set[tuple[int, int]] = set()
    question_access: list[bool] = []
    for dst, step in enumerate(steps):
        refs = list(step.get("access_list", []))
        refs += list(step.get("artifact_inputs", []))
        refs += list(step.get("depends_on", []))
        question_access.append("question" in refs)
        for ref in refs:
            src = ids.get(str(ref))
            if src is not None and src != dst:
                edges.add((src, dst))

    n = len(steps)
    incoming = [[] for _ in range(n)]
    outgoing = [[] for _ in range(n)]
    for src, dst in edges:
        incoming[dst].append(src)
        outgoing[src].append(dst)

    # Valid plans are topologically ordered, but use a small fixed-point loop so
    # the analysis is robust to any unusual ordering.
    layers = [0] * n
    for _ in range(n):
        changed = False
        for node in range(n):
            layer = 0 if not incoming[node] else 1 + max(layers[p] for p in incoming[node])
            if layer != layers[node]:
                layers[node] = layer
                changed = True
        if not changed:
            break

    return Graph(
        models=tuple(str(step["model_id"]) for step in steps),
        instructions=tuple(str(step["instruction"]) for step in steps),
        edges=frozenset(edges),
        layers=tuple(layers),
        indegrees=tuple(len(x) for x in incoming),
        outdegrees=tuple(len(x) for x in outgoing),
        question_access=tuple(question_access),
        needs_tools=tuple(bool(step.get("needs_tools", False)) for step in steps),
    )


def max_assignment(scores: np.ndarray) -> tuple[float, list[tuple[int, int]]]:
    """Exact maximum-weight one-to-one assignment for matrices up to 7x7."""
    rows, cols = scores.shape
    transposed = False
    if rows > cols:
        scores = scores.T
        rows, cols = scores.shape
        transposed = True

    states: dict[int, tuple[float, list[tuple[int, int]]]] = {0: (0.0, [])}
    for row in range(rows):
        next_states: dict[int, tuple[float, list[tuple[int, int]]]] = {}
        for mask, (total, pairs) in states.items():
            for col in range(cols):
                if mask & (1 << col):
                    continue
                next_mask = mask | (1 << col)
                candidate = total + float(scores[row, col])
                prior = next_states.get(next_mask)
                if prior is None or candidate > prior[0]:
                    next_states[next_mask] = (candidate, pairs + [(row, col)])
        states = next_states
    total, pairs = max(states.values(), key=lambda item: item[0])
    if transposed:
        pairs = [(col, row) for row, col in pairs]
    return total, pairs


def structural_similarity(a: Graph, b: Graph) -> tuple[float, float, float]:
    """Return combined structural, topology, and model-routing similarity.

    Nodes are optimally paired by graph role, without looking at instruction
    text or model name.  Unmatched nodes contribute zero.  Topology combines
    node-role agreement and exact dependency-edge overlap after that pairing.
    """
    n, m = a.size, b.size
    local = np.zeros((n, m), dtype=np.float32)
    max_layer_a = max(a.layers, default=0)
    max_layer_b = max(b.layers, default=0)
    for i in range(n):
        for j in range(m):
            pos_a = i / max(n - 1, 1)
            pos_b = j / max(m - 1, 1)
            layer_a = a.layers[i] / max(max_layer_a, 1)
            layer_b = b.layers[j] / max(max_layer_b, 1)
            local[i, j] = mean(
                [
                    1.0 - abs(pos_a - pos_b),
                    ratio_similarity(a.indegrees[i], b.indegrees[j]),
                    ratio_similarity(a.outdegrees[i], b.outdegrees[j]),
                    1.0 - abs(layer_a - layer_b),
                    float(a.question_access[i] == b.question_access[j]),
                    float(a.needs_tools[i] == b.needs_tools[j]),
                ]
            )
    role_total, pairs = max_assignment(local)
    role_similarity = role_total / max(n, m)
    mapping = dict(pairs)
    overlap = sum((mapping.get(src), mapping.get(dst)) in b.edges for src, dst in a.edges)
    edge_similarity = (
        1.0 if not a.edges and not b.edges else 2.0 * overlap / (len(a.edges) + len(b.edges))
    )
    topology = 0.5 * role_similarity + 0.5 * edge_similarity
    model_matches = sum(a.models[i] == b.models[j] for i, j in pairs)
    routing = model_matches / max(n, m)
    combined = 0.7 * topology + 0.3 * routing
    return combined, topology, routing


def semantic_similarity(a: Graph, b: Graph, embeddings: dict[str, np.ndarray]) -> float:
    matrix = np.asarray(
        [[float(np.dot(embeddings[x], embeddings[y])) for y in b.instructions] for x in a.instructions],
        dtype=np.float32,
    )
    matrix = np.clip(matrix, 0.0, 1.0)
    total, _ = max_assignment(matrix)
    return total / max(a.size, b.size)


def load_rollouts(input_dir: Path) -> tuple[list[Workflow], dict[str, list[Workflow]]]:
    files = sorted(input_dir.glob("rollout-*/plans.jsonl"))
    if len(files) != 64:
        raise ValueError(f"Expected 64 rollout files, found {len(files)} in {input_dir}")
    all_rows: list[Workflow] = []
    by_question: dict[str, list[Workflow]] = defaultdict(list)
    for path in files:
        rollout = int(path.parent.name.split("-")[-1])
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(rows) != 202:
            raise ValueError(f"Expected 202 records in {path}, found {len(rows)}")
        for row in rows:
            item = Workflow(
                rollout=rollout,
                dataset_index=int(row["dataset_index"]),
                dataset_id=str(row["dataset_id"]),
                question=str(row["question"]),
                plan=row.get("plan"),
                error_type=row.get("error_type"),
                error=row.get("error"),
            )
            all_rows.append(item)
            by_question[item.dataset_id].append(item)
    if len(by_question) != 202 or any(len(rows) != 64 for rows in by_question.values()):
        raise ValueError("Rollouts do not form a complete 202-question by 64-rollout grid")
    for rows in by_question.values():
        rows.sort(key=lambda row: row.rollout)
    return all_rows, by_question


def embed_instructions(
    instructions: list[str], cache_path: Path, batch_size: int
) -> dict[str, np.ndarray]:
    unique = sorted(set(instructions))
    digest = hashlib.sha256("\0".join(unique).encode()).hexdigest()
    if cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        if str(cached["digest"].item()) == digest:
            matrix = cached["embeddings"]
            return {text: matrix[i] for i, text in enumerate(cached["instructions"].tolist())}

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    model = AutoModel.from_pretrained(EMBEDDING_MODEL)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    vectors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(unique), batch_size):
            batch = unique[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            encoded = {name: tensor.to(device) for name, tensor in encoded.items()}
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            vectors.append(pooled.cpu().numpy().astype(np.float32))
            print(f"embedded={min(start + batch_size, len(unique))}/{len(unique)}", flush=True)
    matrix = np.concatenate(vectors)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        digest=np.asarray(digest),
        instructions=np.asarray(unique, dtype=object),
        embeddings=matrix,
    )
    return {text: matrix[i] for i, text in enumerate(unique)}


def instruction_coverage(
    graphs: list[Graph], embeddings: dict[str, np.ndarray], selected: list[int], threshold: float
) -> float:
    selected_instructions = [text for i in selected for text in graphs[i].instructions]
    if not selected_instructions:
        return 0.0
    selected_matrix = np.stack([embeddings[text] for text in selected_instructions])
    all_matrix = np.stack([embeddings[text] for graph in graphs for text in graph.instructions])
    # Chunking avoids a large temporary for unusually long workflows.
    covered = 0
    for start in range(0, len(all_matrix), 256):
        similarity = all_matrix[start : start + 256] @ selected_matrix.T
        covered += int(np.sum(np.max(similarity, axis=1) >= threshold))
    return covered / len(all_matrix)


def simulate_subsets(
    rows: list[Workflow],
    graphs: list[Graph],
    combined: np.ndarray,
    embeddings: dict[str, np.ndarray],
    trials: int,
    rng: random.Random,
) -> dict[int, dict[str, float]]:
    valid_rollouts = [row.rollout for row in rows if row.valid]
    rollout_to_graph = {rollout: i for i, rollout in enumerate(valid_rollouts)}
    topo = [graph.topology_signature for graph in graphs]
    route = [graph.route_signature for graph in graphs]
    result: dict[int, dict[str, float]] = {}
    for k in (16, 32):
        metrics: dict[str, list[float]] = defaultdict(list)
        for _ in range(trials):
            sampled_rollouts = rng.sample(range(64), k)
            selected = [rollout_to_graph[r] for r in sampled_rollouts if r in rollout_to_graph]
            metrics["valid_count"].append(float(len(selected)))
            if not selected:
                metrics["workflow_coverage_080"].append(0.0)
                metrics["workflow_coverage_085"].append(0.0)
                metrics["workflow_coverage_090"].append(0.0)
                metrics["instruction_coverage_090"].append(0.0)
                metrics["topology_mass"].append(0.0)
                metrics["route_mass"].append(0.0)
                continue
            nearest = np.max(combined[:, selected], axis=1)
            for threshold in (0.80, 0.85, 0.90):
                metrics[f"workflow_coverage_{int(threshold * 100):03d}"].append(
                    float(np.mean(nearest >= threshold))
                )
            metrics["instruction_coverage_090"].append(
                instruction_coverage(graphs, embeddings, selected, 0.90)
            )
            selected_topo = {topo[i] for i in selected}
            selected_route = {route[i] for i in selected}
            metrics["topology_mass"].append(mean(signature in selected_topo for signature in topo))
            metrics["route_mass"].append(mean(signature in selected_route for signature in route))

        flat: dict[str, float] = {}
        for name, values in metrics.items():
            flat[name] = mean(values)
            flat[f"{name}_p05"] = percentile(values, 5)
        result[k] = flat
    return result


def pair_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices(len(matrix), 1)]


def short_question(text: str, limit: int = 55) -> str:
    clean = " ".join(text.split()).replace("|", "\\|")
    return clean if len(clean) <= limit else clean[: limit - 1] + "…"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-rollout-csv", type=Path, required=True)
    parser.add_argument("--pairwise-csv", type=Path, required=True)
    parser.add_argument("--metrics-json", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=21510)
    args = parser.parse_args()

    all_rows, by_question = load_rollouts(args.input_dir)
    graphs_by_question: dict[str, list[Graph]] = {}
    instructions: list[str] = []
    for qid, rows in by_question.items():
        graphs = [plan_graph(row.plan) for row in rows if row.plan is not None]
        graphs_by_question[qid] = graphs
        instructions.extend(text for graph in graphs for text in graph.instructions)
    embeddings = embed_instructions(instructions, args.embedding_cache, args.batch_size)

    question_results: list[dict[str, Any]] = []
    rollout_results: list[dict[str, Any]] = []
    pairwise_results: list[tuple[Any, ...]] = []
    global_pairs: dict[str, list[float]] = defaultdict(list)
    downsample_by_k: dict[int, dict[str, list[float]]] = {
        16: defaultdict(list),
        32: defaultdict(list),
    }
    rng = random.Random(args.seed)

    ordered_questions = sorted(by_question.items(), key=lambda item: item[1][0].dataset_index)
    for qnum, (qid, rows) in enumerate(ordered_questions, 1):
        valid_rows = [row for row in rows if row.valid]
        graphs = graphs_by_question[qid]
        n = len(graphs)
        structural = np.eye(n, dtype=np.float32)
        topology = np.eye(n, dtype=np.float32)
        routing = np.eye(n, dtype=np.float32)
        semantic = np.eye(n, dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                s, t, r = structural_similarity(graphs[i], graphs[j])
                sem = semantic_similarity(graphs[i], graphs[j], embeddings)
                structural[i, j] = structural[j, i] = s
                topology[i, j] = topology[j, i] = t
                routing[i, j] = routing[j, i] = r
                semantic[i, j] = semantic[j, i] = sem
        combined = 0.5 * structural + 0.5 * semantic
        valid_rollouts = [row.rollout for row in valid_rows]
        for i in range(n):
            for j in range(i + 1, n):
                pairwise_results.append(
                    (
                        rows[0].dataset_index,
                        qid,
                        valid_rollouts[i],
                        valid_rollouts[j],
                        graphs[i].size,
                        graphs[j].size,
                        float(topology[i, j]),
                        float(routing[i, j]),
                        float(structural[i, j]),
                        float(semantic[i, j]),
                        float(combined[i, j]),
                    )
                )
        pair_struct = pair_values(structural)
        pair_topo = pair_values(topology)
        pair_route = pair_values(routing)
        pair_sem = pair_values(semantic)
        pair_combined = pair_values(combined)
        for name, values in (
            ("structural", pair_struct),
            ("topology", pair_topo),
            ("routing", pair_route),
            ("semantic", pair_sem),
            ("combined", pair_combined),
        ):
            global_pairs[name].extend(float(x) for x in values)

        simulations = simulate_subsets(rows, graphs, combined, embeddings, args.trials, rng)
        for k, metrics in simulations.items():
            for name, value in metrics.items():
                downsample_by_k[k][name].append(value)

        topo_counts = Counter(graph.topology_signature for graph in graphs)
        route_counts = Counter(graph.route_signature for graph in graphs)
        exact_instruction_sets = Counter(
            tuple(sorted(normalized_text(x) for x in graph.instructions)) for graph in graphs
        )
        invalid_types = Counter(row.error_type or "unknown" for row in rows if not row.valid)
        qr = {
            "dataset_index": rows[0].dataset_index,
            "dataset_id": qid,
            "question": rows[0].question,
            "valid": n,
            "invalid": 64 - n,
            "invalid_types": dict(invalid_types),
            "mean_structural": float(np.mean(pair_struct)),
            "p10_structural": percentile(pair_struct, 10),
            "mean_topology": float(np.mean(pair_topo)),
            "mean_routing": float(np.mean(pair_route)),
            "mean_semantic": float(np.mean(pair_sem)),
            "p10_semantic": percentile(pair_sem, 10),
            "mean_combined": float(np.mean(pair_combined)),
            "topology_archetypes": len(topo_counts),
            "route_archetypes": len(route_counts),
            "exact_instruction_sets": len(exact_instruction_sets),
            "largest_topology_share": max(topo_counts.values()) / n,
            "largest_route_share": max(route_counts.values()) / n,
            "k16": simulations[16],
            "k32": simulations[32],
        }
        question_results.append(qr)

        for i, row in enumerate(valid_rows):
            other = [j for j in range(n) if j != i]
            nearest = max(other, key=lambda j: combined[i, j])
            rollout_results.append(
                {
                    "dataset_index": row.dataset_index,
                    "dataset_id": row.dataset_id,
                    "rollout": row.rollout,
                    "valid": True,
                    "error_type": "",
                    "step_count": graphs[i].size,
                    "mean_structural_to_others": float(np.mean(structural[i, other])),
                    "mean_semantic_to_others": float(np.mean(semantic[i, other])),
                    "mean_combined_to_others": float(np.mean(combined[i, other])),
                    "nearest_rollout": valid_rollouts[nearest],
                    "nearest_structural": float(structural[i, nearest]),
                    "nearest_semantic": float(semantic[i, nearest]),
                    "nearest_combined": float(combined[i, nearest]),
                    "novelty_one_minus_nearest_combined": float(1.0 - combined[i, nearest]),
                }
            )
        for row in rows:
            if not row.valid:
                rollout_results.append(
                    {
                        "dataset_index": row.dataset_index,
                        "dataset_id": row.dataset_id,
                        "rollout": row.rollout,
                        "valid": False,
                        "error_type": row.error_type or "unknown",
                        "step_count": "",
                        "mean_structural_to_others": "",
                        "mean_semantic_to_others": "",
                        "mean_combined_to_others": "",
                        "nearest_rollout": "",
                        "nearest_structural": "",
                        "nearest_semantic": "",
                        "nearest_combined": "",
                        "novelty_one_minus_nearest_combined": "",
                    }
                )
        print(f"questions={qnum}/202", flush=True)

    rollout_results.sort(key=lambda row: (row["dataset_index"], row["rollout"]))
    args.per_rollout_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.per_rollout_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rollout_results[0]))
        writer.writeheader()
        writer.writerows(rollout_results)

    args.pairwise_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.pairwise_csv.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "dataset_index",
                "dataset_id",
                "rollout_a",
                "rollout_b",
                "steps_a",
                "steps_b",
                "topology",
                "routing",
                "structural",
                "semantic",
                "combined",
            )
        )
        writer.writerows(pairwise_results)

    validity = [row.valid for row in all_rows]
    step_counts = [graph.size for graphs in graphs_by_question.values() for graph in graphs]
    invalid_by_question = sorted(question_results, key=lambda row: (-row["invalid"], row["dataset_index"]))
    summary: dict[str, Any] = {
        "run": 21510,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "records": len(all_rows),
        "question_count": len(by_question),
        "rollouts_per_question": 64,
        "valid": sum(validity),
        "invalid": len(validity) - sum(validity),
        "valid_rate": mean(validity),
        "step_count_mean": mean(step_counts),
        "step_count_distribution": dict(sorted(Counter(step_counts).items())),
        "pairwise": {
            name: {
                "count": len(values),
                "min": min(values),
                "q1": percentile(values, 25),
                "mean": mean(values),
                "median": median(values),
                "q3": percentile(values, 75),
                "max": max(values),
                "p10": percentile(values, 10),
                "p90": percentile(values, 90),
            }
            for name, values in global_pairs.items()
        },
        "questions": question_results,
        "downsampling": {},
    }
    for k, metrics in downsample_by_k.items():
        summary["downsampling"][str(k)] = {
            name: {
                "question_mean": mean(values),
                "question_median": median(values),
                "question_p10": percentile(values, 10),
                "question_p90": percentile(values, 90),
            }
            for name, values in metrics.items()
        }
    args.metrics_json.write_text(json.dumps(summary, indent=2))

    pair = summary["pairwise"]
    down = summary["downsampling"]
    mean_topology_archetypes = mean(row["topology_archetypes"] for row in question_results)
    mean_route_archetypes = mean(row["route_archetypes"] for row in question_results)
    mean_exact_instruction_sets = mean(row["exact_instruction_sets"] for row in question_results)
    lines: list[str] = []
    lines += [
        "# Rollout similarity analysis — Slurm run 21510",
        "",
        f"Generated {summary['generated_at']}. Analysis code: `scripts/analyze_rollout_similarity.py`.",
        "",
        "## Decision",
        "",
    ]
    c32 = down["32"]["workflow_coverage_085"]["question_mean"]
    i32 = down["32"]["instruction_coverage_090"]["question_mean"]
    c16 = down["16"]["workflow_coverage_085"]["question_mean"]
    i16 = down["16"]["instruction_coverage_090"]["question_mean"]
    if c32 >= 0.95 and i32 >= 0.95:
        lines.append(
            f"**The diversity evidence supports reducing to 32 rollouts/question:** random 32-rollout subsets cover "
            f"{c32:.1%} of observed workflow variants at the 0.85 combined-similarity threshold and "
            f"{i32:.1%} of individual instructions at cosine ≥ 0.90."
        )
    else:
        lines.append(
            f"**The diversity evidence does not cleanly support 32 rollouts/question:** its estimated coverage is "
            f"{c32:.1%} for workflow variants and {i32:.1%} for individual instructions."
        )
    lines += ["", (
        f"**Reducing to 16 is materially more aggressive:** estimated coverage falls to {c16:.1%} for workflow "
        f"variants and {i16:.1%} for individual instructions. Treat 16 as an experiment, not a default, unless an "
        "end-to-end training ablation shows unchanged reward and gradient stability."
    ), "", (
        "This run contains plans only: every valid record has the neutral placeholder reward and no worker execution. "
        "Similarity can show redundancy, but it cannot establish that fewer GRPO samples preserve reward tails, "
        "advantage estimation, or final model quality."
    ), "", "## Dataset integrity", "", (
        f"All 64 rollout files contain all 202 question IDs ({len(all_rows):,} records). "
        f"{summary['valid']:,} plans are valid ({summary['valid_rate']:.2%}); {summary['invalid']:,} are parse/validation "
        "failures and are retained in coverage simulations as unusable draws."
    ), "", f"Mean workflow length is {summary['step_count_mean']:.2f} steps. Distribution: " + ", ".join(
        f"{k} step{'s' if int(k) != 1 else ''}: {v:,}" for k, v in summary["step_count_distribution"].items()
    ) + ".", "", "## Similarity results", "", (
        "Scores range from 0 (unrelated) to 1 (identical under the metric). These are all valid-valid pairs within "
        "the same question; invalid plans have no invented similarity score."
    ), "", "| Measure | Min | Q1 | Median | Q3 | Max | Mean | P10 | P90 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for label, key in (
        ("Topology", "topology"),
        ("Model routing", "routing"),
        ("Structural (70% topology, 30% routing)", "structural"),
        ("Instruction semantic", "semantic"),
        ("Combined (50% structural, 50% semantic)", "combined"),
    ):
        item = pair[key]
        lines.append(
            f"| {label} | {item['min']:.3f} | {item['q1']:.3f} | {item['median']:.3f} | "
            f"{item['q3']:.3f} | {item['max']:.3f} | {item['mean']:.3f} | "
            f"{item['p10']:.3f} | {item['p90']:.3f} |"
        )

    lines += ["", "### What the measures mean", "", (
        "Topology matches steps by graph role (normalized position, dependency layer, in/out degree, question access, "
        "and tool use), penalizes unmatched steps, and measures dependency-edge overlap. Model routing is exact model "
        "agreement for those matched roles."
    ), "", (
        f"Instruction semantics uses `{EMBEDDING_MODEL}` embeddings and exact maximum-weight one-to-one matching of "
        "individual step instructions. Unmatched steps score zero. This catches paraphrases without allowing several "
        "instructions in one plan to all claim the same counterpart in another."
    ), "", (
        "The combined score is used only for the coverage simulation. The separate columns should remain the primary "
        "evidence because any combined weighting is a policy choice."
    ), "", (
        f"Exact text makes the plans look almost entirely unique ({mean_exact_instruction_sets:.1f} distinct normalized "
        f"instruction sets per question out of about 64), but structure collapses to {mean_topology_archetypes:.1f} "
        f"topology archetypes on average. Adding exact model assignments raises that to {mean_route_archetypes:.1f} "
        "route archetypes. This gap is why neither exact-string matching nor structure alone is adequate."
    ), "", "## Simulated reduction from 64", "", (
        f"For each question, {args.trials} random subsets of 16 and 32 positions were drawn from the actual 64, including "
        "invalid positions. Workflow coverage is the share of all valid 64-rollout workflows having a selected neighbor "
        "at or above the stated combined-similarity threshold. Instruction coverage is the share of all individual "
        "instructions having a selected instruction with embedding cosine ≥ 0.90."
    ), "", "| Budget | Valid draws (mean) | Workflow cov. ≥.80 | ≥.85 | ≥.90 | Instruction cov. ≥.90 | Exact topology mass | Exact route mass |", "|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for k in (32, 16):
        d = down[str(k)]
        lines.append(
            f"| {k} | {d['valid_count']['question_mean']:.2f} | "
            f"{d['workflow_coverage_080']['question_mean']:.1%} | "
            f"{d['workflow_coverage_085']['question_mean']:.1%} | "
            f"{d['workflow_coverage_090']['question_mean']:.1%} | "
            f"{d['instruction_coverage_090']['question_mean']:.1%} | "
            f"{d['topology_mass']['question_mean']:.1%} | {d['route_mass']['question_mean']:.1%} |"
        )
    lines += ["", (
        "“Mass” is prevalence-weighted: a full-set rollout is covered only when its exact observed topology or exact "
        "topology-plus-model route occurs in the subset. This is deliberately stricter than semantic coverage."
    ), "", "### Variation across questions", "", "| Budget | Metric | P10 question | Median question | P90 question |", "|---:|---|---:|---:|---:|"]
    for k in (32, 16):
        d = down[str(k)]
        for label, key in (
            ("Workflow coverage ≥.85", "workflow_coverage_085"),
            ("Instruction coverage ≥.90", "instruction_coverage_090"),
            ("Exact topology mass", "topology_mass"),
            ("Exact route mass", "route_mass"),
        ):
            lines.append(
                f"| {k} | {label} | {d[key]['question_p10']:.1%} | {d[key]['question_median']:.1%} | {d[key]['question_p90']:.1%} |"
            )

    lines += ["", "## Recommendation and training implications", "", (
        "Use **32 rollouts/question as the next default candidate**, conditional on a controlled training ablation. It "
        "halves generation cost while retaining substantially more of the observed semantic and structural support than "
        "16. Keep 64 for questions or batches where reward variance is high, valid generations are scarce, or rare "
        "workflow discovery is itself important."
    ), "", (
        "Do not adopt 16 globally from this analysis alone. If cost pressure makes 16 attractive, a better policy is "
        "adaptive: start at 16, add another 16 when valid count is low, rewards disagree, or the first 16 occupy many "
        "distinct structural/semantic modes. The present plan-only data can motivate that policy, but executed, judged "
        "rollouts are required to tune its trigger."
    ), "", "A minimum ablation should compare 64 vs 32 vs 16 with the same question batches and seeds, tracking:", "", 
        "- held-out reward and pass@k / best-of-k behavior;",
        "- within-question reward variance and fraction of all-equal reward groups;",
        "- GRPO advantage standard deviation, zero-advantage groups, and gradient norm;",
        "- parse-failure rate and cost per effective (valid, non-duplicate) rollout;",
        "- final benchmark confidence intervals across multiple training seeds.",
        "", "## Per-question results", "", (
        "`Struct` and `Sem` are mean valid-valid pairwise scores. `Topo`/`Route` are counts of exact observed "
        "topology and topology-plus-model-routing archetypes. Coverage columns are expected coverage under random "
        f"subsampling ({args.trials} trials/question)."
    ), "", "| # | Question ID | Question | Valid | Struct | Sem | Topo | Route | 32 wf | 32 instr | 16 wf | 16 instr |", "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    ]
    for row in question_results:
        lines.append(
            f"| {row['dataset_index']} | `{row['dataset_id']}` | {short_question(row['question'])} | "
            f"{row['valid']} | {row['mean_structural']:.3f} | {row['mean_semantic']:.3f} | "
            f"{row['topology_archetypes']} | {row['route_archetypes']} | "
            f"{row['k32']['workflow_coverage_085']:.1%} | {row['k32']['instruction_coverage_090']:.1%} | "
            f"{row['k16']['workflow_coverage_085']:.1%} | {row['k16']['instruction_coverage_090']:.1%} |"
        )

    lines += ["", "## Questions with the most invalid generations", "", "| # | Question ID | Invalid / 64 | Error types |", "|---:|---|---:|---|"]
    for row in invalid_by_question[:20]:
        if row["invalid"] == 0:
            break
        errors = ", ".join(f"{k}: {v}" for k, v in row["invalid_types"].items())
        lines.append(f"| {row['dataset_index']} | `{row['dataset_id']}` | {row['invalid']} | {errors} |")

    lines += ["", "## Artifacts and reproducibility", "", 
        f"- Per-rollout measurements (all 12,928 positions): `{args.per_rollout_csv}`",
        f"- Plot-ready valid-valid pairwise similarities ({len(pairwise_results):,} pairs): `{args.pairwise_csv}`",
        f"- Full per-question metrics and simulation summaries: `{args.metrics_json}`",
        f"- Cached MiniLM instruction embeddings: `{args.embedding_cache}`",
        f"- Source data: `{args.input_dir}/rollout-*/plans.jsonl`",
        "", "Re-run:", "", "```bash", (
            f".venv/bin/python scripts/analyze_rollout_similarity.py {args.input_dir} "
            f"--report {args.report} --per-rollout-csv {args.per_rollout_csv} "
            f"--pairwise-csv {args.pairwise_csv} "
            f"--metrics-json {args.metrics_json} --embedding-cache {args.embedding_cache} "
            f"--trials {args.trials} --seed {args.seed}"
        ), "```", "", "### Important limitations", "", 
        "- The embedding model is a compact general-purpose English model, not a physics-specific entailment judge. High cosine means similar intent/topic, not logical equivalence.",
        "- Instructions were capped at 256 wordpieces; 597 of 44,271 distinct instructions (1.35%) exceeded that cap and were represented by their first 256 wordpieces.",
        "- Thresholds 0.80/0.85/0.90 are sensitivity points, not externally calibrated definitions of a distinct strategy.",
        "- The 64 observed samples are treated as the reference population. Unseen rare strategies are therefore not measurable here.",
        "- Random subset simulation estimates coverage under exchangeable sampling; it does not model training dynamics.",
        "- Exact topology and route archetypes can overstate meaningful diversity when two syntactically different plans implement the same reasoning strategy.",
    ]
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
