from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from colorama import just_fix_windows_console
from rich.console import Console as RichConsole
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .grpo import parse_conductor_json
from .models.registry import ModelRegistry
from .prompt import build_conductor_prompt, build_conductor_response_format
from .runner import Runner
from .schema import RunResult, Step, StepOutput, Task
from .scheduler import topological_sort


class CliConsole:
    def __init__(self, *, color: bool = True, quiet: bool = False) -> None:
        self.quiet = quiet
        color_system = "auto" if color else None
        self.out = RichConsole(color_system=color_system, highlight=False)
        self.err = RichConsole(stderr=True, color_system=color_system, highlight=False)

    def log(self, marker: str, message: str, *, color: str = "cyan") -> None:
        if not self.quiet:
            self.err.print(f"[{color} bold]{marker}[/] {message}")

    def plan(self, task: Task) -> None:
        if self.quiet:
            return

        table = Table(show_header=True, header_style="bold cyan", box=None, padding=(0, 1))
        table.add_column("Layer", justify="right", style="cyan", no_wrap=True)
        table.add_column("Step", style="green")
        table.add_column("Model", style="dim")

        for layer_number, layer in enumerate(topological_sort(task), start=1):
            for index, step in enumerate(layer):
                table.add_row(str(layer_number) if index == 0 else "", step.step_id, str(step.model_id))

        self.err.print(Panel(table, title="[bold]Execution plan[/]", border_style="cyan", expand=False))

    def step_event(self, event: str, step: Step, output: StepOutput | None) -> None:
        if self.quiet:
            return
        if event == "started":
            self.err.print(
                Text.assemble(
                    ("·", "yellow bold"),
                    " ",
                    step.step_id,
                    (f" [{step.model_id}]", "dim"),
                )
            )
            return
        latency = f" in {output.latency_ms / 1000:.1f}s" if output and output.latency_ms else ""
        self.log("✓", f"{step.step_id}{latency}", color="green")

    def result(self, result: RunResult, elapsed: float) -> None:
        final_step_id = result.task.workflow[-1].step_id
        final = result.outputs.get(final_step_id)
        if final is None:
            raise ValueError(f"Workflow completed without output from its last step {final_step_id!r}")

        if self.quiet:
            self.out.print(final.text.strip(), markup=False)
            return

        self.out.print(Panel(Text(final.text.strip()), title="[bold green]Answer[/]", border_style="green"))
        tokens = sum(int(output.usage.get("total_tokens", 0)) for output in result.outputs.values() if output.usage)
        token_text = f"  •  {tokens:,} tokens" if tokens else ""
        self.err.print(f"[dim]✓ {len(result.outputs)} steps  •  {elapsed:.1f}s{token_text}[/]")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="theo-conductor",
        description="Turn a question into a multi-model workflow and run it.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("task", nargs="?", type=Path, metavar="FILE", help="Run an existing workflow JSON file.")
    source.add_argument("-q", "--question", help="Ask the conductor to create and run a workflow.")
    parser.add_argument(
        "--conductor-model",
        metavar="MODEL_ID",
        help="Model that creates the workflow (default: first configured model).",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path("configs"),
        help="Directory containing model YAML files (default: configs).",
    )
    parser.add_argument("--json", action="store_true", help="Print the complete run result as JSON.")
    parser.add_argument("--quiet", action="store_true", help="Print only the answer.")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors.")
    return parser.parse_args(argv)


def load_task(path: Path) -> Task:
    with path.open(encoding="utf-8") as task_file:
        data = json.load(task_file)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return Task.from_dict(data)


async def create_task(question: str, registry: ModelRegistry, conductor_model: str | None) -> Task:
    if conductor_model is None:
        try:
            conductor_model = next(iter(registry._models))
        except StopIteration as exc:
            raise ValueError("No models are configured") from exc

    response = await registry.get(conductor_model).client.generate(
        instruction=build_conductor_prompt(question, registry),
        question=question,
        context={},
        max_tokens=2048,
        temperature=0.1,
        response_format=build_conductor_response_format(registry),
    )
    return parse_conductor_json(
        response.text,
        question=question,
        model_registry=registry,
    )


async def async_main(argv: Sequence[str] | None = None) -> RunResult:
    args = parse_args(argv)
    just_fix_windows_console()
    console = CliConsole(color=not args.no_color, quiet=args.quiet or args.json)
    registry = ModelRegistry.from_config_dir(args.config_dir)

    if args.question is not None:
        model_id = args.conductor_model or next(iter(registry._models), None)
        console.log("◆", f"Planning workflow with {model_id}")
        task = await create_task(args.question, registry, args.conductor_model)
        console.log("✓", f"Created a {task.difficulty.value} {task.task_type} workflow", color="green")
    else:
        task = load_task(args.task)
        console.log("✓", f"Loaded workflow from {args.task}", color="green")

    console.plan(task)
    console.log("▶", f"Running {len(task.workflow)} worker steps")
    started = time.perf_counter()
    result = await Runner(model_registry=registry, event_handler=console.step_event).run(task)
    elapsed = time.perf_counter() - started

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        console.result(result, elapsed)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    try:
        asyncio.run(async_main(argv))
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
