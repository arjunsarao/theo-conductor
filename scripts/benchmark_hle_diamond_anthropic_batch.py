#!/usr/bin/env python3
"""Compatibility entry point for Anthropic batch evaluation."""
import asyncio
from theo_conductor.anthropic_benchmark import main

if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
