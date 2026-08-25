"""Run LangGraph or CrewAI research workflows."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .crewai.crew_setup import build_crew
from .langgraph.graph_builder import build_graph, run_or_resume, save_mermaid

PROJECT_ROOT = Path(__file__).resolve().parents[2]
env_artifact = os.getenv("ARTIFACT_DIR")
if env_artifact and not env_artifact.startswith("/app"):
    DEFAULT_ARTIFACT_DIR = Path(env_artifact) / "module_07"
else:
    DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "module_07"

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", choices=["langgraph", "crewai"], required=True)
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--thread-id", default="research-session")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.engine == "langgraph":
        graph = build_graph(args.output / "checkpoints.sqlite")
        save_mermaid(graph, args.output / "workflow.mmd")
        result = run_or_resume(graph, args.ticker, args.thread_id)
        (args.output / f"{args.ticker}_langgraph.md").write_text(
            result.get("final_report", result.get("draft_report", str(result))),
            encoding="utf-8",
        )
    else:
        result = build_crew(args.ticker).kickoff(inputs={"ticker": args.ticker})
        (args.output / f"{args.ticker}_crewai.md").write_text(str(result), encoding="utf-8")

if __name__ == "__main__":
    main()