"""
Main Entry Point
Can be used to run the system or evaluation.

Usage:
  python main.py --mode cli           # Run CLI interface
  python main.py --mode web           # Run web interface
  python main.py --mode evaluate      # Run evaluation
"""

import argparse
import asyncio
import sys
from pathlib import Path


def run_cli():
    """Run CLI interface."""
    from src.ui.cli import main as cli_main
    cli_main()


def run_web():
    """Run web interface."""
    import subprocess
    print("Starting Streamlit web interface...")
    subprocess.run(["streamlit", "run", "src/ui/streamlit_app.py"])


async def run_evaluation():
    """Run full batch evaluation using SystemEvaluator + LLM-as-a-Judge."""
    import yaml
    from dotenv import load_dotenv
    from src.autogen_orchestrator import AutoGenOrchestrator
    from src.evaluation.evaluator import SystemEvaluator

    load_dotenv()

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("=" * 70)
    print("MULTI-AGENT SYSTEM — BATCH EVALUATION")
    print("=" * 70)

    # Initialize orchestrator
    print("\nInitializing orchestrator...")
    orchestrator = AutoGenOrchestrator(config)

    # Initialize evaluator
    evaluator = SystemEvaluator(config, orchestrator=orchestrator)

    # Choose query file (fall back to example_queries if test file missing)
    import os
    query_file = "data/example_queries.json"
    if not os.path.exists(query_file):
        print(f"Query file not found: {query_file}")
        return

    print(f"Running evaluation on: {query_file}")
    print("This may take several minutes...\n")

    # Run evaluation
    report = await evaluator.evaluate_system(query_file)

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    summary = report.get("summary", {})
    scores  = report.get("scores", {})

    print(f"\nTotal Queries : {summary.get('total_queries', 0)}")
    print(f"Successful     : {summary.get('successful', 0)}")
    print(f"Failed         : {summary.get('failed', 0)}")
    print(f"Success Rate   : {summary.get('success_rate', 0):.1%}")
    print(f"\nOverall Avg Score : {scores.get('overall_average', 0):.3f} / 1.0")

    print("\nScores by Criterion:")
    for criterion, score in scores.get("by_criterion", {}).items():
        bar = "█" * int(score * 20)
        print(f"  {criterion:<20} {score:.3f}  {bar}")

    best  = report.get("best_result")
    worst = report.get("worst_result")
    if best:
        print(f"\nBest  query (score {best['score']:.3f}): {best['query'][:70]}")
    if worst:
        print(f"Worst query (score {worst['score']:.3f}): {worst['query'][:70]}")

    print("\nFull results saved to outputs/")
    print("=" * 70)


def run_autogen():
    """Run AutoGen example."""
    import subprocess
    print("Running AutoGen example...")
    subprocess.run([sys.executable, "example_autogen.py"])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research Assistant"
    )
    parser.add_argument(
        "--mode",
        choices=["cli", "web", "evaluate", "autogen"],
        default="autogen",
        help="Mode to run: cli, web, evaluate, or autogen (default)"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        run_web()
    elif args.mode == "evaluate":
        asyncio.run(run_evaluation())
    elif args.mode == "autogen":
        run_autogen()


if __name__ == "__main__":
    main()
