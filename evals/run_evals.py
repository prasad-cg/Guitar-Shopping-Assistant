"""
Evaluation Runner
Run answer and prompt evaluations from the command line.

Usage:
    python -m evals.run_evals --type answer     # Evaluate system answers
    python -m evals.run_evals --type prompt      # Evaluate agent prompts
    python -m evals.run_evals --type all         # Run both evaluations
"""
import sys
import os
import json
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evals.answer_evaluator import AnswerEvaluator
from evals.prompt_evaluator import PromptEvaluator


# ──────────────────────────────────────────────────────────────────────────
# Sample test cases for answer evaluation
# ──────────────────────────────────────────────────────────────────────────
SAMPLE_TEST_CASES = [
    {
        "query": "I'm a beginner looking for an acoustic guitar under $500 for folk music",
        "description": "Beginner recommendation query",
    },
    {
        "query": "What's the difference between a Stratocaster and a Telecaster?",
        "description": "Information comparison query",
    },
    {
        "query": "Can you give me a deal on a Les Paul?",
        "description": "Price negotiation query",
    },
    {
        "query": "I want an electric guitar for jazz, budget around $1000",
        "description": "Genre-specific recommendation",
    },
    {
        "query": "What guitars do you have for professional players?",
        "description": "Skill-level information query",
    },
]


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_score_bar(score: float, max_score: float = 100):
    """Print a visual score bar."""
    filled = int((score / max_score) * 30)
    bar = "█" * filled + "░" * (30 - filled)
    color_label = "🟢" if score >= 75 else "🟡" if score >= 50 else "🔴"
    print(f"  {color_label} [{bar}] {score}/{max_score}")


def run_answer_evaluation():
    """Run answer evaluation by generating live responses from the system."""
    print_header("ANSWER EVALUATION")
    print("\nGenerating responses from the multi-agent system and evaluating them...\n")

    from agents.orchestrator import GuitarShoppingOrchestrator

    orchestrator = GuitarShoppingOrchestrator()
    evaluator = AnswerEvaluator()

    test_cases_with_responses = []

    for i, case in enumerate(SAMPLE_TEST_CASES, 1):
        print(f"  [{i}/{len(SAMPLE_TEST_CASES)}] Processing: {case['description']}...")
        result = orchestrator.process_customer_query(case["query"])
        test_cases_with_responses.append({
            "query": case["query"],
            "response": result["final_response"],
        })

    print("\n  Evaluating responses...\n")
    eval_results = evaluator.evaluate_batch(test_cases_with_responses)

    # Print individual results
    for i, result in enumerate(eval_results["results"], 1):
        if "error" in result:
            print(f"\n  ❌ Test Case {i}: Evaluation failed")
            continue

        print(f"\n  ─── Test Case {i}: {result['query'][:60]}...")
        print_score_bar(result["overall_score"])

        dims = result["dimensions"]
        for dim_name in ["relevance", "completeness", "groundedness", "helpfulness", "clarity"]:
            dim = dims.get(dim_name, {})
            score = dim.get("score", 0)
            reason = dim.get("reason", "")
            emoji = "✅" if score >= 4 else "⚠️" if score >= 3 else "❌"
            print(f"    {emoji} {dim_name.capitalize():15s} {score}/5 — {reason}")

    # Print aggregate
    if eval_results["aggregate"]:
        agg = eval_results["aggregate"]
        print_header("ANSWER EVALUATION SUMMARY")
        print(f"\n  Total Test Cases: {agg['total_evaluated']}")
        print(f"  Failed Evaluations: {agg['total_failed']}")
        print(f"\n  Overall Average Score:")
        print_score_bar(agg["overall_avg_score"])
        print(f"\n  Dimension Averages (out of 5):")
        for dim, avg in agg["dimension_averages"].items():
            emoji = "✅" if avg >= 4 else "⚠️" if avg >= 3 else "❌"
            print(f"    {emoji} {dim.capitalize():15s} {avg}/5")

    return eval_results


def run_prompt_evaluation():
    """Run prompt evaluation on all agent system prompts."""
    print_header("PROMPT EVALUATION")
    print("\nEvaluating agent system prompts for effectiveness...\n")

    evaluator = PromptEvaluator()
    results = evaluator.evaluate_all_agents()

    for result in results["results"]:
        if "error" in result:
            print(f"\n  ❌ {result.get('agent_name', 'Unknown')}: Evaluation failed")
            continue

        print(f"\n  ─── {result['agent_name']} ───")
        print_score_bar(result["overall_score"])

        dims = result["dimensions"]
        for dim_name in ["role_clarity", "constraint_effectiveness", "task_alignment",
                         "tone_guidance", "completeness", "conciseness"]:
            dim = dims.get(dim_name, {})
            score = dim.get("score", 0)
            reason = dim.get("reason", "")
            emoji = "✅" if score >= 4 else "⚠️" if score >= 3 else "❌"
            label = dim_name.replace("_", " ").capitalize()
            print(f"    {emoji} {label:25s} {score}/5 — {reason}")

        if result.get("suggestions"):
            print(f"\n    💡 Suggestions:")
            for suggestion in result["suggestions"]:
                print(f"       • {suggestion}")

    # Aggregate
    if results["aggregate"]:
        agg = results["aggregate"]
        print_header("PROMPT EVALUATION SUMMARY")
        print(f"\n  Agents Evaluated: {agg['total_evaluated']}")
        print(f"\n  Overall Average Score:")
        print_score_bar(agg["overall_avg_score"])
        if agg["strongest_agent"]:
            print(f"\n  🏆 Strongest Prompt: {agg['strongest_agent']}")
        if agg["weakest_agent"]:
            print(f"  ⚠️  Weakest Prompt:   {agg['weakest_agent']}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Run evaluation framework")
    parser.add_argument(
        "--type",
        choices=["answer", "prompt", "all"],
        default="all",
        help="Type of evaluation to run (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Save results to a JSON file",
    )

    args = parser.parse_args()

    print("\n🎸 Guitar Shopping Assistant — Evaluation Framework")
    print("─" * 70)

    all_results = {}

    if args.type in ("prompt", "all"):
        all_results["prompt_evaluation"] = run_prompt_evaluation()

    if args.type in ("answer", "all"):
        all_results["answer_evaluation"] = run_answer_evaluation()

    # Save to file if requested
    if args.output:
        # Make results JSON-serializable
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\n  📄 Results saved to: {args.output}")

    print("\n" + "=" * 70)
    print("  Evaluation complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
