import argparse
from dotenv import load_dotenv
import asyncio
import json
import logging

from eval_harness.core.settings import settings
from eval_harness.runner import collect_responses
from eval_harness.metrics.ragas_metrics import run_ragas_evaluation
from eval_harness.metrics.deepeval_metrics import run_tool_correctness_evaluation
from eval_harness.dataset.loader import load_golden_dataset
from eval_harness.adapters.knowledge_agent_adapter import KnowledgeAgentAdapter

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

async def main():
    parser = argparse.ArgumentParser(description="Run RAGAS + DeepEval scoring against a golden dataset")
    parser.add_argument("--dataset", required=True, help="Path to a golden dataset JSON file")
    args = parser.parse_args()

    golden_dataset = load_golden_dataset(args.dataset)
    logger.info(f"Loaded {len(golden_dataset)} golden test case(s) from {args.dataset}")

    target = KnowledgeAgentAdapter(base_url=settings.agent.base_url)

    logger.info("=== Querying target agent (once per question, shared by both scorers) ===")
    collected = await collect_responses(golden_dataset, target)

    # Split BEFORE any scoring - both citation-gate refusals and hard
    # scope-refusals collapse to the same signal on TargetResponse: empty
    # retrieved_contexts. Feeding these into RAGAS produced the inconsistent
    # scores just observed (two structurally identical empty-context cases
    # scoring context_recall at 1.0 and 0.0) - RAGAS has no concept of "the
    # system correctly/incorrectly declined," so it shouldn't be asked to
    # score these at all.
    grounded = [(e,r) for e,r in collected if r.retrieved_contexts]
    ungrounded = [(e,r) for e,r in collected if not r.retrieved_contexts]

    logger.info(f"=== Grounding: {len(grounded)}/{len(collected)} question(s) had real retrieved content ===")
    for entry, response in ungrounded:
        print(f"  [UNGROUNDED] {entry['user_input'][:60]!r} - tools_called={response.tools_called}, "
            f"answer={response.answer[:80]!r}")

    logger.info("=== Running DeepEval tool-correctness scoring ===")
    tool_results = run_tool_correctness_evaluation(collected)

    for r in tool_results:
        status = "PASS" if r["success"] else "FAIL"
        logger.info(f"  [{status}] {r['question'][:60]!r} - called={r['tools_called']}, score={r['score']}")

    logger.info("\n=== Running RAGAS quality scoring (faithfulness, context precision/recall, answer relevancy) ===")
    if grounded:
        ragas_result = run_ragas_evaluation(grounded)
        logger.info(f"Aggregate scores: {ragas_result}")
        ragas_df = ragas_result.to_pandas()
        per_question_scores = ragas_df.to_dict(orient="records")
    else:
        logger.warning("No grounded questions to score - skipping RAGAS entirely")
        ragas_result = None
        per_question_scores = []

    with open("eval_results.json", "w") as f:
        json.dump({
            "tool_correctness": tool_results,
            "grounding_summary": {
                "total": len(collected),
                "grounded": len(grounded),
                "ungrounded": len(ungrounded),
                "ungrounded_questions": [e["user_input"] for e, _ in ungrounded],
            },
            "ragas_aggregate_scores": str(ragas_result) if ragas_result else None,
            "ragas_per_question_scores": per_question_scores,
        }, f, indent=2, default=str)

    logger.info("\nSaved combined results to eval_results.json")

if __name__ == "__main__":
    asyncio.run(main())