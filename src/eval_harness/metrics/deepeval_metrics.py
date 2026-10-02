"""
Scores whether the agent called the right tool for each question -
genuinely agent-specific, not RAG-specific (a coding agent, a support
bot, anything tool-calling could reuse this same metric with a different
target adapter). RAGAS has no equivalent to this.
 
EXPECTED-TOOLS ASSUMPTION: every entry in the current golden dataset is
a genuine, on-topic content question (confirmed directly from the actual
generated output) - so expected_tools defaults to
["search_knowledge_base"] for every case. This assumption breaks if the
golden dataset is ever extended with deliberately off-topic/negative
test cases (worth doing eventually, to test the input guardrail and
citation gate specifically) - at that point, expected_tools needs to
vary per entry, not stay a single default.
"""
import logging
from typing import List, Dict, Tuple
 
from eval_harness.adapters.base import TargetResponse

logger = logging.getLogger(__name__)

DEFAULT_EXPECTED_TOOLS = ["search_knowledge_base"]

def run_tool_correctness_evaluation(
    collected: List[Tuple[Dict, TargetResponse]],
    expected_tools: List[str] = None,
):
    from deepeval.metrics import ToolCorrectnessMetric
    from deepeval.test_case import LLMTestCase, ToolCall

    expected = expected_tools or DEFAULT_EXPECTED_TOOLS
    metrics = ToolCorrectnessMetric()

    results = []
    for entry, response in collected:
        question = entry["user_input"]
        test_case = LLMTestCase(
            input=question,
            actual_output=response.answer,
            tools_called=[ToolCall(name=name) for name in response.tools_called],
            expected_tools=[ToolCall(name=name) for name in expected]
        )

        metrics.measure(test_case)
        results.append({
            "question":question,
            "answer": response.answer,
            "tools_called": response.tools_called,
            "expected_tools": expected,
            "score": metrics.score,
            "success": metrics.is_successful(),
            "reason": getattr(metrics, "reason", None),
        })
        logger.info(
            f"Tool correctness: {question[:50]!r} -> called={response.tools_called}, "
            f"expected={expected}, score={metrics.score}"
        )
 
    return results