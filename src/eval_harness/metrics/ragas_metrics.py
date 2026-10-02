"""
Runs RAGAS's core RAG-quality metrics against real agent responses.
"""
import logging
from typing import List, Dict, Tuple

from eval_harness.adapters.base import TargetResponse
from eval_harness.core.settings import settings

logger = logging.getLogger(__name__)

def run_ragas_evaluation(collected: List[Tuple[Dict, TargetResponse]]):
    """
    For each golden entry: calls the target adapter for a REAL agent
    response, assembles a RAGAS sample (question + real answer + real
    retrieved_contexts + the golden reference answer), then scores the
    whole batch with RAGAS's LLM-judge metrics.
 
    Uses the golden dataset's own "reference" field (RAGAS's synthesized
    ground-truth answer) for context recall / semantic comparison - NOT
    reference_contexts, since that's what the SOURCE documents looked
    like during generation, not what the live agent actually retrieved
    for THIS specific run. Comparing the agent's real answer against a
    golden reference answer, using the agent's OWN real retrieved
    contexts, is what makes these scores about the real system's current
    behavior, not a comparison against generation-time artifacts.
    """
    from ragas import evaluate, EvaluationDataset, SingleTurnSample
    from ragas.metrics import Faithfulness, ContextPrecision, ContextRecall, AnswerRelevancy
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from langchain_openai import ChatOpenAI
    from langchain_huggingface import HuggingFaceEmbeddings

    samples = []
    for entry, response in collected:
        samples.append(
            SingleTurnSample(
                user_input=entry["user_input"],
                response=response.answer,
                retrieved_contexts=response.retrieved_contexts or [""],
                reference=entry.get("reference", "")
            )
        )

    eval_dataset = EvaluationDataset(samples=samples)

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=settings.ragas.generator_model))
    judge_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name=settings.ragas.embedding_model))

    logger.info(f"Scoring {len(samples)} sample(s) with RAGAS metrics...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[Faithfulness(), ContextPrecision(), ContextRecall(), AnswerRelevancy()],
        llm = judge_llm,
        embeddings=judge_embeddings,
    )
    return result
