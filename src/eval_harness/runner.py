"""
Generic runner core - queries the target adapter exactly ONCE per golden
dataset entry, producing (entry, response) pairs that every metric
scorer consumes.
 
This function has no idea what RAGAS or DeepEval are - it only depends
on the generic TargetAdapter interface, matching the reusability design
from this project's original scoping.
"""
import logging
from typing import List, Dict, Tuple
from eval_harness.adapters.base import TargetAdapter, TargetResponse

logger = logging.getLogger(__name__)

async def collect_responses(golden_dataset:List[Dict], target:TargetAdapter) -> List[Tuple[Dict, TargetResponse]]:
    collected = []
    
    for entry in golden_dataset:
        question = entry["user_input"]
        logger.info(f"Querying target for: {question[:60]!r}")
        response = await target.ask(question=question)
        collected.append((entry, response))
    return collected