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
import asyncio

logger = logging.getLogger(__name__)

async def collect_responses(golden_dataset:List[Dict], target:TargetAdapter, max_concurrent:int=4) -> List[Tuple[Dict, TargetResponse]]:
    sem = asyncio.Semaphore(max_concurrent)

    async def one(entry):
        async with sem:
            try:
                return entry, await target.ask(entry["user_input"])
            except Exception as e:
                logger.error(f"FAILED {entry['user_input'][:60]!r}: {e!r}")
                return entry, None

    results = await asyncio.gather(*[one(e) for e in golden_dataset])
    ok = [(e, r) for e, r in results if r is not None]
    logger.info(f"Collected {len(ok)}/{len(golden_dataset)} responses")
    return ok
    
    # collected = []
    
    # for entry in golden_dataset:
    #     question = entry["user_input"]
    #     logger.info(f"Querying target for: {question[:60]!r}")
    #     response = await target.ask(question=question)
    #     collected.append((entry, response))
    # return collected