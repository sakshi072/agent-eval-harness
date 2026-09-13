"""
Run this ONCE to generate the golden dataset and save it to disk - not
regenerated on every eval run. Each test case costs real LLM calls
(entity/theme extraction during graph-building, plus question/answer
synthesis) - treat generation as a one-time cost, reuse the saved file
for every subsequent eval run against it.
 
Usage:
    poetry run python scripts/generate_dataset.py \
        --source-dir /path/to/raw/markdown/source/files \
        --testset-size 30 \
        --output golden_dataset.json
 
Start with a small --testset-size (20-30) to confirm the generated
questions are actually good before committing to a larger, more
expensive run - per the cost-control discussion, corpus size and
testset_size are the two real cost levers here, not something to
maximize on the first attempt.
 
NOT LIVE-VERIFIED: dataset.to_pandas() is written as the most commonly
documented serialization method across RAGAS versions, but wasn't
confirmed by direct introspection in this session (see
dataset/generator.py's docstring for why). If this line errors, check
your installed RAGAS version's actual Testset object methods - there may
be a more direct .to_jsonl() or .save() method depending on version.
Saving as JSON rather than CSV deliberately: the "contexts" field is a
list of strings per row, which CSV would flatten into a single
stringified cell (fragile to reload correctly); JSON preserves it
natively.
"""
import argparse
import json
import logging

from eval_harness.dataset.generator import generate_golden_dataset

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Generate a RAGAS synthetic golden dataset from raw source documents"
    )
    parser.add_argument("--source-dir", required=True, help="Directory containing raw .md source documents")
    parser.add_argument(
        "--testset-size", type=int, default=30,
        help="Number of test cases to generate - start small (20-30), each one costs real LLM calls",
    )
    parser.add_argument("--output", default="golden_dataset.json", help="Where to save the generated dataset")
    args = parser.parse_args()

    logger.info(f"Generating dataset: source_dir={args.source_dir}, testset_size={args.testset_size}")

    dataset = generate_golden_dataset(source_dir=args.source_dir, testset_size=args.testset_size)

    df = dataset.to_pandas()
    records = df.to_dict(orient="records")

    with open(args.output, "w") as f:
        json.dump(records, f, indent=2, default=str)

    logger.info(f"Saved {len(records)} test case(s) to {args.output}")
    print(f"\nSaved {len(records)} test case(s) to {args.output}")

    if records:
        print(f"\nFirst test case, for a quick sanity check:")
        print(json.dumps(records[0], indent=2, default=str)[:500])

if __name__ == "__main__":
    main()
    
