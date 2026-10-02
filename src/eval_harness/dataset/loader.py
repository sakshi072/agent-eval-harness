import json
from typing import List, Dict

def load_golden_dataset(path:str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)