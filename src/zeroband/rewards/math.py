from math_verify import parse, verify
from typing import Dict

def compute_math_reward(completion: str, verification_info: Dict):
    split_response = completion.split("</think>")    
    
    # format error
    if len(split_response) == 1:
        return -1
    
    try:
        response = parse(split_response[1])
        gold = parse(verification_info["ground_truth"])
        correct = verify(gold, response)
        
        if correct:
            return 1
        else:
            return -1
    
    except Exception as e:
        print(f"error verifying math: {e}")
        return -1
            