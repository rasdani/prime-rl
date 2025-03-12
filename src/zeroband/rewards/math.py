from math_verify import parse, verify
from typing import Dict
from zeroband.logger import get_logger


def compute_math_reward(completion: str, verification_info: Dict):
    logger = get_logger()

    split_response = completion.split("</think>")

    # format error
    if len(split_response) == 1:
        return 0

    try:
        response = parse(split_response[1])
        gold = parse(verification_info["ground_truth"])
        correct = verify(gold, response)

        if correct:
            return 1
        else:
            return 0

    except Exception as e:
        logger.warning(f"error verifying math response: {e}. Returning negative reward")
        return 0
