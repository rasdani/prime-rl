from typing import Callable, Literal

from zeroband.inference.genesys.ascii_tree_formatting import compute_reward as compute_ascii_tree_reward
from zeroband.inference.genesys.code import evaluate_code
from zeroband.inference.genesys.code_output_prediction import verify_code_output_prediction
from zeroband.inference.genesys.git_diff import compute_git_diff_reward
from zeroband.inference.genesys.ifeval import verify_ifeval
from zeroband.inference.genesys.math import compute_math_reward
from zeroband.inference.genesys.pydantic_json_adherance import validate_pydantic_json
from zeroband.inference.genesys.reasoning_gym import verify_reasoning_gym
from zeroband.inference.genesys.reverse_text import reverse_text
from zeroband.inference.genesys.unscramble_sentence import compute_reward as compute_unscramble_reward

TaskType = Literal[
    "verifiable_math",
    "prime_rl_code",
    "reasoning_gym",
    "code_output_prediction",
    "reverse_text",
    "unscramble_sentence",
    "ascii_tree_formatting",
    "pydantic_adherance",
    "ifeval",
    "git_diff",
]


def get_reward_function(task_type: TaskType) -> Callable[[str, dict], float]:
    try:
        return _REWARD_FUNCTIONS[task_type]
    except KeyError:
        raise ValueError(f"Invalid task type: {task_type}")


_REWARD_FUNCTIONS: dict[TaskType, Callable] = {
    "verifiable_math": compute_math_reward,
    "prime_rl_code": evaluate_code,
    "reasoning_gym": verify_reasoning_gym,
    "code_output_prediction": verify_code_output_prediction,
    "reverse_text": reverse_text,
    "unscramble_sentence": compute_unscramble_reward,
    "ascii_tree_formatting": compute_ascii_tree_reward,
    "pydantic_adherance": validate_pydantic_json,
    "ifeval": verify_ifeval,
    "git_diff": compute_git_diff_reward,
}
