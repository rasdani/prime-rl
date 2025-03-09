from zeroband.rewards.math import compute_math_reward
from zeroband.rewards.kod_code import verify_kod_code

REWARD_FUNCTIONS = {
    "verifiable_math": compute_math_reward,
    "kod_code": verify_kod_code
}