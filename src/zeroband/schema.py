from jaxtyping import Float, Int
from pydantic import BaseModel
from torch import Tensor


class ActorOutput(BaseModel):
    logits: Int[Tensor, "batch_size seq_len vocab_size"]
    rewards: Float[Tensor, "batch_size"]
    ref_log_probs: Float[Tensor, "batch_size"] | None = None
