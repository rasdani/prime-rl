import os

world_info = None


class WorldInfo:
    """This class parse env var about torch world into class variables."""

    world_size: int
    rank: int
    local_rank: int
    local_world_size: int

    def __init__(self):
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.nnodes = self.world_size // self.local_world_size

    def __repr__(self):
        return f"WorldInfo(world_size={self.world_size}, rank={self.rank}, local_rank={self.local_rank}, local_world_size={self.local_world_size}, nnodes={self.nnodes}, device_placement={self.device_placement})"

    def json(self) -> dict[str, int | str]:
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "nnodes": self.nnodes,
        }


def get_world_info() -> WorldInfo:
    """
    Return a WorldInfo singleton.
    """
    global world_info
    if world_info is None:
        world_info = WorldInfo()
    return world_info
