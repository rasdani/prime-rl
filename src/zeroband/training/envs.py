from typing import TYPE_CHECKING, Any, List
import os

if TYPE_CHECKING:
    TRAINING_ENABLE_ACCEPTED_CHECK: bool = False
    PRIME_DASHBOARD_BASE_URL: str | None = None
    PRIME_DASHBOARD_AUTH_TOKEN: str | None = None

_env = {
    "TRAINING_ENABLE_ACCEPTED_CHECK": lambda: os.getenv("TRAINING_ENABLE_ACCEPTED_CHECK", "false").lower() in ["true", "1", "yes", "y"],
    "PRIME_API_BASE_URL": lambda: os.getenv("PRIME_API_BASE_URL"),
    "PRIME_DASHBOARD_AUTH_TOKEN": lambda: os.getenv("PRIME_DASHBOARD_AUTH_TOKEN"),
}


def __getattr__(name: str) -> Any:
    if name not in _env:
        raise AttributeError(f"Invalid environment variable: {name}")
    return _env[name]()


def __dir__() -> List[str]:
    return list(_env.keys())
