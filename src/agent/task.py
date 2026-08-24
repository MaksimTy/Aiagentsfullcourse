import os
from dataclasses import dataclass
from typing import Callable

# --------------------------------------------------------------------------- #
# Задача
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Task:
    """Иммутабельна: агент не имеет права переписать задачу под себя."""
    prompt: str
    postcondition: Callable[[str], tuple[bool, str]] | None = None
    max_steps: int = int(os.environ.get("AGENT_MAX_STEPS", 16))
    max_seconds: float = float(os.environ.get("AGENT_MAX_SECONDS", 300))
    max_tokens: int = int(os.environ.get("AGENT_MAX_TOKENS", 200_000))
    max_cost_usd: float = (os.environ.get("AAGENT_MAX_COST_USD", 0.50))