"""Abstract base class shared by all sub-agents."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mas.matlab_bridge import MatlabBridge


class BaseAgent(ABC):
    def __init__(self) -> None:
        self._bridge: MatlabBridge = MatlabBridge.get()

    @property
    def matlab_available(self) -> bool:
        return self._bridge.available

    @abstractmethod
    def analyze(self, *args: Any, **kwargs: Any) -> dict:
        """Run the agent and return a feature dict."""
