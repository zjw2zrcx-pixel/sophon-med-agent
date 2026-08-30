from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolResult:
    success: bool
    data: str = ""
    error: str = ""
    error_type: str = ""
    empty: bool = False
    retryable: bool = False
    recovery_hint: str = ""
    facts: Dict[str, Any] = field(default_factory=dict)
    # Structured, non-authoritative telemetry for traces and debugging.  It is
    # deliberately excluded from Harness facts and model prompt history.
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    # Process-local handoff data (for example recorded PCM).  It must never be
    # serialized into prompts, logs or trajectory JSON.
    transient: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class ToolContext:
    session: Optional[object] = None
    ros_bridge: Optional[object] = None
    image: Optional[str] = None
    scene_description: str = ""
    extra: dict = field(default_factory=dict)


class Tool(ABC):
    name: str = ""
    description: str = ""
    param_schema: Dict[str, str] = field(default_factory=dict)
    modes: List[str] = field(default_factory=list)
    harness_metadata: Dict[str, Any] = {
        "effect": "READ",
        "idempotent": True,
        "produces": [],
        "invalidates": [],
        "retry": {"max_attempts": 2},
    }

    @abstractmethod
    async def call(self, params: Dict[str, Any], context: ToolContext) -> ToolResult:
        ...

    def get_description_text(self) -> str:
        params_text = ""
        if self.param_schema:
            parts = []
            for pname, pdesc in self.param_schema.items():
                parts.append(f"    {pname}: {pdesc}")
            params_text = "\n".join(parts)
        else:
            params_text = "    (无参数)"

        metadata = self.get_harness_metadata()
        harness_text = ""
        if metadata.produces:
            harness_text = "\nHarness事实: " + ", ".join(metadata.produces)
        return (
            f"### {self.name}\n"
            f"描述: {self.description}\n"
            f"参数:\n{params_text}"
            f"{harness_text}"
        )

    def get_harness_metadata(self):
        from ..Harness import ToolMetadata

        return ToolMetadata.from_dict(self.harness_metadata)
