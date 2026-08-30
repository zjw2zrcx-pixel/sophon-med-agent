from __future__ import annotations

import re

from ..CallRoute.parser.parser import Command
from .prompt_config import BENCHMARK_SYSTEM_PROMPT
from .voice import VoiceMode


class BenchmarkMode(VoiceMode):
    """Script-driven Agent mode with a deterministic compact workflow.

    The model prompt is intentionally left untouched.  This adapter takes
    ownership of routing and tool sequencing, while the Harness validates each
    directly scheduled business tool and keeps trajectories deterministic.
    """

    name = "Benchmark"

    def get_system_prompt(self) -> str:
        return BENCHMARK_SYSTEM_PROMPT

    def get_compact_workflow_plan(self, user_input: str):
        # Benchmark cases require a deterministic plan even for general tasks;
        # production Voice uses the shared builder only for confident intents.
        return self._build_compact_workflow_plan(
            user_input, atomic_navigation=True
        )

    def get_scheduled_workflow_command(self, user_input: str):
        return self._get_scheduled_compact_command(
            user_input, atomic_navigation=True
        )

    def is_compact_navigation_announcement_terminal(
        self, command, call_result, execution_state
    ) -> bool:
        step = execution_state.active_step
        return bool(
            command.params.get("announcement")
            and step is None
            and execution_state.status == "GOAL_SATISFIED"
        )

    def adapt_compact_terminal_action(self, command, execution_state):
        """Treat FINISH as answer content only when the workflow is ready."""
        if command.type != "tool_call" or command.name != "act":
            return None
        params = command.params
        if str(params.get("action_type", "")).upper() != "FINISH":
            return None
        text = str(params.get("response", "") or "").strip()
        step = execution_state.active_step
        terminal_ready = bool(
            (step is not None and step.preferred_tool == "speak")
            or (
                step is None
                and execution_state.status == "GOAL_SATISFIED"
                and not execution_state.active_step_id
            )
        )
        if not text or not terminal_ready:
            return None
        return Command(
            type="tool_call", name="speak", params={"text": text},
            raw="[agent_action_repaired:finish_to_speak]\n" + command.raw,
            confidence=command.confidence,
        )

    def adapt_compact_truncated_terminal_output(self, response_text, execution_state):
        """Recover a cut-off final speak envelope without asking the model again.

        This path is deliberately narrow: it applies only while the canonical
        workflow is at its final speak step and the output has an unfinished
        XML tool wrapper.  The raw model output remains in its call record.
        """
        step = execution_state.active_step
        if step is None or step.preferred_tool != "speak":
            return None
        if "<tool>" not in response_text or "</tool>" in response_text:
            return None
        if not re.search(r'"(?:tool_call|action_type)"\s*:', response_text):
            return None
        match = re.search(r'"text"\s*:\s*"(.*)$', response_text, re.DOTALL)
        if match is None:
            return None
        text = match.group(1).rstrip()
        # When generation stops immediately before </tool>, the captured
        # value can include the already-complete JSON suffix (`"}}`).
        text = re.sub(r'(?<!\\)"\s*\}+\s*$', '', text)
        # A token boundary can leave one dangling escape; omit only that
        # incomplete byte and decode the common JSON escapes conservatively.
        text = re.sub(r'\\(?:u[0-9a-fA-F]{0,3})?$', '', text)
        text = text.replace(r'\n', '\n').replace(r'\"', '"').replace(r'\\', '\\')
        text = text.strip().rstrip('，、；：')
        if len(text) < 4:
            return None
        return Command(
            type="tool_call", name="speak", params={"text": text},
            raw="[agent_action_repaired:truncated_terminal_to_speak]\n" + response_text,
            confidence=1.0,
        )
