from __future__ import annotations

import unittest

from agents.Harness import (
    ActionValidator,
    ExecutionState,
    TaskPlan,
    ToolMetadata,
    TriState,
)
from agents.API.session import PromptSlots
from agents.CallRoute.parser import Command
from agents.Modes.base import ModeBase


def service_plan():
    return TaskPlan.from_payload({
        "plan_id": "plan_service",
        "goal": "ensure_service_running",
        "success_conditions": [{
            "fact": "service.status", "operator": "eq",
            "value": "running", "require_valid": True,
        }],
        "steps": [
            {
                "step_id": "s1", "goal": "inspect_service_status",
                "preferred_tool": "get_service_status", "depends_on": [],
            },
            {
                "step_id": "s2", "goal": "start_service",
                "preferred_tool": "start_service", "depends_on": ["s1"],
                "condition": {
                    "fact": "service.status", "operator": "neq", "value": "running",
                },
            },
            {
                "step_id": "s3", "goal": "verify_service_running",
                "preferred_tool": "get_service_status", "depends_on": ["s2"],
                "verification": True,
            },
        ],
    })


READ = ToolMetadata(
    effect="READ", produces=("service.status",), max_attempts=2,
)
WRITE = ToolMetadata(
    effect="WRITE", invalidates=("service.status",),
    verification_tool="get_service_status", max_attempts=2,
)


class HarnessStateTests(unittest.TestCase):
    def test_success_condition_short_circuits_remaining_steps(self):
        state = ExecutionState.create(service_plan())
        state.apply_tool_result(
            tool="get_service_status", arguments={"service": "A"}, metadata=READ,
            success=True, facts={"service.status": "running"}, observation="running",
        )
        self.assertEqual(state.status, "GOAL_SATISFIED")
        self.assertEqual(state.active_step_id, "")
        self.assertNotEqual(state.step_states["s3"].status, "ACTIVE")

    def test_write_invalidates_fact_and_epoch_allows_verification(self):
        state = ExecutionState.create(service_plan())
        validator = ActionValidator()
        args = {"service": "A"}

        self.assertEqual(state.active_step_id, "s1")
        self.assertTrue(validator.validate_tool(
            state=state, tool="get_service_status", arguments=args,
            metadata=READ, available_tools=["get_service_status", "start_service"],
        ).allowed)
        state.apply_tool_result(
            tool="get_service_status", arguments=args, metadata=READ,
            success=True, facts={"service.status": "stopped"}, observation="stopped",
        )
        self.assertEqual(state.active_step_id, "s2")
        projection = state.projection()
        self.assertEqual(projection["goal"], "ensure_service_running")
        self.assertEqual(projection["current_step"], 2)
        self.assertEqual(projection["current_step_id"], "s2")
        self.assertEqual(projection["completed_steps"][0]["result"], "stopped")
        self.assertEqual(
            projection["remaining_steps"],
            ["start_service", "verify_service_running"],
        )
        self.assertEqual(
            projection["success_condition"], 'service.status == "running"'
        )
        self.assertEqual(state.facts["service.status"].observed_epoch, 0)

        state.apply_tool_result(
            tool="start_service", arguments=args, metadata=WRITE,
            success=True, facts={}, observation="accepted",
        )
        self.assertEqual(state.world_epoch, 1)
        self.assertFalse(state.facts["service.status"].valid)
        self.assertEqual(state.active_step_id, "s3")
        self.assertTrue(validator.validate_tool(
            state=state, tool="get_service_status", arguments=args,
            metadata=READ, available_tools=["get_service_status", "start_service"],
        ).allowed)

        state.apply_tool_result(
            tool="get_service_status", arguments=args, metadata=READ,
            success=True, facts={"service.status": "running"}, observation="running",
        )
        self.assertEqual(state.status, "GOAL_SATISFIED")
        blocked = validator.validate_tool(
            state=state, tool="get_service_status", arguments=args,
            metadata=READ, available_tools=["get_service_status"],
        )
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.error_code, "GOAL_ALREADY_SATISFIED")

    def test_same_signature_same_epoch_is_never_reexecuted(self):
        plan = TaskPlan.from_payload({
            "goal": "inspect_twice",
            "steps": [
                {"step_id": "s1", "goal": "first", "preferred_tool": "read"},
                {"step_id": "s2", "goal": "verify", "preferred_tool": "read",
                 "depends_on": ["s1"], "verification": True},
            ],
        })
        state = ExecutionState.create(plan)
        state.apply_tool_result(
            tool="read", arguments={"id": 1}, metadata=ToolMetadata(),
            success=True, observation="same world",
        )
        denied = ActionValidator().validate_tool(
            state=state, tool="read", arguments={"id": 1},
            metadata=ToolMetadata(), available_tools=["read"],
        )
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.error_code, "DUPLICATE_NO_WORLD_CHANGE")

    def test_unknown_condition_is_not_true(self):
        state = ExecutionState.create(service_plan())
        self.assertEqual(state.evaluate_condition({
            "fact": "missing", "operator": "neq", "value": "running",
        }), TriState.UNKNOWN)

    def test_failure_does_not_complete_step_and_retry_is_bounded(self):
        plan = TaskPlan.from_payload({
            "goal": "lookup", "steps": [{
                "step_id": "s1", "goal": "lookup", "preferred_tool": "read",
            }],
        })
        state = ExecutionState.create(plan)
        retry = ToolMetadata(max_attempts=2, allowed_errors=("TIMEOUT",))
        state.apply_tool_result(
            tool="read", arguments={"q": "x"}, metadata=retry,
            success=False, error_type="TIMEOUT", retryable=True,
        )
        self.assertNotEqual(state.step_states["s1"].status, "COMPLETED")
        self.assertEqual(state.step_states["s1"].attempts, 1)
        # A changed argument can consume the second permitted attempt.
        state.apply_tool_result(
            tool="read", arguments={"q": "y"}, metadata=retry,
            success=False, error_type="TIMEOUT", retryable=True,
        )
        self.assertEqual(state.step_states["s1"].status, "FAILED")
        self.assertEqual(state.status, "BLOCKED")

    def test_transient_error_allows_one_exact_retry_within_budget(self):
        plan = TaskPlan.from_payload({
            "goal": "lookup", "steps": [{
                "step_id": "s1", "goal": "lookup", "preferred_tool": "read",
            }],
        })
        state = ExecutionState.create(plan)
        retry = ToolMetadata(max_attempts=2, allowed_errors=("TIMEOUT",))
        state.apply_tool_result(
            tool="read", arguments={"q": "x"}, metadata=retry,
            success=False, error_type="TIMEOUT", retryable=True,
        )
        allowed = ActionValidator().validate_tool(
            state=state, tool="read", arguments={"q": "x"},
            metadata=retry, available_tools=["read"],
        )
        self.assertTrue(allowed.allowed)

    def test_model_visible_state_events_are_append_only(self):
        prompt = PromptSlots()
        prompt.start_task(system="fixed", user_input="task")
        prompt.set_plan(service_plan().to_dict())
        state = ExecutionState.create(service_plan())
        first = state.append_event("PLAN_ATTACHED")
        prompt.append_execution_event(first)
        before = prompt.to_request_dict()["history"]
        first["state"]["status"] = "MUTATED_OUTSIDE"
        self.assertNotEqual(prompt.execution_events[0]["state"]["status"], "MUTATED_OUTSIDE")
        state.last_observation = {"success": False, "error_type": "TEST"}
        second = state.append_event("CONTROL_OBSERVATION")
        prompt.commit(
            command_type="tool_call", name="read", params={"q": "x"},
            model_output="raw", result="a raw current-task result",
        )
        prompt.append_execution_event(second)
        after = prompt.to_request_dict()["history"]
        self.assertIn('"index":1', before)
        self.assertTrue(after.startswith(before))
        self.assertNotIn("a raw current-task result", after)
        self.assertEqual([x["index"] for x in prompt.execution_events], [1, 2])

    def test_terminal_tool_may_produce_remaining_success_fact(self):
        plan = TaskPlan.from_payload({
            "goal": "answer",
            "success_conditions": [
                {"fact": "medical.consultation", "operator": "exists"},
                {"fact": "speech.last_text", "operator": "exists"},
            ],
            "steps": [
                {"step_id": "s1", "goal": "consult", "preferred_tool": "medical_consult"},
                {"step_id": "s2", "goal": "speak", "preferred_tool": "speak", "depends_on": ["s1"]},
            ],
        })
        state = ExecutionState.create(plan)
        state.apply_tool_result(
            tool="medical_consult", arguments={"query": "x"},
            metadata=ToolMetadata(produces=("medical.consultation",)),
            success=True, facts={"medical.consultation": "evidence"},
        )
        speak = ToolMetadata(
            effect="WRITE", terminal=True, produces=("speech.last_text",)
        )
        allowed = ActionValidator().validate_tool(
            state=state, tool="speak", arguments={"text": "answer"},
            metadata=speak, available_tools=["speak"],
        )
        self.assertTrue(allowed.allowed)

    def test_terminal_tool_cannot_skip_unrelated_success_fact(self):
        state = ExecutionState.create(service_plan())
        speak = ToolMetadata(
            effect="WRITE", terminal=True, produces=("speech.last_text",)
        )
        denied = ActionValidator().validate_tool(
            state=state, tool="speak", arguments={"text": "done"},
            metadata=speak, available_tools=["speak"],
        )
        self.assertFalse(denied.allowed)
        # The active service step owns the tool contract before the more
        # general goal-satisfaction check is reached.
        self.assertEqual(denied.error_code, "STEP_TOOL_MISMATCH")

    def test_session_terminal_speak_is_allowed_after_goal_satisfied(self):
        state = ExecutionState.create(service_plan())
        state.apply_tool_result(
            tool="get_service_status", arguments={"service": "A"}, metadata=READ,
            success=True, facts={"service.status": "running"}, observation="running",
        )
        speak = ToolMetadata(
            effect="WRITE", terminal=True, turn_terminal=True,
            session_terminal=True, produces=("speech.last_text",), max_attempts=1,
        )
        decision = Command(
            type="tool_call", name="act",
            params={
                "step_id": "", "action_type": "CALL_TOOL", "tool": "speak",
                "arguments": {"text": "服务正在运行。"},
            },
        )
        command, error = ModeBase._unwrap_act_command(decision, state)
        self.assertIsNone(error)
        self.assertEqual(command.name, "speak")
        allowed = ActionValidator().validate_tool(
            state=state, tool="speak", arguments=command.params,
            metadata=speak, available_tools=["speak"],
        )
        self.assertTrue(allowed.allowed)
        state.apply_post_goal_terminal_result(
            tool="speak", arguments=command.params, metadata=speak,
            success=True, facts={"speech.last_text": "服务正在运行。"},
            observation="服务正在运行。",
        )
        self.assertEqual(state.tool_history[-1].step_id, "")
        self.assertEqual(
            state.projection()["known_facts"]["speech.last_text"], "服务正在运行。"
        )


if __name__ == "__main__":
    unittest.main()
