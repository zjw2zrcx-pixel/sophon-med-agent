"""Authoritative mutable Harness state with append-only model projections.

The model proposes a plan and actions.  Only this module mutates execution
progress, fact validity, retry counters, epochs, and terminal status.
"""
from __future__ import annotations

import hashlib
import copy
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence

from ..medical_policy import consultation_dict, department_names


class TriState(str, Enum):
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNKNOWN = "UNKNOWN"


TERMINAL_STEPS = {"COMPLETED", "SKIPPED", "FAILED"}
CONDITION_OPERATORS = {
    "eq", "neq", "in", "not_in", "gt", "gte", "lt", "lte",
    "exists", "not_exists",
}


def _validate_condition_shape(condition: Dict[str, Any], label: str) -> None:
    fact = str(condition.get("fact", "") or "").strip()
    operator = str(condition.get("operator", "eq") or "eq")
    if not fact:
        raise ValueError(f"{label}.fact is required")
    if operator not in CONDITION_OPERATORS:
        raise ValueError(f"{label}.operator is unsupported: {operator}")
    if operator not in {"exists", "not_exists"} and "value" not in condition:
        raise ValueError(f"{label}.value is required")


@dataclass(frozen=True)
class HarnessLimits:
    max_agent_steps: int = 12
    max_tool_calls: int = 10
    max_replans: int = 2
    max_attempts_per_step: int = 2


@dataclass(frozen=True)
class ToolMetadata:
    effect: str = "READ"
    idempotent: bool = True
    produces: tuple[str, ...] = ()
    invalidates: tuple[str, ...] = ()
    verification_tool: str = ""
    terminal: bool = False
    turn_terminal: bool = False
    session_terminal: bool = False
    max_attempts: int = 2
    allowed_errors: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Optional[Dict[str, Any]]) -> "ToolMetadata":
        raw = value or {}
        retry = raw.get("retry") if isinstance(raw.get("retry"), dict) else {}
        effect = str(raw.get("effect", "READ")).upper()
        if effect not in {"READ", "WRITE"}:
            raise ValueError("tool x-harness.effect must be READ or WRITE")
        return cls(
            effect=effect,
            idempotent=bool(raw.get("idempotent", True)),
            produces=tuple(str(x) for x in raw.get("produces", ()) or ()),
            invalidates=tuple(str(x) for x in raw.get("invalidates", ()) or ()),
            verification_tool=str(raw.get("verification_tool", "") or ""),
            terminal=bool(raw.get("terminal", False)),
            turn_terminal=bool(raw.get("turn_terminal", raw.get("terminal", False))),
            session_terminal=bool(raw.get("session_terminal", False)),
            max_attempts=max(1, int(retry.get("max_attempts", 2))),
            allowed_errors=tuple(str(x).upper() for x in retry.get("allowed_errors", ())),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "effect": self.effect,
            "idempotent": self.idempotent,
            "produces": list(self.produces),
            "invalidates": list(self.invalidates),
            "verification_tool": self.verification_tool or None,
            "terminal": self.terminal,
            "turn_terminal": self.turn_terminal,
            "session_terminal": self.session_terminal,
            "retry": {
                "max_attempts": self.max_attempts,
                "allowed_errors": list(self.allowed_errors),
            },
        }


@dataclass(frozen=True)
class PlanStep:
    step_id: str
    goal: str
    preferred_tool: str = ""
    allowed_tools: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    condition: Optional[Dict[str, Any]] = None
    verification: bool = False
    max_attempts: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "goal": self.goal,
            "preferred_tool": self.preferred_tool or None,
            "allowed_tools": list(self.allowed_tools),
            "depends_on": list(self.depends_on),
            "condition": self.condition,
            "verification": self.verification,
        }


@dataclass(frozen=True)
class TaskPlan:
    plan_id: str
    revision: int
    goal: str
    goal_description: str
    success_conditions: tuple[Dict[str, Any], ...]
    steps: tuple[PlanStep, ...]
    done_when: str = ""
    parent_revision: Optional[int] = None
    reason: str = ""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "TaskPlan":
        if not isinstance(payload, dict):
            raise ValueError("plan must be an object")
        goal = str(payload.get("goal", "") or "").strip()
        if not goal:
            raise ValueError("plan.goal is required")
        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= 8:
            raise ValueError("plan.steps must contain 1 to 8 steps")

        steps: List[PlanStep] = []
        prior_id = ""
        for index, raw in enumerate(raw_steps, 1):
            default_id = f"s{index}"
            if isinstance(raw, str):
                text = raw.strip()
                preferred = ""
                match = re.match(r"(?:调用|call)\s+([A-Za-z0-9_.-]+)", text, re.I)
                if match:
                    preferred = match.group(1)
                step = PlanStep(
                    step_id=default_id,
                    goal=text,
                    preferred_tool=preferred,
                    depends_on=(prior_id,) if prior_id else (),
                )
            elif isinstance(raw, dict):
                step_id = str(raw.get("step_id", default_id) or default_id).strip()
                step_goal = str(
                    raw.get("goal") or raw.get("description") or raw.get("action") or ""
                ).strip()
                preferred = str(
                    raw.get("preferred_tool") or raw.get("name") or ""
                ).strip()
                if not step_goal:
                    step_goal = f"call_{preferred}" if preferred else ""
                depends = raw.get("depends_on")
                if depends is None:
                    depends = [prior_id] if prior_id else []
                if not isinstance(depends, list):
                    raise ValueError(f"{step_id}.depends_on must be an array")
                allowed = raw.get("allowed_tools", []) or []
                if not isinstance(allowed, list):
                    raise ValueError(f"{step_id}.allowed_tools must be an array")
                condition = raw.get("condition")
                if isinstance(condition, str) and condition.strip().lower() in {
                    "", "none", "null"
                }:
                    condition = None
                if condition is not None and not isinstance(condition, dict):
                    raise ValueError(f"{step_id}.condition must be an object or null")
                if condition is not None:
                    _validate_condition_shape(condition, f"{step_id}.condition")
                step = PlanStep(
                    step_id=step_id,
                    goal=step_goal,
                    preferred_tool=preferred,
                    allowed_tools=tuple(str(x) for x in allowed),
                    depends_on=tuple(str(x) for x in depends if x),
                    condition=condition,
                    verification=bool(raw.get("verification", False)),
                    max_attempts=(
                        max(1, int(raw["max_attempts"]))
                        if raw.get("max_attempts") is not None else None
                    ),
                )
            else:
                raise ValueError(f"step {index} must be a string or object")
            if not step.step_id or not step.goal:
                raise ValueError(f"step {index} requires step_id and goal")
            steps.append(step)
            prior_id = step.step_id

        ids = [step.step_id for step in steps]
        if len(set(ids)) != len(ids):
            raise ValueError("plan step_id values must be unique")
        prior: set[str] = set()
        for step in steps:
            if any(dep not in prior for dep in step.depends_on):
                raise ValueError(f"invalid dependency in {step.step_id}")
            prior.add(step.step_id)

        conditions = payload.get("success_conditions", []) or []
        if not isinstance(conditions, list) or any(not isinstance(x, dict) for x in conditions):
            raise ValueError("success_conditions must be an array of objects")
        for index, condition in enumerate(conditions):
            _validate_condition_shape(condition, f"success_conditions[{index}]")
        return cls(
            plan_id=str(payload.get("plan_id") or f"plan_{uuid.uuid4().hex[:10]}"),
            revision=max(1, int(payload.get("revision", 1))),
            goal=goal,
            goal_description=str(payload.get("goal_description") or goal).strip(),
            success_conditions=tuple(dict(x) for x in conditions),
            steps=tuple(steps),
            done_when=str(payload.get("done_when", "") or "").strip(),
            parent_revision=payload.get("parent_revision"),
            reason=str(payload.get("reason", "") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "revision": self.revision,
            "parent_revision": self.parent_revision,
            "goal": self.goal,
            "goal_description": self.goal_description,
            "success_conditions": [dict(x) for x in self.success_conditions],
            "steps": [step.to_dict() for step in self.steps],
            "done_when": self.done_when,
        }

    def validate_capabilities(self, capabilities: Dict[str, ToolMetadata]) -> None:
        """Reject impossible tool/fact references before execution starts."""
        produced_so_far: set[str] = set()
        for step in self.steps:
            referenced = set(step.allowed_tools)
            if step.preferred_tool:
                referenced.add(step.preferred_tool)
            unknown = sorted(name for name in referenced if name not in capabilities)
            if unknown:
                raise ValueError(
                    f"{step.step_id} references unavailable tools: {', '.join(unknown)}"
                )
            if step.condition:
                fact = str(step.condition.get("fact", ""))
                if not fact or fact not in produced_so_far:
                    raise ValueError(
                        f"{step.step_id} condition depends on unavailable prior fact: {fact or '<empty>'}"
                    )
            for name in referenced:
                produced_so_far.update(capabilities[name].produces)
        available_facts = {
            fact for metadata in capabilities.values() for fact in metadata.produces
        }
        for condition in self.success_conditions:
            fact = str(condition.get("fact", ""))
            if not fact or fact not in available_facts:
                raise ValueError(
                    f"success condition references unavailable fact: {fact or '<empty>'}"
                )


@dataclass
class FactRecord:
    value: Any
    valid: bool
    observed_epoch: int
    source: str
    observed_at_step: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "value": self.value,
            "valid": self.valid,
            "observed_epoch": self.observed_epoch,
            "source": self.source,
            "observed_at_step": self.observed_at_step,
        }


@dataclass
class StepState:
    status: str = "BLOCKED"
    attempts: int = 0
    last_error_type: str = ""
    result: Any = None


@dataclass(frozen=True)
class ToolCallRecord:
    signature: str
    tool: str
    arguments: Dict[str, Any]
    world_epoch: int
    step_id: str
    success: bool
    error_type: str = ""
    retryable: bool = False


@dataclass(frozen=True)
class ValidationResult:
    allowed: bool
    error_code: str = ""
    message: str = ""


def canonical_signature(tool: str, arguments: Dict[str, Any]) -> str:
    raw = tool + "\0" + json.dumps(
        arguments, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _compact_model_value(value: Any, max_chars: int = 1600) -> Any:
    """Bound projection size without changing the authoritative FactStore."""
    if isinstance(value, str):
        return value if len(value) <= max_chars else value[:max_chars] + "…[truncated]"
    if isinstance(value, (dict, list)):
        encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if len(encoded) <= max_chars:
            return value
        return encoded[:max_chars] + "…[truncated]"
    return value


def _bounded_text(value: Any, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…[truncated]"


def _list_items(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _medical_evidence_id(row: Dict[str, Any], index: int) -> str:
    stable = {
        key: row.get(key) for key in (
            "source", "source_line", "type", "aspect", "subject", "question"
        ) if row.get(key) not in (None, "")
    }
    digest = hashlib.sha256(json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:12]
    return f"medical-evidence-{index}-{digest}"


def _medical_association_id(row: Dict[str, Any], index: int) -> str:
    stable = {
        key: row.get(key) for key in (
            "matched", "relation", "related", "related_type", "direction", "source"
        ) if row.get(key) not in (None, "")
    }
    digest = hashlib.sha256(json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()[:12]
    return f"medical-association-{index}-{digest}"


def project_medical_consultation(value: Any) -> Dict[str, Any]:
    """Render a valid, answer-oriented medical fact for the model.

    The authoritative FactStore retains the complete consultation.  Model
    projections used to JSON-encode that object and cut it at 900 characters,
    often leaving broken JSON before the most relevant evidence.  This
    projection bounds individual fields instead, so intent, evidence gaps and
    every retained evidence row remain structurally valid.
    """
    medical = consultation_dict(value)
    evidence = []
    for index, raw in enumerate(_list_items(medical.get("evidence"))[:3], 1):
        if not isinstance(raw, dict):
            continue
        row = {
            "evidence_id": _medical_evidence_id(raw, index),
            **{
                key: raw.get(key) for key in (
                    "type", "aspect", "subject", "question", "source", "source_line"
                ) if raw.get(key) not in (None, "")
            },
        }
        if raw.get("text") not in (None, ""):
            row["text"] = _bounded_text(raw.get("text"), 520)
        evidence.append(row)

    associations = []
    for index, raw in enumerate(_list_items(medical.get("associations"))[:3], 1):
        if not isinstance(raw, dict):
            continue
        associations.append({
            "association_id": _medical_association_id(raw, index),
            **{
                key: _bounded_text(raw.get(key), 120)
                for key in (
                    "matched", "relation", "related", "related_type", "direction", "source"
                ) if raw.get(key) not in (None, "")
            },
        })

    terms = []
    for raw in _list_items(medical.get("normalized_terms"))[:5]:
        if not isinstance(raw, dict):
            continue
        terms.append({
            key: raw.get(key) for key in (
                "surface", "canonical", "type", "confidence", "match"
            ) if raw.get(key) not in (None, "")
        })

    status = str(medical.get("status", "") or "")
    intent = str(medical.get("intent", "") or "")
    departments = []
    for item in _list_items(medical.get("departments"))[:3]:
        if isinstance(item, dict):
            departments.append({
                key: item.get(key) for key in (
                    "department", "name", "relation", "source", "confidence"
                ) if item.get(key) not in (None, "")
            })
        elif str(item).strip():
            departments.append(str(item).strip())
    has_answer_evidence = bool(evidence or associations or departments)
    evidence_gap = status not in {"ok", "urgent"} or (
        not has_answer_evidence and status != "urgent"
    )
    questions = [
        _bounded_text(item, 160)
        for item in _list_items(medical.get("questions"))[:3]
    ]
    result = {
        "schema_version": "medical-answer-context.v1",
        "status": status,
        "query": _bounded_text(medical.get("query"), 300),
        "requested_aspect": intent,
        "message": _bounded_text(medical.get("message"), 260),
        "evidence_gap": evidence_gap,
        "normalized_terms": terms,
        "positive_symptoms": [
            _bounded_text(item, 80)
            for item in _list_items(medical.get("positive_symptoms"))[:8]
        ],
        "negative_symptoms": [
            _bounded_text(item, 80)
            for item in _list_items(medical.get("negative_symptoms"))[:8]
        ],
        "red_flags": [
            _bounded_text(item, 120)
            for item in _list_items(medical.get("red_flags"))[:5]
        ],
        "urgency": str(medical.get("urgency", "") or ""),
        "recommended_destination": _bounded_text(
            medical.get("recommended_destination"), 80
        ),
        "departments": departments,
        "medication_allowed": bool(medical.get("medication_allowed", False)),
        "medication_notice": _bounded_text(medical.get("medication_notice"), 240),
        "questions": questions,
        "followup_questions": questions,
        "evidence": evidence,
        "associations": associations,
        "answer_rules": {
            "answer_requested_aspect_first": True,
            "claims_must_use_retained_evidence": True,
            "state_evidence_gap_explicitly": evidence_gap,
            "do_not_infer_diagnosis_from_associations": True,
        },
    }
    # Keep the projection structurally valid under a total budget.  Remove
    # lower-priority context first, then shorten (never byte-slice) evidence.
    def rendered_chars() -> int:
        return len(json.dumps(
            result, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ))

    if rendered_chars() > 2400:
        result["associations"] = []
        result["normalized_terms"] = result["normalized_terms"][:3]
    for text_limit in (360, 240, 160):
        if rendered_chars() <= 2400:
            break
        for row in result["evidence"]:
            if "text" in row:
                row["text"] = _bounded_text(row["text"], text_limit)
    while rendered_chars() > 2400 and len(result["evidence"]) > 1:
        result["evidence"].pop()
    if rendered_chars() > 2400:
        result["positive_symptoms"] = result["positive_symptoms"][:4]
        result["negative_symptoms"] = result["negative_symptoms"][:4]
        result["message"] = _bounded_text(result["message"], 140)
    return result


def _condition_expression(condition: Dict[str, Any]) -> str:
    operator = str(condition.get("operator", "eq"))
    symbols = {
        "eq": "==", "neq": "!=", "gt": ">", "gte": ">=",
        "lt": "<", "lte": "<=", "in": "in", "not_in": "not in",
        "exists": "exists", "not_exists": "not exists",
    }
    prefix = f"{condition['fact']} {symbols.get(operator, operator)}"
    if operator in {"exists", "not_exists"}:
        return prefix
    return prefix + " " + json.dumps(
        condition.get("value"), ensure_ascii=False, separators=(",", ":")
    )


def _compare(actual: Any, operator: str, expected: Any) -> TriState:
    try:
        if operator == "exists":
            return TriState.TRUE
        if operator == "not_exists":
            return TriState.FALSE
        if operator == "eq":
            return TriState.TRUE if actual == expected else TriState.FALSE
        if operator == "neq":
            return TriState.TRUE if actual != expected else TriState.FALSE
        if operator == "in":
            return TriState.TRUE if actual in expected else TriState.FALSE
        if operator == "not_in":
            return TriState.TRUE if actual not in expected else TriState.FALSE
        if operator == "gt":
            return TriState.TRUE if actual > expected else TriState.FALSE
        if operator == "gte":
            return TriState.TRUE if actual >= expected else TriState.FALSE
        if operator == "lt":
            return TriState.TRUE if actual < expected else TriState.FALSE
        if operator == "lte":
            return TriState.TRUE if actual <= expected else TriState.FALSE
    except (TypeError, ValueError):
        return TriState.UNKNOWN
    raise ValueError(f"unsupported condition operator: {operator}")


@dataclass
class ExecutionState:
    execution_id: str
    plan: TaskPlan
    limits: HarnessLimits = field(default_factory=HarnessLimits)
    status: str = "RUNNING"
    active_step_id: str = ""
    step_states: Dict[str, StepState] = field(default_factory=dict)
    facts: Dict[str, FactRecord] = field(default_factory=dict)
    tool_history: List[ToolCallRecord] = field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = None
    last_observation: Optional[Dict[str, Any]] = None
    tool_call_count: int = 0
    total_actions: int = 0
    world_epoch: int = 0
    turn: int = 0
    replan_count: int = 0
    event_index: int = 0

    @classmethod
    def create(
        cls, plan: TaskPlan, initial_facts: Optional[Dict[str, Any]] = None,
        limits: Optional[HarnessLimits] = None,
    ) -> "ExecutionState":
        state = cls(
            execution_id=f"exec_{uuid.uuid4().hex[:10]}",
            plan=plan,
            limits=limits or HarnessLimits(),
            step_states={step.step_id: StepState() for step in plan.steps},
        )
        for key, value in (initial_facts or {}).items():
            state.set_fact(key, value, source="user", step_id="")
        state.refresh_step_readiness()
        return state

    def evaluate_condition(self, condition: Optional[Dict[str, Any]]) -> TriState:
        if not condition:
            return TriState.TRUE
        fact_name = str(condition.get("fact", ""))
        operator = str(condition.get("operator", "eq"))
        record = self.facts.get(fact_name)
        if record is None or not record.valid:
            if operator == "exists":
                return TriState.FALSE
            if operator == "not_exists":
                return TriState.TRUE
            return TriState.UNKNOWN
        return _compare(record.value, operator, condition.get("value"))

    def evaluate_success(self) -> TriState:
        if self.plan.success_conditions:
            results = []
            for condition in self.plan.success_conditions:
                fact_name = str(condition.get("fact", ""))
                operator = str(condition.get("operator", "eq"))
                record = self.facts.get(fact_name)
                if record is None or (condition.get("require_valid", True) and not record.valid):
                    if operator == "exists":
                        results.append(TriState.FALSE)
                    elif operator == "not_exists":
                        results.append(TriState.TRUE)
                    else:
                        results.append(TriState.UNKNOWN)
                else:
                    results.append(_compare(
                        record.value,
                        operator,
                        condition.get("value"),
                    ))
            if any(item == TriState.FALSE for item in results):
                return TriState.FALSE
            if any(item == TriState.UNKNOWN for item in results):
                return TriState.UNKNOWN
            return TriState.TRUE
        if all(item.status in {"COMPLETED", "SKIPPED"} for item in self.step_states.values()):
            return TriState.TRUE
        return TriState.FALSE

    def refresh_step_readiness(self) -> None:
        if self.status in {"GOAL_SATISFIED", "FINISHED", "FAILED", "CANCELLED", "BUDGET_EXHAUSTED"}:
            return
        for step in self.plan.steps:
            item = self.step_states[step.step_id]
            if item.status in TERMINAL_STEPS or item.status == "ACTIVE":
                continue
            dependencies = [self.step_states[dep].status for dep in step.depends_on]
            if not all(status in {"COMPLETED", "SKIPPED"} for status in dependencies):
                item.status = "BLOCKED"
                continue
            condition = self.evaluate_condition(step.condition)
            if condition == TriState.FALSE:
                item.status = "SKIPPED"
            elif condition == TriState.UNKNOWN:
                item.status = "BLOCKED"
            else:
                item.status = "READY"
        # Success conditions are authoritative and short-circuit all remaining
        # optional/verification work.  No later action may run once satisfied.
        if self.evaluate_success() == TriState.TRUE:
            self.active_step_id = ""
            self.status = "GOAL_SATISFIED"
            return
        ready = next(
            (step.step_id for step in self.plan.steps if self.step_states[step.step_id].status == "READY"),
            "",
        )
        self.active_step_id = ready
        if ready:
            self.step_states[ready].status = "ACTIVE"
            self.status = "RUNNING"
        elif self.evaluate_success() == TriState.TRUE:
            self.status = "GOAL_SATISFIED"
        elif any(item.status == "FAILED" for item in self.step_states.values()):
            self.status = "BLOCKED"
        else:
            self.status = "BLOCKED"

    @property
    def active_step(self) -> Optional[PlanStep]:
        return next((x for x in self.plan.steps if x.step_id == self.active_step_id), None)

    def set_fact(self, key: str, value: Any, *, source: str, step_id: str) -> None:
        self.facts[str(key)] = FactRecord(
            value=value,
            valid=True,
            observed_epoch=self.world_epoch,
            source=source,
            observed_at_step=step_id,
        )

    def invalidate_facts(self, names: Iterable[str]) -> None:
        for name in names:
            if name in self.facts:
                self.facts[name].valid = False

    def begin_model_turn(self) -> None:
        self.turn += 1
        if self.turn > self.limits.max_agent_steps:
            self.status = "BUDGET_EXHAUSTED"

    def reject_action(self, action: str, error_code: str, message: str) -> None:
        self.total_actions += 1
        self.last_action = {"type": "REJECTED", "action": action}
        self.last_observation = {
            "success": False,
            "error_type": error_code,
            "data": message,
        }
        if self.total_actions >= self.limits.max_agent_steps:
            self.status = "BUDGET_EXHAUSTED"

    def mark_plain_finish(self) -> ValidationResult:
        if self.status == "GOAL_SATISFIED":
            self.status = "FINISHED"
            return ValidationResult(True)
        if self.status == "BLOCKED" and not self.active_step_id:
            # The plan cannot progress.  Permit only a truthful final failure
            # report; tool execution remains blocked by ActionValidator.
            self.status = "FAILED"
            return ValidationResult(True)
        step = self.active_step
        if step and not step.preferred_tool and not step.allowed_tools and not self.plan.success_conditions:
            self.step_states[step.step_id].status = "COMPLETED"
            self.active_step_id = ""
            self.refresh_step_readiness()
            if self.status == "GOAL_SATISFIED":
                self.status = "FINISHED"
                return ValidationResult(True)
        return ValidationResult(
            False, "GOAL_NOT_YET_SATISFIED",
            "Harness 判定目标尚未满足；请执行 CURRENT STEP，不要提前结束。",
        )

    def apply_tool_result(
        self, *, tool: str, arguments: Dict[str, Any], metadata: ToolMetadata,
        success: bool, facts: Optional[Dict[str, Any]] = None,
        error_type: str = "", retryable: bool = False, observation: Any = None,
    ) -> None:
        step = self.active_step
        if step is None:
            raise RuntimeError("no active step for tool result")
        step_state = self.step_states[step.step_id]
        step_state.attempts += 1
        self.tool_call_count += 1
        self.total_actions += 1
        signature = canonical_signature(tool, arguments)
        epoch_before = self.world_epoch
        self.tool_history.append(ToolCallRecord(
            signature=signature,
            tool=tool,
            arguments=dict(arguments),
            world_epoch=epoch_before,
            step_id=step.step_id,
            success=success,
            error_type=(error_type or "").upper(),
            retryable=retryable,
        ))
        self.last_action = {"type": "CALL_TOOL", "step_id": step.step_id, "tool": tool}
        self.last_observation = {
            "success": success,
            "error_type": error_type or None,
            "data": observation,
        }
        if success:
            if metadata.effect == "WRITE":
                self.world_epoch += 1
                self.invalidate_facts(metadata.invalidates)
            for key, value in (facts or {}).items():
                self.set_fact(key, value, source=f"tool:{tool}", step_id=step.step_id)
            step_state.status = "COMPLETED"
            step_state.last_error_type = ""
            step_state.result = copy.deepcopy(observation)
            self.active_step_id = ""
            self.refresh_step_readiness()
            return

        normalized_error = (error_type or "TOOL_ERROR").upper()
        step_state.last_error_type = normalized_error
        max_attempts = min(
            step.max_attempts or metadata.max_attempts,
            self.limits.max_attempts_per_step,
        )
        retry_allowed = retryable and (
            not metadata.allowed_errors or normalized_error in metadata.allowed_errors
        )
        if retry_allowed and step_state.attempts < max_attempts:
            step_state.status = "READY"
            self.active_step_id = ""
            self.refresh_step_readiness()
            if self.active_step_id == step.step_id:
                self.status = "RECOVERING"
        else:
            step_state.status = "FAILED"
            self.active_step_id = ""
            self.status = "BLOCKED"

    def apply_post_goal_terminal_result(
        self, *, tool: str, arguments: Dict[str, Any], metadata: ToolMetadata,
        success: bool, facts: Optional[Dict[str, Any]] = None,
        error_type: str = "", retryable: bool = False, observation: Any = None,
    ) -> None:
        """Record a session-terminal action after all planned steps completed."""
        if (
            self.status not in {"GOAL_SATISFIED", "BLOCKED", "FAILED"}
            or self.active_step is not None
        ):
            raise RuntimeError(
                "post-plan terminal action requires a satisfied or blocked state"
            )
        self.tool_call_count += 1
        self.total_actions += 1
        signature = canonical_signature(tool, arguments)
        self.tool_history.append(ToolCallRecord(
            signature=signature,
            tool=tool,
            arguments=dict(arguments),
            world_epoch=self.world_epoch,
            step_id="",
            success=success,
            error_type=(error_type or "").upper(),
            retryable=retryable,
        ))
        self.last_action = {"type": "CALL_TOOL", "step_id": "", "tool": tool}
        self.last_observation = {
            "success": success,
            "error_type": error_type or None,
            "data": observation,
        }
        if success:
            if metadata.effect == "WRITE":
                self.world_epoch += 1
                self.invalidate_facts(metadata.invalidates)
            for key, value in (facts or {}).items():
                self.set_fact(key, value, source=f"tool:{tool}", step_id="")

    def projection(self) -> Dict[str, Any]:
        step = self.active_step
        consultation_record = self.facts.get("medical.consultation")
        projected_medical = (
            project_medical_consultation(consultation_record.value)
            if consultation_record is not None and consultation_record.valid
            else None
        )
        medical_observation = (
            {
                "tool": "medical_consult",
                "status": projected_medical.get("status", ""),
                "requested_aspect": projected_medical.get("requested_aspect", ""),
                "evidence_count": len(projected_medical.get("evidence", [])),
                "evidence_gap": bool(projected_medical.get("evidence_gap", False)),
                "followup_required": bool(projected_medical.get("questions", [])),
            }
            if projected_medical is not None else None
        )
        step_index = (
            next(
                index for index, item in enumerate(self.plan.steps, 1)
                if item.step_id == step.step_id
            )
            if step else None
        )
        completed_steps = []
        remaining_steps = []
        for index, plan_step in enumerate(self.plan.steps, 1):
            state = self.step_states[plan_step.step_id]
            if state.status == "COMPLETED":
                completed_steps.append({
                    "id": index,
                    "step_id": plan_step.step_id,
                    "goal": plan_step.goal,
                    "result": (
                        medical_observation
                        if plan_step.preferred_tool == "medical_consult"
                        and medical_observation is not None
                        else _compact_model_value(state.result, max_chars=240)
                    ),
                })
            elif state.status not in {"SKIPPED", "FAILED"}:
                remaining_steps.append(plan_step.goal)
        success_parts = [
            _condition_expression(condition)
            for condition in self.plan.success_conditions
        ]
        dialogue_policy: Dict[str, Any] = {}
        followup = self.facts.get("dialogue.followup_required")
        if followup is not None and followup.valid and followup.value is True:
            dialogue_policy = {
                "required_next_tool": "query_unless_session_turn_limit",
                "reason": "medical_consult explicitly requires more information",
                "forbid_speak_before_query": True,
            }
        medical_response_policy: Dict[str, Any] = {}
        if consultation_record is not None and consultation_record.valid:
            medical = consultation_dict(consultation_record.value)
            allowed_departments = sorted(department_names(medical.get("departments", [])))
            destination = str(medical.get("recommended_destination", "")).strip()
            if destination:
                allowed_departments.append(destination)
            navigation = self.facts.get("navigation.target")
            if navigation is not None and navigation.valid and str(navigation.value).strip():
                allowed_departments.append(str(navigation.value).strip())
            medical_response_policy = {
                "allowed_departments": sorted(set(allowed_departments)),
                "department_rule": (
                    "do_not_name_or_recommend_any_department"
                    if not allowed_departments
                    else "only_name_allowed_departments"
                ),
            }
        return {
            "execution_id": self.execution_id,
            "plan_id": self.plan.plan_id,
            "plan_revision": self.plan.revision,
            "goal": self.plan.goal,
            "status": self.status,
            "turn": self.turn,
            "world_epoch": self.world_epoch,
            "current_step": step_index,
            "current_step_id": step.step_id if step else None,
            "current_step_detail": (
                {
                    "goal": step.goal,
                    "preferred_tool": step.preferred_tool or None,
                    "verification": step.verification,
                }
                if step else None
            ),
            "completed_steps": completed_steps,
            "remaining_steps": remaining_steps,
            "success_condition": (
                " AND ".join(success_parts)
                if success_parts else (self.plan.done_when or "all steps completed")
            ),
            "step_states": {
                key: {"status": value.status, "attempts": value.attempts}
                for key, value in self.step_states.items()
            },
            "known_facts": {
                key: (
                    projected_medical
                    if key == "medical.consultation"
                    else _compact_model_value(record.value, max_chars=900)
                )
                for key, record in self.facts.items() if record.valid
            },
            "invalidated_facts": [key for key, record in self.facts.items() if not record.valid],
            "dialogue_policy": dialogue_policy or None,
            "medical_response_policy": medical_response_policy or None,
            "last_action": self.last_action,
            "last_observation": (
                {
                    **self.last_observation,
                    "data": (
                        medical_observation
                        if self.last_action
                        and self.last_action.get("tool") == "medical_consult"
                        and medical_observation is not None
                        else _compact_model_value(
                            self.last_observation.get("data"), max_chars=180
                        )
                    ),
                }
                if self.last_observation else None
            ),
            "budgets": {
                "agent_steps_remaining": max(0, self.limits.max_agent_steps - self.turn),
                "tool_calls_remaining": max(0, self.limits.max_tool_calls - self.tool_call_count),
                "replans_remaining": max(0, self.limits.max_replans - self.replan_count),
            },
        }

    def append_event(self, event_type: str) -> Dict[str, Any]:
        self.event_index += 1
        return {
            "index": self.event_index,
            "type": event_type,
            "timestamp": time.time(),
            "state": copy.deepcopy(self.projection()),
        }

    def to_dict(self) -> Dict[str, Any]:
        value = self.projection()
        value["facts"] = {key: record.to_dict() for key, record in self.facts.items()}
        value["tool_history"] = [
            {
                "signature": item.signature,
                "tool": item.tool,
                "arguments": item.arguments,
                "world_epoch": item.world_epoch,
                "step_id": item.step_id,
                "success": item.success,
                "error_type": item.error_type,
                "retryable": item.retryable,
            }
            for item in self.tool_history
        ]
        return value


class ActionValidator:
    """Validate a proposed action before any external tool is invoked."""

    def validate_tool(
        self, *, state: ExecutionState, tool: str, arguments: Dict[str, Any],
        metadata: ToolMetadata, available_tools: Sequence[str],
    ) -> ValidationResult:
        if state.status == "FINISHED":
            return ValidationResult(
                False, "GOAL_ALREADY_SATISFIED",
                "目标已满足，禁止继续调用工具；请直接给出最终答复。",
            )
        if state.status == "BUDGET_EXHAUSTED" or state.total_actions >= state.limits.max_agent_steps:
            state.status = "BUDGET_EXHAUSTED"
            return ValidationResult(False, "ACTION_BUDGET_EXHAUSTED", "动作预算已耗尽。")
        if state.tool_call_count >= state.limits.max_tool_calls:
            state.status = "BUDGET_EXHAUSTED"
            return ValidationResult(False, "TOOL_BUDGET_EXHAUSTED", "工具调用预算已耗尽。")
        if tool not in available_tools:
            return ValidationResult(False, "UNKNOWN_TOOL", f"工具 {tool} 不存在或当前模式不可用。")
        if state.status in {"GOAL_SATISFIED", "BLOCKED", "FAILED"}:
            if not metadata.session_terminal:
                return ValidationResult(
                    False, "GOAL_ALREADY_SATISFIED",
                    "计划已结束或阻塞，只允许调用 session-terminal 工具结束会话。",
                )
            signature = canonical_signature(tool, arguments)
            if any(
                item.signature == signature and item.world_epoch == state.world_epoch
                for item in state.tool_history
            ):
                return ValidationResult(
                    False, "DUPLICATE_NO_WORLD_CHANGE",
                    "终止动作已使用相同参数调用过，不得重复执行。",
                )
            return ValidationResult(True)
        step = state.active_step
        if step is None:
            return ValidationResult(False, "NO_EXECUTABLE_STEP", "当前没有可执行步骤，需要修正计划或结束任务。")
        step_tools = set(step.allowed_tools)
        if step.preferred_tool:
            step_tools.add(step.preferred_tool)
        if step_tools and tool not in step_tools:
            return ValidationResult(
                False, "STEP_TOOL_MISMATCH",
                f"当前步骤 {step.step_id} 只允许工具: {', '.join(sorted(step_tools))}。",
            )
        if metadata.session_terminal:
            later_steps = [
                plan_step.step_id for plan_step in state.plan.steps
                if plan_step.step_id != step.step_id
                and state.step_states[plan_step.step_id].status
                not in {"COMPLETED", "SKIPPED", "FAILED"}
            ]
            if later_steps:
                return ValidationResult(
                    False, "SESSION_TERMINAL_BEFORE_FINAL_STEP",
                    "session-terminal 工具只能在最后一个未完成步骤执行；"
                    "请先完成后续步骤: " + ", ".join(later_steps),
                )
        # A turn-terminal action such as query intentionally pauses before the
        # final goal is satisfied. Only a session-terminal action (speak) must
        # prove that every remaining condition is produced or already true.
        if metadata.session_terminal and state.plan.success_conditions:
            unmet = [
                condition for condition in state.plan.success_conditions
                if state.evaluate_condition(condition) != TriState.TRUE
            ]
            # A terminal tool may itself establish the remaining terminal
            # facts (for example speak -> speech.last_text). Reject only when
            # some unmet condition cannot be produced by this action.
            unproducible = [
                condition for condition in unmet
                if str(condition.get("fact", "")) not in metadata.produces
            ]
            if unproducible:
                return ValidationResult(
                    False, "GOAL_NOT_YET_SATISFIED",
                    "终止型动作不能在 success_conditions 满足前执行。",
                )
        step_state = state.step_states[step.step_id]
        max_attempts = min(
            step.max_attempts or metadata.max_attempts,
            state.limits.max_attempts_per_step,
        )
        if step_state.attempts >= max_attempts:
            step_state.status = "FAILED"
            state.status = "BLOCKED"
            return ValidationResult(False, "RETRY_EXHAUSTED", f"步骤 {step.step_id} 已达到重试上限。")
        signature = canonical_signature(tool, arguments)
        same_epoch = [
            item for item in state.tool_history
            if item.signature == signature and item.world_epoch == state.world_epoch
        ]
        if same_epoch:
            previous = same_epoch[-1]
            transient_retry = (
                not previous.success
                and previous.retryable
                and previous.error_type in {"TIMEOUT", "TEMPORARY_UNAVAILABLE"}
                and state.status == "RECOVERING"
            )
            if transient_retry:
                return ValidationResult(True)
            return ValidationResult(
                False, "DUPLICATE_NO_WORLD_CHANGE",
                f"相同工具和参数已在 world_epoch={state.world_epoch} 执行；世界状态未变化，禁止重复。",
            )
        return ValidationResult(True)
