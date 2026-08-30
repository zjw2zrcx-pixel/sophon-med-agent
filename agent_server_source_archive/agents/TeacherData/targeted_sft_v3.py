"""Build and gate targeted ``agent-sft-decision.v3`` canary data.

Generation is deliberately split in two phases.  ``generate`` executes the
current local medical tool and freezes the exact production prompt at the
final model decision point.  It does not invent a target answer.  ``compile``
accepts separately generated teacher outputs only when both deterministic
validation and an evidence-only semantic approval are present.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sqlite3
from typing import Any, Iterable

from agents.API.session import Session
from agents.Harness import ExecutionState, TaskPlan, ToolMetadata
from agents.Harness.state import project_medical_consultation
from agents.agent import Agent, AgentConfig
from agents.MCP.base import ToolResult
from agents.medical_policy import unsupported_departments


SCENARIO_SCHEMA = "targeted-sft-scenario.v1"
SAMPLE_SCHEMA = "agent-sft-decision.v3"
REVIEW_SCHEMA = "targeted-semantic-review.v1"
ASPECTS = (
    "病因", "症状", "检查", "手术治疗", "治疗", "预防", "并发症", "风险因素",
)
ASPECT_QUESTIONS = {
    "病因": "{subject}的病因是什么？",
    "症状": "{subject}通常有哪些症状？",
    "检查": "{subject}一般需要做哪些检查？",
    "手术治疗": "本地资料里有{subject}的手术治疗方法吗？",
    "治疗": "{subject}通常如何治疗？",
    "预防": "怎样预防{subject}？",
    "并发症": "{subject}可能有哪些并发症？",
    "风险因素": "{subject}有哪些风险因素？",
}
EXPECTED_RETRIEVAL_INTENT = {
    "病因": "causes",
    "症状": "symptoms",
    "检查": "checks",
    "手术治疗": "surgery",
    "治疗": "treatment",
    "预防": "prevention",
    "并发症": "complications",
    "风险因素": "risk_factors",
}
REQUEST_FACT_ASPECTS = {
    "病因": ("病因", "发病机制", "遗传因素", "传播途径"),
    "症状": ("症状",),
    "检查": ("检查", "检查结果"),
    "手术治疗": ("手术治疗",),
    "治疗": ("治疗", "辅助治疗", "放射治疗", "化疗"),
    "预防": ("预防",),
    "并发症": ("并发症",),
    "风险因素": ("风险因素",),
}


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    raw = value if isinstance(value, str) else _canonical(value)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+|[，。！？；、,.!?;:'\"“”‘’（）()【】\[\]]", "", str(text)).lower()


def _walk_strings(value: Any, keys: set[str] | None = None) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(child, str) and (keys is None or key in keys):
                yield child
            else:
                yield from _walk_strings(child, keys)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_strings(child, keys)


def exclusion_registry(root: Path) -> dict[str, set[Any]]:
    """Collect benchmark surfaces plus frozen-training source identities."""
    patterns = (
        "benchmarks/**/final_benchmark.json",
        "docs/final_agent_benchmark/**/dataset.json",
        "teacher_trajectories/agent_current_review_*/dataset.json",
    )
    hashes: set[str] = set()
    source_rows: set[tuple[str, int]] = set()
    answer_hashes: set[str] = set()
    family_ids: set[str] = set()
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.glob(pattern):
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for text in _walk_strings(payload, {"prompt", "user", "input", "text"}):
                normalized = _norm(text)
                if normalized:
                    hashes.add(_sha(normalized))
    final_prompts = root / "teacher_trajectories" / "final_v7_1000" / "prompts.json"
    if final_prompts.is_file():
        payload = json.loads(final_prompts.read_text(encoding="utf-8"))
        for case in payload.get("cases", []):
            for text in list(case.get("turns") or [case.get("prompt", "")]):
                normalized = _norm(str(text))
                if normalized:
                    hashes.add(_sha(normalized))
            source = case.get("medical_source")
            if isinstance(source, dict):
                table = str(source.get("database_table", ""))
                row_id = int(source.get("row_id", 0) or 0)
                if table and row_id:
                    source_rows.add((table, row_id))
                answer_hash = str(source.get("answer_sha256", ""))
                if answer_hash:
                    answer_hashes.add(answer_hash)
            metadata = case.get("intent_metadata")
            if not isinstance(metadata, dict):
                metadata = (case.get("final_tags") or {}).get("intent_metadata")
            if isinstance(metadata, dict) and metadata.get("semantic_family_id"):
                family_ids.add(str(metadata["semantic_family_id"]))
    # Targeted batches are JSONL rather than the legacy JSON container above.
    # Freeze their source identities as soon as a batch has been materialized,
    # so a new seed cannot silently reproduce an earlier teacher sample.
    for path in root.glob("teacher_trajectories/targeted_sft_v3_*/scenarios.jsonl"):
        try:
            rows = (
                json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            for row in rows:
                source = row.get("source") or {}
                table = str(source.get("source", ""))
                row_id = int(source.get("id", 0) or 0)
                if table and row_id:
                    source_rows.add((table, row_id))
                answer_hash = str(source.get("answer_sha256", ""))
                if answer_hash:
                    answer_hashes.add(answer_hash)
                family_id = str(row.get("semantic_family_id", ""))
                if family_id:
                    family_ids.add(family_id)
                prompt = _norm(str(row.get("prompt", "")))
                if prompt:
                    hashes.add(_sha(prompt))
        except Exception:
            continue
    return {
        "prompt_hashes": hashes,
        "source_rows": source_rows,
        "answer_hashes": answer_hashes,
        "family_ids": family_ids,
    }


def benchmark_prompt_hashes(root: Path) -> set[str]:
    """Backward-compatible view used by tests and diagnostics."""
    return set(exclusion_registry(root)["prompt_hashes"])


def _source_rows(database: Path, count: int, seed: int) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        maximum = int(connection.execute("SELECT max(id) FROM facts").fetchone()[0])
        attempts = 0
        while len(rows) < count and attempts < count * 100:
            attempts += 1
            aspect = ASPECTS[len(rows) % len(ASPECTS)]
            start = randomizer.randint(1, maximum)
            row = connection.execute(
                "SELECT id,subject,aspect,answer,source,source_line FROM facts "
                "WHERE id>=? AND aspect=? AND quality>=0.8 "
                "AND source LIKE '%/train' AND length(subject) BETWEEN 2 AND 18 "
                "AND length(answer)>=8 ORDER BY id LIMIT 1",
                (start, aspect),
            ).fetchone()
            if not row or int(row[0]) in used:
                continue
            used.add(int(row[0]))
            rows.append({
                "id": int(row[0]), "subject": str(row[1]).strip(),
                "aspect": str(row[2]).strip(), "answer_sha256": _sha(str(row[3])),
                "source": str(row[4]), "source_line": int(row[5]),
            })
    if len(rows) < count:
        raise RuntimeError(f"only sampled {len(rows)} of {count} requested fact rows")
    return rows


def _gap_source_rows(database: Path, count: int, seed: int) -> list[dict[str, Any]]:
    """Sample entity anchors that lack the requested structured fact aspect."""
    randomizer = random.Random(seed ^ 0xE71D3ACE)
    rows: list[dict[str, Any]] = []
    used: set[int] = set()
    with sqlite3.connect(f"file:{database.resolve()}?mode=ro", uri=True) as connection:
        maximum = int(connection.execute("SELECT max(id) FROM facts").fetchone()[0])
        attempts = 0
        while len(rows) < count and attempts < count * 250:
            attempts += 1
            requested_aspect = ASPECTS[len(rows) % len(ASPECTS)]
            absent_aspects = REQUEST_FACT_ASPECTS[requested_aspect]
            placeholders = ",".join("?" for _ in absent_aspects)
            start = randomizer.randint(1, maximum)
            row = connection.execute(
                "SELECT f.id,f.subject,f.aspect,f.answer,f.source,f.source_line "
                "FROM facts f WHERE f.id>=? AND f.quality>=0.8 "
                "AND f.source LIKE '%/train' AND length(f.subject) BETWEEN 2 AND 18 "
                "AND length(f.answer)>=8 AND f.aspect NOT IN (" + placeholders + ") "
                "AND EXISTS (SELECT 1 FROM entities e WHERE e.name=f.subject "
                "AND (e.label LIKE '%疾病%' OR lower(e.label) IN "
                "('disease','illness','disorder'))) "
                "AND NOT EXISTS (SELECT 1 FROM facts g WHERE g.subject=f.subject "
                "AND g.aspect IN (" + placeholders + ")) "
                "ORDER BY f.id LIMIT 1",
                (start, *absent_aspects, *absent_aspects),
            ).fetchone()
            if not row or int(row[0]) in used:
                continue
            used.add(int(row[0]))
            rows.append({
                "id": int(row[0]), "subject": str(row[1]).strip(),
                "aspect": requested_aspect,
                "anchor_aspect": str(row[2]).strip(),
                "answer_sha256": _sha(str(row[3])),
                "source": str(row[4]), "source_line": int(row[5]),
                "requested_evidence_gap": True,
            })
    if len(rows) < count:
        raise RuntimeError(
            f"only sampled {len(rows)} of {count} requested evidence-gap anchors"
        )
    return rows


def _teacher_instruction() -> str:
    return (
        "只根据 medical_context 中保留的 evidence/associations 回答用户当前 requested_aspect。"
        "不得使用外部医学知识，不得添加证据外疾病、药物、剂量、疗程、检查、病因、症状或科室。"
        "evidence_gap=true 时不得猜测：一般问题直接说明本地资料不足、无法确定；治疗或手术治疗问题"
        "必须拒绝给出具体方案，并说明无法安全作答。回答不超过 supervision.max_output_chars，"
        "以完整中文句末标点结束。只输出一个严格的 <tool> JSON："
        '<tool>{"tool_call":"act","param":{"step_id":"s_final",'
        '"action_type":"CALL_TOOL","tool":"speak","arguments":{"text":"..."}}}</tool>'
    )


def _call_preloaded_medical_tool_sync(medical_tool: Any, query: str) -> ToolResult:
    """Execute the local read-only consult with production-equivalent wrapping.

    The regular async tool boundary is appropriate for the live agent, but the
    sandbox's generic worker makes this CPU-heavy offline batch path unusably
    slow. The same configured retriever and ToolResult fact contract are used.
    """
    result = medical_tool._retriever(medical_tool._index_path()).consult(query)
    if result.get("status") == "not_found":
        return ToolResult(
            success=False,
            data=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            error="医疗知识库未返回可用证据",
            error_type="NOT_FOUND",
            empty=True,
            retryable=True,
            recovery_hint="请保留症状要点并换一种简短句式查询；不要重复原参数。",
        )
    questions = [str(item) for item in result.get("questions", []) if str(item).strip()]
    needs_followup = medical_tool._needs_followup(result)
    return ToolResult(
        success=True,
        data=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        facts={
            "medical.consultation": result,
            "dialogue.followup_required": needs_followup,
            "dialogue.followup_questions": questions[:3],
        },
    )


async def generate_scenarios(
    *, count: int, database: Path, seed: int, repository: Path,
    evidence_gap_count: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be positive")
    if evidence_gap_count < 0 or evidence_gap_count > count:
        raise ValueError("evidence_gap_count must be between zero and count")
    exclusions = exclusion_registry(repository)
    excluded = exclusions["prompt_hashes"]
    grounded_target = count - evidence_gap_count
    candidate_rows = [
        (source, False) for source in _source_rows(database, max(1, grounded_target) * 6, seed)
    ]
    if evidence_gap_count:
        candidate_rows.extend(
            (source, True) for source in _gap_source_rows(
                database, evidence_gap_count * 30, seed
            )
        )
    agent = Agent(AgentConfig(
        default_mode="Voice", camera_backend="dummy", medical_dense_enabled=False,
        emit_agent_events=False, trajectory_enabled=False,
    ))
    agent.initialize()
    mode = agent.voice_mode
    assert mode is not None
    medical_tool = agent.mcp.tools["medical_consult"]
    print(json.dumps({
        "stage": "medical_index_loading", "database": str(database.resolve())
    }, ensure_ascii=False), flush=True)
    # Preload synchronously once. In this short-lived compiler process the
    # Python-heavy alias-trie build is markedly slower in asyncio's generic
    # worker; subsequent production tool calls still use their normal async
    # boundary and hit this cache.
    medical_tool._retriever(medical_tool._index_path())
    print(json.dumps({"stage": "medical_index_ready"}, ensure_ascii=False), flush=True)
    raw_metadata = medical_tool.get_harness_metadata()
    metadata = (
        raw_metadata if isinstance(raw_metadata, ToolMetadata)
        else ToolMetadata.from_dict(raw_metadata)
    )
    scenarios: list[dict[str, Any]] = []
    rejected = Counter()
    seen_families: set[str] = set()
    seen_source_rows: set[int] = set()
    generated_by_gap = Counter()
    for source, desired_gap in candidate_rows:
        if len(scenarios) >= count:
            break
        if desired_gap and generated_by_gap[True] >= evidence_gap_count:
            continue
        if not desired_gap and generated_by_gap[False] >= grounded_target:
            continue
        if source["id"] in seen_source_rows:
            rejected["duplicate_source_row"] += 1
            continue
        prompt = ASPECT_QUESTIONS[source["aspect"]].format(subject=source["subject"])
        if ("facts", source["id"]) in exclusions["source_rows"]:
            rejected["frozen_source_row_collision"] += 1
            continue
        if source["answer_sha256"] in exclusions["answer_hashes"]:
            rejected["frozen_answer_hash_collision"] += 1
            continue
        prompt_hash = _sha(_norm(prompt))
        if prompt_hash in excluded:
            rejected["benchmark_surface_collision"] += 1
            continue
        family_id = _sha({
            "subject": source["subject"], "aspect": source["aspect"],
            "source": source["source"], "source_line": source["source_line"],
        })[:20]
        if family_id in exclusions["family_ids"]:
            rejected["frozen_semantic_family_collision"] += 1
            continue
        if family_id in seen_families:
            rejected["duplicate_semantic_family"] += 1
            continue
        result = _call_preloaded_medical_tool_sync(medical_tool, prompt)
        if not result.success:
            rejected[f"tool_{result.error_type or 'failure'}"] += 1
            continue
        medical = project_medical_consultation(
            result.facts.get("medical.consultation", {})
        )
        expected_intent = EXPECTED_RETRIEVAL_INTENT[source["aspect"]]
        observed_intent = str(medical.get("requested_aspect", ""))
        if observed_intent != expected_intent:
            rejected[f"intent_mismatch:{source['aspect']}->{observed_intent or 'missing'}"] += 1
            continue
        observed_gap = bool(medical.get("evidence_gap"))
        if observed_gap != desired_gap:
            rejected[
                "gap_candidate_has_evidence" if desired_gap else "grounded_candidate_has_gap"
            ] += 1
            continue
        if not desired_gap:
            source_subject = _norm(source["subject"])
            allowed_fact_aspects = set(REQUEST_FACT_ASPECTS[source["aspect"]])
            aligned_fact = any(
                str(item.get("type", "")) == "fact"
                and _norm(item.get("subject", "")) == source_subject
                and str(item.get("aspect", "")) in allowed_fact_aspects
                for item in medical.get("evidence", [])
                if isinstance(item, dict)
            )
            if not aligned_fact:
                rejected["grounded_source_fact_not_retrieved"] += 1
                continue
        plan_payload = mode.get_compact_workflow_plan(prompt)
        if not plan_payload:
            rejected["no_compact_workflow"] += 1
            continue
        state = ExecutionState.create(TaskPlan.from_payload(plan_payload))
        slots = Session(mode="Voice").prompt_slots
        slots.start_task(system=mode.get_base_prompt(), user_input=prompt)
        slots.set_plan(plan_payload)
        slots.append_execution_event(state.append_event("AGENT_WORKFLOW_ATTACHED"))
        state.apply_tool_result(
            tool="medical_consult", arguments={"query": prompt}, metadata=metadata,
            success=True, facts=dict(result.facts), observation=result.data,
        )
        slots.append_execution_event(state.append_event("ACTION_APPLIED"))
        active = state.active_step
        if active is None or active.preferred_tool != "speak":
            rejected["not_final_speak_decision"] += 1
            continue
        session = Session(mode="Voice", prompt_slots=slots)
        prompt_slots = agent.api._preflight_prompt(session)
        evidence_ids = [
            str(row.get("evidence_id")) for row in medical.get("evidence", [])
            if row.get("evidence_id")
        ]
        evidence_ids.extend(
            str(row.get("association_id"))
            for row in medical.get("associations", [])
            if row.get("association_id")
        )
        split_bucket = int(_sha(family_id)[:8], 16) % 10
        split = "test" if split_bucket == 0 else "dev" if split_bucket == 1 else "train"
        case_id = f"targeted-med-{source['aspect']}-{len(scenarios) + 1:04d}"
        scenario = {
            "schema_version": SCENARIO_SCHEMA,
            "case_id": case_id,
            "category": "medical",
            "prompt": prompt,
            "split": split,
            "semantic_family_id": family_id,
            "input": {"prompt_slots": prompt_slots, "model": agent.config.model_name},
            "medical_context": medical,
            "supervision": {
                "decision_role": "medical_final_synthesis",
                "intent_aspect": medical.get("requested_aspect"),
                "source_aspect": source["aspect"],
                "expected_retrieval_intent": expected_intent,
                "required_tool": "speak", "required_step_id": "s_final",
                "forbidden_tools": ["medical_consult", "navigate", "query"],
                "evidence_ids": evidence_ids,
                "evidence_gap": bool(medical.get("evidence_gap")),
                "max_output_chars": 90,
                "semantic_review_required": True,
            },
            "source": source,
            "provenance": {
                "generator": "targeted_sft_v3.generate",
                "prompt_protocol": "suha.v3",
                "medical_tool_executed": True,
                "medical_tool_execution_adapter": "preloaded_sync_equivalent",
                "benchmark_prompts_excluded": True,
                "prompt_sha256": _sha(prompt),
                "source_row_sha256": _sha(source),
                "reasoning_removed": True,
            },
            "prompt_preflight": dict(session.last_prompt_preflight),
            "target_status": "teacher_required",
        }
        scenario["scenario_sha256"] = _sha(scenario)
        scenarios.append(scenario)
        seen_families.add(family_id)
        seen_source_rows.add(source["id"])
        generated_by_gap[desired_gap] += 1
        print(json.dumps({
            "stage": "scenario_generated", "completed": len(scenarios),
            "requested": count, "case_id": case_id,
        }, ensure_ascii=False), flush=True)
    await agent.api.close()
    if len(scenarios) < count:
        raise RuntimeError(
            f"only generated {len(scenarios)} of {count}; rejected={dict(rejected)}"
        )
    report = {
        "schema_version": "targeted-sft-generation-report.v1",
        "requested": count, "generated": len(scenarios),
        "rejected": dict(rejected),
        "aspects": dict(Counter(x["source"]["aspect"] for x in scenarios)),
        "splits": dict(Counter(x["split"] for x in scenarios)),
        "evidence_gap": dict(Counter(
            str(x["supervision"]["evidence_gap"]).lower() for x in scenarios
        )),
        "benchmark_surface_hash_count": len(excluded),
        "frozen_source_row_count": len(exclusions["source_rows"]),
        "frozen_answer_hash_count": len(exclusions["answer_hashes"]),
        "frozen_semantic_family_count": len(exclusions["family_ids"]),
        "teacher_outputs_included": False,
        "requested_evidence_gap_count": evidence_gap_count,
        "generated_evidence_gap_count": generated_by_gap[True],
    }
    return scenarios, report


def parse_target_output(output: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    value = str(output or "").strip()
    if not (value.startswith("<tool>") and value.endswith("</tool>")):
        return None, ["INVALID_TOOL_ENVELOPE"]
    try:
        payload = json.loads(value[6:-7])
    except json.JSONDecodeError:
        return None, ["INVALID_TOOL_JSON"]
    if payload.get("tool_call") != "act":
        errors.append("NOT_ACT")
    param = payload.get("param") if isinstance(payload.get("param"), dict) else {}
    if param.get("step_id") != "s_final":
        errors.append("WRONG_STEP_ID")
    if str(param.get("action_type", "")).upper() != "CALL_TOOL":
        errors.append("WRONG_ACTION_TYPE")
    if param.get("tool") != "speak":
        errors.append("WRONG_TOOL")
    arguments = param.get("arguments") if isinstance(param.get("arguments"), dict) else {}
    text = str(arguments.get("text", "") or "").strip()
    if not text:
        errors.append("EMPTY_SPEAK_TEXT")
    return {"payload": payload, "text": text}, errors


def validate_target(scenario: dict[str, Any], output: str) -> dict[str, Any]:
    parsed, errors = parse_target_output(output)
    text = parsed["text"] if parsed else ""
    supervision = scenario.get("supervision") or {}
    if len(text) > int(supervision.get("max_output_chars", 90)):
        errors.append("OUTPUT_TOO_LONG")
    if text and not re.search(r"[。！？!?]$", text):
        errors.append("INCOMPLETE_SENTENCE_BOUNDARY")
    if supervision.get("evidence_gap") and text and not re.search(
        r"资料.{0,8}(?:不足|有限|未覆盖|没有)|没有.{0,8}(?:资料|证据)|无法从.{0,8}资料",
        text,
    ):
        errors.append("EVIDENCE_GAP_NOT_EXPLICIT")
    unsupported = unsupported_departments(
        text, scenario.get("medical_context") or {}
    )
    if unsupported:
        errors.extend(f"UNSUPPORTED_DEPARTMENT:{item}" for item in unsupported)
    return {
        "deterministic_pass": not errors,
        "errors": sorted(set(errors)),
        "text": text,
        "semantic_review_required": True,
    }


def _semantic_review_instruction() -> str:
    return (
        "你是独立证据审核器，不是答案改写器。只依据 medical_context 审核 candidate_text，"
        "不得使用外部医学知识。逐项列出候选中的每个事实、保守安全建议或不确定性声明。"
        "医疗事实必须由 allowed_evidence_ids 中至少一个证据直接支持；安全建议和明确的"
        "资料缺口可使用 support=conservative_policy 且 evidence_ids=[]。任何错 aspect、"
        "错疾病、夸大、证据外药物/剂量/检查/科室或遗漏缺口均 decision=reject。"
        "只输出一个 JSON 对象：schema_version='targeted-semantic-review.v1'、case_id、"
        "decision('approve'或'reject')、claim_evidence_map；其中每项含 claim、"
        "kind('medical_fact'|'safety_advice'|'uncertainty')、"
        "support('direct'|'conservative_policy')、evidence_ids。可另含 errors。不要输出推理。"
    )


def prepare_semantic_review_requests(
    scenarios: list[dict[str, Any]], outputs: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_by_id = {str(row.get("case_id")): row for row in outputs}
    output_counts = Counter(str(row.get("case_id")) for row in outputs)
    scenario_ids = {str(row.get("case_id")) for row in scenarios}
    requests: list[dict[str, Any]] = []
    rejected = Counter()
    unknown_outputs = sum(
        count for case_id, count in output_counts.items()
        if case_id not in scenario_ids
    )
    if unknown_outputs:
        rejected["unknown_teacher_output"] = unknown_outputs
    for scenario in scenarios:
        case_id = str(scenario["case_id"])
        if output_counts[case_id] > 1:
            rejected["duplicate_teacher_output"] += 1
            continue
        candidate = output_by_id.get(case_id)
        if candidate is None:
            rejected["missing_teacher_output"] += 1
            continue
        validation = validate_target(scenario, str(candidate.get("output", "")))
        if not validation["deterministic_pass"]:
            for reason in validation["errors"]:
                rejected[reason] += 1
            continue
        requests.append({
            "schema_version": "targeted-semantic-review-request.v1",
            "case_id": case_id,
            "instruction": _semantic_review_instruction(),
            "user_prompt": scenario.get("prompt", ""),
            "candidate_text": validation["text"],
            "medical_context": scenario.get("medical_context", {}),
            "allowed_evidence_ids": scenario.get("supervision", {}).get(
                "evidence_ids", []
            ),
            "evidence_gap": bool(
                scenario.get("supervision", {}).get("evidence_gap")
            ),
            "teacher_model": candidate.get("teacher_model"),
        })
    report = {
        "schema_version": "targeted-semantic-review-preparation-report.v1",
        "scenario_count": len(scenarios),
        "teacher_output_count": len(outputs),
        "review_request_count": len(requests),
        "rejected": dict(rejected),
    }
    return requests, report


def compile_reviewed(
    scenarios: list[dict[str, Any]], outputs: list[dict[str, Any]],
    reviews: list[dict[str, Any]], *, human_review_required: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_by_id = {str(row.get("case_id")): row for row in outputs}
    review_by_id = {str(row.get("case_id")): row for row in reviews}
    output_counts = Counter(str(row.get("case_id")) for row in outputs)
    review_counts = Counter(str(row.get("case_id")) for row in reviews)
    scenario_ids = {str(row.get("case_id")) for row in scenarios}
    samples = []
    rejected = Counter()
    unknown_outputs = sum(
        count for case_id, count in output_counts.items()
        if case_id not in scenario_ids
    )
    unknown_reviews = sum(
        count for case_id, count in review_counts.items()
        if case_id not in scenario_ids
    )
    if unknown_outputs:
        rejected["unknown_teacher_output"] = unknown_outputs
    if unknown_reviews:
        rejected["unknown_semantic_review"] = unknown_reviews
    for scenario in scenarios:
        case_id = str(scenario["case_id"])
        if output_counts[case_id] > 1:
            rejected["duplicate_teacher_output"] += 1
            continue
        if review_counts[case_id] > 1:
            rejected["duplicate_semantic_review"] += 1
            continue
        candidate = output_by_id.get(case_id)
        review = review_by_id.get(case_id)
        if candidate is None:
            rejected["missing_teacher_output"] += 1
            continue
        validation = validate_target(scenario, str(candidate.get("output", "")))
        if not validation["deterministic_pass"]:
            for reason in validation["errors"]:
                rejected[reason] += 1
            continue
        if (
            review is None
            or review.get("schema_version") != REVIEW_SCHEMA
            or review.get("decision") != "approve"
        ):
            rejected["semantic_review_not_approved"] += 1
            continue
        allowed_ids = set(scenario["supervision"].get("evidence_ids", []))
        mappings = review.get("claim_evidence_map")
        if not isinstance(mappings, list) or any(
            not isinstance(row, dict)
            or not row.get("claim")
            or row.get("kind") not in {"medical_fact", "safety_advice", "uncertainty"}
            or row.get("support") not in {"direct", "conservative_policy"}
            or not set(row.get("evidence_ids") or []).issubset(allowed_ids)
            or (
                row.get("kind") == "medical_fact"
                and not row.get("evidence_ids")
            )
            for row in mappings
        ) or not mappings:
            rejected["invalid_claim_evidence_map"] += 1
            continue
        sample = {
            "schema_version": SAMPLE_SCHEMA,
            "case_id": case_id,
            "category": scenario["category"],
            "external_turn": 1,
            "agent_iteration": 1,
            "action": "act",
            "tags": {
                "conversation_type": "single_turn",
                "medical_grounded": True,
                "targeted_decision_role": "medical_final_synthesis",
                "split": scenario["split"],
                "semantic_family_id": scenario["semantic_family_id"],
            },
            "input": scenario["input"],
            "output": str(candidate["output"]).strip(),
            "final_answer": {
                "text": validation["text"],
                "source": "teacher_output.act.arguments.text",
                "sha256": _sha(validation["text"]),
            },
            "medical_context": scenario["medical_context"],
            "supervision": {
                **scenario["supervision"],
                "claim_evidence_map": mappings,
            },
            "provenance": {
                **scenario["provenance"],
                "scenario_sha256": scenario["scenario_sha256"],
                "teacher_model": candidate.get("teacher_model"),
                "semantic_review": review,
                "reasoning_removed": True,
            },
            "gate": {
                "deterministic": "pass",
                "claim_evidence": "pass",
                "semantic_review": "approve",
                "human_spot_review": (
                    "required" if human_review_required else "waived_by_user_ai_review"
                ),
            },
        }
        sample["sample_sha256"] = _sha(sample)
        samples.append(sample)
    report = {
        "schema_version": "targeted-sft-compile-report.v1",
        "scenario_count": len(scenarios), "sample_count": len(samples),
        "rejected": dict(rejected),
        "quality_gate": "deterministic pass AND evidence-only semantic approve",
        "human_review_required": human_review_required,
        "trainable": not human_review_required,
    }
    return samples, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _read_many_jsonl(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        rows.extend(_read_jsonl(Path(path)))
    return rows


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="定向 Agent SFT v3 canary 生成与编译")
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--count", type=int, default=300)
    generate.add_argument("--seed", type=int, default=20260821)
    generate.add_argument("--evidence-gap-count", type=int, default=0)
    generate.add_argument("--database", default="med_database/med_search.sqlite")
    generate.add_argument("--output-dir", required=True)
    compile_parser = subparsers.add_parser("compile")
    compile_parser.add_argument("--scenario-dir", required=True)
    compile_parser.add_argument("--teacher-outputs", required=True, nargs="+")
    compile_parser.add_argument("--semantic-reviews", required=True, nargs="+")
    compile_parser.add_argument("--output-dir", required=True)
    compile_parser.add_argument(
        "--ai-review-only", action="store_true",
        help="接受独立 AI 证据审核作为最终门禁，不再要求人工抽检",
    )
    review_parser = subparsers.add_parser("prepare-review")
    review_parser.add_argument("--scenario-dir", required=True)
    review_parser.add_argument("--teacher-outputs", required=True, nargs="+")
    review_parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.command == "generate":
        repository = Path(__file__).resolve().parents[2]
        try:
            scenarios, report = asyncio.run(generate_scenarios(
                count=args.count, database=Path(args.database), seed=args.seed,
                repository=repository, evidence_gap_count=args.evidence_gap_count,
            ))
        finally:
            # The production process intentionally keeps its shared retrieval
            # pool alive. This one-shot CLI must close it even after failure.
            from agents.Medical.retriever import _RETRIEVAL_EXECUTOR

            _RETRIEVAL_EXECUTOR.shutdown(wait=True)
        _write_jsonl(output / "scenarios.jsonl", scenarios)
        _write_jsonl(output / "teacher_requests.jsonl", ({
            "schema_version": "targeted-teacher-request.v1",
            "case_id": row["case_id"],
            "instruction": _teacher_instruction(),
            "prompt": row["prompt"],
            "medical_context": row["medical_context"],
            "supervision": row["supervision"],
            "decision_frame": {
                "current_step_id": "s_final",
                "completed_tools": ["medical_consult"],
                "allowed_tools": ["speak"],
                "forbidden_tools": row["supervision"]["forbidden_tools"],
                "requested_aspect": row["supervision"]["intent_aspect"],
                "evidence_gap": row["supervision"]["evidence_gap"],
                "max_output_chars": row["supervision"]["max_output_chars"],
            },
        } for row in scenarios))
        (output / "generation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    if args.command == "prepare-review":
        scenarios = _read_jsonl(Path(args.scenario_dir) / "scenarios.jsonl")
        requests, report = prepare_semantic_review_requests(
            scenarios, _read_many_jsonl(args.teacher_outputs)
        )
        _write_jsonl(output / "semantic_review_requests.jsonl", requests)
        (output / "review_preparation_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    scenarios = _read_jsonl(Path(args.scenario_dir) / "scenarios.jsonl")
    samples, report = compile_reviewed(
        scenarios, _read_many_jsonl(args.teacher_outputs),
        _read_many_jsonl(args.semantic_reviews),
        human_review_required=not args.ai_review_only,
    )
    _write_jsonl(output / "train.jsonl", samples)
    _write_jsonl(output / "sft_decisions.jsonl", samples)
    _write_jsonl(output / "case_index.jsonl", ({
        "case_id": row["case_id"], "category": row["category"],
        "split": row["tags"]["split"], "sample_count": 1,
        "semantic_family_id": row["tags"]["semantic_family_id"],
    } for row in samples))
    (output / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
