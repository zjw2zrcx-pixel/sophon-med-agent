"""Case-level semantic/ASR augmentation pilot.

The frozen ``final_v7_1000`` directory is read-only input.  This module writes
all derived artifacts into a new output directory and keeps the root trajectory
hash, semantic contract and variant lineage on every derived case.

The first implementation deliberately uses a deterministic generator.  It is
small enough to audit and provides the complete data contract before an LLM
generator is enabled.  ``--generator deepseek`` is reserved for a later stage;
the validator and compiler are shared by both generators.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from agents.MCP.tools.navigate import HOSPITAL_LOCATIONS, LOCATION_ALIASES


SCHEMA = "teacher-augmentation-v1"
ROOT_SCHEMA = "teacher-root-case.v1"
VARIANT_SCHEMA = "teacher-augmented-case.v1"


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: Any) -> str:
    if isinstance(value, (str, bytes)):
        raw = value.encode("utf-8") if isinstance(value, str) else value
    else:
        raw = _canonical(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).lower()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value))


_PROTECTED_MARKERS = (
    "不是", "不要", "别", "不想", "不需要", "不记得", "不确定", "不知道",
    "可能", "大概", "对吧", "那个", "还没", "暂时没",
)


def _marker_preserved(marker: str, text: str) -> bool:
    """Allow only conservative paraphrases for a protected surface marker."""
    alternatives = {
        "还没": ("还没", "暂时没", "未提供", "没有提供", "没说"),
        "暂时没": ("还没", "暂时没", "未提供", "没有提供", "没说"),
        "不确定": ("不确定", "不清楚", "不知道"),
        "不知道": ("不知道", "不清楚", "不确定"),
        "不是": ("不是", "并非", "而非"),
    }
    return any(option in str(text or "") for option in alternatives.get(marker, (marker,)))


def _protected_surface(prompt: str) -> dict[str, list[str]]:
    text = str(prompt or "")
    return {
        "markers": [marker for marker in _PROTECTED_MARKERS if marker in text],
        "numbers": sorted(set(re.findall(r"\d+(?:\.\d+)?", text))),
    }


def _recover_location(text: str) -> str:
    """Recover the longest registered location before checking aliases.

    The runtime navigation matcher historically checks registered locations in
    declaration order; for augmentation validation we must avoid the shorter
    ``产科`` matching inside the alias ``妇产科`` before alias normalization.
    """
    for alias, target in sorted(LOCATION_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in str(text or ""):
            return target
    for location in sorted(HOSPITAL_LOCATIONS, key=len, reverse=True):
        if location in str(text or ""):
            return location
    return ""


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object expected: {path}")
    return payload


def _trajectory_hash(path: Path) -> str:
    parts = []
    for file in sorted(path.rglob("*")):
        if not file.is_file():
            continue
        parts.append({
            "name": str(file.relative_to(path)),
            "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
        })
    if not parts:
        raise FileNotFoundError(f"trajectory directory is empty: {path}")
    return _sha(parts)


def _tree_hash(path: Path) -> str:
    """Hash a directory without changing it, for frozen-root auditability."""
    parts = []
    for file in sorted(path.rglob("*")):
        if file.is_file():
            parts.append({
                "name": str(file.relative_to(path)),
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
            })
    return _sha(parts)


def _line_count(path: Path) -> int:
    with path.open(encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _source_index(root_dir: Path) -> dict[str, dict[str, Any]]:
    payload = _read_json(root_dir / "trajectory_sources.json")
    return {str(row["final_case_id"]): row for row in payload.get("sources", [])}


def semantic_contract(case: dict[str, Any], trajectory_hash: str) -> dict[str, Any]:
    expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
    source = case.get("medical_source") if isinstance(case.get("medical_source"), dict) else {}
    turns = [str(value) for value in (case.get("turns") or [case.get("prompt", "")])]
    contract = {
        "schema_version": "teacher-semantic-contract.v1",
        "root_case_id": str(case["id"]),
        "category": str(case.get("category", "")),
        "intent": {
            "required_tools": sorted(str(value) for value in expected.get("required_tools", [])),
            "forbidden_tools": sorted(str(value) for value in expected.get("forbidden_tools", [])),
            "navigation_target": str(expected.get("navigation_target") or "") or None,
        },
        "critical_entities": {
            "navigation_target": str(expected.get("navigation_target") or "") or None,
            "medical_question": str(source.get("question") or "") or None,
            "medical_answer_sha256": str(source.get("answer_sha256") or "") or None,
            "medical_row_id": source.get("row_id"),
        },
        "constraints": {
            "turn_count": len(turns),
            "success_notes": str(expected.get("notes") or ""),
            "risk_tags": sorted(str(value) for value in case.get("risk_tags", [])),
            "protected_surface": _protected_surface(turns[0] if turns else ""),
        },
        "trajectory_hash": trajectory_hash,
    }
    contract["semantic_signature"] = _sha({
        key: value for key, value in contract.items() if key != "trajectory_hash"
    })
    return contract


def _replace_once(text: str, old: str, new: str) -> str:
    index = str(text).find(old)
    if index < 0:
        return str(text)
    return str(text)[:index] + new + str(text)[index + len(old):]


def _manual_semantic_candidates(case: dict[str, Any], count: int) -> list[dict[str, Any]]:
    """Generate conservative, auditable candidates without inventing facts."""
    category = str(case.get("category", ""))
    prompt = str(case.get("prompt", "")).strip()
    expected = case.get("expected") or {}
    target = str(expected.get("navigation_target") or "").strip()
    candidates: list[tuple[str, list[str]]] = []
    if category == "navigation" and target:
        candidates = [
            (f"麻烦带我到{target}。", ["direct_polite"]),
            (f"我想去{target}，请给我带路。", ["colloquial"]),
            (f"能指引我到{target}吗？", ["indirect"]),
        ]
    elif category == "mixed" and target:
        # Keep the medical wording untouched in the candidate, changing only
        # the request framing around the existing medical question and target.
        medical = prompt
        marker = f"，另外请带我去{target}。"
        if marker in medical:
            medical = medical.replace(marker, "")
        candidates = [
            (f"{medical}，查完后麻烦再带我到{target}。", ["constraint_rich"]),
            (f"先帮我看看这个问题，另外我还要去{target}，请带路。", ["mixed_intent"]),
        ]
    elif category == "general":
        required = set(expected.get("required_tools", []))
        if "get_time" in required:
            candidates = [
                ("现在是什么时间？", ["elliptical"]),
                ("能报一下当前时间吗？", ["direct_polite"]),
            ]
        elif "get_system_stats" in required:
            candidates = [
                ("帮我看下机器现在运行得怎么样。", ["colloquial"]),
                ("能查一下当前系统状态吗？", ["direct_polite"]),
            ]
        else:
            candidates = [(f"请回答：{prompt}", ["direct_polite"])]
    elif category == "medical":
        # The pilot only rewrites the controlled masked first turn.  For a
        # fully specified medical question, changing wording can change the
        # database query; those cases are intentionally deferred to the
        # contract-aware LLM generator.
        if "还没说具体名称" in prompt:
            candidates = [
                (prompt.replace("我想咨询一种疾病", "我想了解某种疾病"), ["colloquial"]),
                (prompt.replace("但还没说具体名称", "具体名称暂时没提供"), ["indirect"]),
            ]
    unique = []
    seen = {_norm(prompt)}
    for value, tags in candidates:
        value = value.strip()
        if not value or _norm(value) in seen:
            continue
        seen.add(_norm(value))
        unique.append({"prompt": value, "style_tags": tags})
        if len(unique) >= count:
            break
    return unique


def _turn_variant(case: dict[str, Any], candidate_prompt: str) -> list[str]:
    turns = [str(value) for value in (case.get("turns") or [case.get("prompt", "")])]
    return [candidate_prompt] + turns[1:]


def validate_semantic_candidate(
    case: dict[str, Any], contract: dict[str, Any], candidate_prompt: str,
    candidate_turns: list[str], style_tags: Iterable[str], *,
    strict_surface: bool = True,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected = case.get("expected") or {}
    if not candidate_prompt.strip():
        reasons.append("EMPTY_PROMPT")
    if _norm(candidate_prompt) == _norm(str(case.get("prompt", ""))):
        reasons.append("UNCHANGED_PROMPT")
    if len(candidate_turns) != int(contract["constraints"]["turn_count"]):
        reasons.append("TURN_COUNT_CHANGED")
    if not candidate_turns or candidate_turns[0] != candidate_prompt:
        reasons.append("FIRST_TURN_MISMATCH")
    if any(not str(tag).strip() for tag in style_tags):
        reasons.append("EMPTY_STYLE_TAG")
    protected = contract["constraints"].get("protected_surface") or {}
    for marker in protected.get("markers", []):
        if not _marker_preserved(marker, candidate_prompt):
            reasons.append(f"PROTECTED_MARKER_DROPPED:{marker}")
    for number in protected.get("numbers", []):
        if number not in candidate_prompt:
            reasons.append(f"PROTECTED_NUMBER_DROPPED:{number}")
    target = str(contract["critical_entities"].get("navigation_target") or "")
    if target:
        actual = _recover_location(candidate_prompt)
        if actual != target:
            reasons.append(f"NAVIGATION_TARGET_NOT_RECOVERABLE:{actual or 'none'}")
    source = contract["critical_entities"]
    if source.get("medical_answer_sha256"):
        # A masked medical turn must remain masked; a specified medical turn
        # is not rewritten by the deterministic pilot.
        original = str(case.get("prompt", ""))
        if "还没说具体名称" in original and any(
            token in candidate_prompt for token in ("阿莫西林", "头孢", "剂量")
        ):
            reasons.append("MEDICAL_CRITICAL_ENTITY_INVENTED")
        medical_question = str(source.get("medical_question") or "")
        if medical_question:
            if strict_surface and str(case.get("category", "")) == "mixed":
                if medical_question in original and medical_question not in candidate_prompt:
                    reasons.append("MIXED_MEDICAL_QUERY_CHANGED")
            elif not strict_surface:
                # Flash may paraphrase a mixed/medical question, but its
                # primary subject must remain explicit.  The reviewer and
                # database evidence handle the rest of the semantic contract.
                subject = re.split(r"[的是？?]", medical_question, maxsplit=1)[0].strip()
                if subject and subject in original and subject not in candidate_prompt:
                    reasons.append("MEDICAL_SUBJECT_CHANGED")
    if set(expected.get("required_tools", [])) != set(contract["intent"]["required_tools"]):
        reasons.append("SOURCE_CONTRACT_MUTATED")
    return not reasons, reasons


def _asr_noise(case: dict[str, Any], prompt: str) -> list[dict[str, Any]]:
    """Return only safe/recoverable pilot noises; no random critical typos."""
    category = str(case.get("category", ""))
    expected = case.get("expected") or {}
    target = str(expected.get("navigation_target") or "")
    rows: list[dict[str, Any]] = []
    if target:
        aliases = [alias for alias, canonical in LOCATION_ALIASES.items() if canonical == target]
        if aliases:
            alias = sorted(aliases, key=len, reverse=True)[0]
            if target in prompt:
                rows.append({
                    "prompt": prompt.replace(target, alias, 1),
                    "noise": [{"type": "registered_alias_substitution", "source": target, "output": alias}],
                    "severity": "recoverable",
                })
    if "，" in prompt or "。" in prompt or "？" in prompt:
        rows.append({
            "prompt": re.sub(r"[，。！？；]", "", prompt),
            "noise": [{"type": "punctuation_deletion", "source": "punctuation", "output": "omitted"}],
            "severity": "safe",
        })
    if category in {"general", "navigation", "mixed"} and prompt:
        rows.append({
            "prompt": "呃，" + prompt,
            "noise": [{"type": "filler_insertion", "source": "", "output": "呃"}],
            "severity": "safe",
        })
    return rows


def validate_asr_candidate(
    case: dict[str, Any], semantic_prompt: str, noisy_prompt: str, *,
    strict_surface: bool = True,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    expected = case.get("expected") or {}
    if _norm(noisy_prompt) == _norm(semantic_prompt):
        reasons.append("UNCHANGED_ASR")
    target = str(expected.get("navigation_target") or "")
    if target:
        recovered = _recover_location(noisy_prompt)
        if recovered != target:
            reasons.append(f"CRITICAL_NAV_SLOT_NOT_RECOVERABLE:{recovered or 'none'}")
    source = case.get("medical_source") or {}
    question = str(source.get("question") or "")
    # A fully specified medical entity must remain literally present in this
    # bootstrap simulator.  We do not train the model to guess a different
    # disease/drug from an ambiguous ASR output.
    if question:
        match = re.search(r"(.+?)的", question)
        subject = match.group(1).strip() if match else ""
        original = str(case.get("prompt", ""))
        if subject and subject in original and subject not in noisy_prompt:
            reasons.append("MEDICAL_CRITICAL_ENTITY_CHANGED")
    # Mixed cases contain a medical query plus navigation.  The bootstrap
    # generator may only reframe the request; it must not silently drop or
    # replace the source medical question.
    if str(case.get("category", "")) == "mixed" and question and strict_surface:
        original = str(case.get("prompt", ""))
        if question in original and question not in noisy_prompt:
            reasons.append("MIXED_MEDICAL_QUERY_CHANGED")
    elif str(case.get("category", "")) == "mixed" and question and not strict_surface:
        original = str(case.get("prompt", ""))
        subject = re.split(r"[的是？?]", question, maxsplit=1)[0].strip()
        if subject and subject in original and subject not in noisy_prompt:
            reasons.append("MEDICAL_SUBJECT_CHANGED")
    return not reasons, reasons


def _replace_surface(row: dict[str, Any], root_case: dict[str, Any], variant_turns: list[str], variant_id: str, root_hash: str, lineage: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(row)
    external_turn = max(1, int(row.get("external_turn") or 1))
    root_turns = [str(value) for value in (root_case.get("turns") or [root_case.get("prompt", "")])]
    old = root_turns[min(external_turn - 1, len(root_turns) - 1)]
    new = variant_turns[min(external_turn - 1, len(variant_turns) - 1)]
    prompt_slots = result.get("input", {}).get("prompt_slots", {})
    for key in ("user", "history", "conversation", "attempt"):
        if isinstance(prompt_slots.get(key), str):
            prompt_slots[key] = _replace_once(prompt_slots[key], old, new)
    result["input"]["prompt_slots"] = prompt_slots
    result["output"] = _replace_once(str(result.get("output", "")), old, new)
    result["case_id"] = variant_id
    result["original_case_id"] = str(root_case["id"])
    result.setdefault("tags", {})["augmentation"] = lineage
    result.setdefault("provenance", {})["augmentation"] = lineage
    result["sample_sha256"] = _sha({key: value for key, value in result.items() if key != "sample_sha256"})
    return result


def run_pipeline(root_dir: Path, output_dir: Path, root_limit: int, variants_per_root: int, asr_per_variant: int) -> dict[str, Any]:
    prompts = _read_json(root_dir / "prompts.json")
    source_index = _source_index(root_dir)
    all_cases = list(prompts.get("cases") or [])
    if root_limit > 0:
        # Stable pilot sample: one hash-stable case per category first, then
        # fill the remaining slots from the global hash order.  A previous
        # generator expression could consume the whole limit in the first
        # category, making a nominally balanced pilot medical-only.
        ordered = sorted(all_cases, key=lambda case: _sha(str(case["id"])))
        selected: list[dict[str, Any]] = []
        for category in ("medical", "navigation", "mixed", "general"):
            pool = [case for case in ordered if case.get("category") == category]
            if pool and len(selected) < root_limit:
                selected.append(pool[0])
        selected_ids = {str(case["id"]) for case in selected}
        selected.extend(
            case for case in ordered
            if str(case["id"]) not in selected_ids and len(selected) < root_limit
        )
        all_cases = selected[:root_limit]
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("semantic_contracts.jsonl", "semantic_candidates.jsonl", "semantic_accepted.jsonl", "semantic_rejected.jsonl", "asr_candidates.jsonl", "asr_accepted.jsonl", "asr_rejected.jsonl", "augmented_cases.jsonl", "lineage.jsonl", "prompt_mapping.jsonl"):
        (output_dir / name).write_text("", encoding="utf-8")
    root_rows: list[dict[str, Any]] = []
    accepted_semantic: list[dict[str, Any]] = []
    accepted_asr: list[dict[str, Any]] = []
    rejected_semantic: list[dict[str, Any]] = []
    rejected_asr: list[dict[str, Any]] = []
    lineages: list[dict[str, Any]] = []
    mapping_rows: dict[str, dict[str, Any]] = {}
    with (output_dir / "semantic_contracts.jsonl").open("a", encoding="utf-8") as contracts, \
         (output_dir / "semantic_candidates.jsonl").open("a", encoding="utf-8") as candidates_file, \
         (output_dir / "semantic_accepted.jsonl").open("a", encoding="utf-8") as sem_ok_file, \
         (output_dir / "semantic_rejected.jsonl").open("a", encoding="utf-8") as sem_bad_file, \
         (output_dir / "asr_candidates.jsonl").open("a", encoding="utf-8") as asr_candidates_file, \
         (output_dir / "asr_accepted.jsonl").open("a", encoding="utf-8") as asr_ok_file, \
         (output_dir / "asr_rejected.jsonl").open("a", encoding="utf-8") as asr_bad_file, \
         (output_dir / "augmented_cases.jsonl").open("a", encoding="utf-8") as augmented_file, \
         (output_dir / "lineage.jsonl").open("a", encoding="utf-8") as lineage_file:
        for root_case in all_cases:
            root_id = str(root_case["id"])
            source = source_index.get(root_id)
            if not source:
                raise KeyError(f"root case missing trajectory source: {root_id}")
            trajectory_dir = root_dir / source["materialized"]
            root_hash = _trajectory_hash(trajectory_dir)
            contract = semantic_contract(root_case, root_hash)
            contracts.write(_canonical(contract) + "\n")
            original_lineage = {
                "case_id": root_id, "root_case_id": root_id,
                "variant_type": "original", "trajectory_hash": root_hash,
                "semantic_signature": contract["semantic_signature"],
                "trajectory_changed": False,
            }
            lineages.append(original_lineage)
            root_rows.append({"case": root_case, "contract": contract, "trajectory_hash": root_hash, "source": source})
            mapping_rows[root_id] = {
                "schema_version": "teacher-prompt-replacement-map.v1",
                "root_case_id": root_id,
                "category": root_case.get("category"),
                "original": {
                    "prompt": root_case.get("prompt"),
                    "turns": root_case.get("turns"),
                    "trajectory_hash": root_hash,
                    "semantic_signature": contract["semantic_signature"],
                    "replacement_allowed": False,
                },
                "replacements": [],
                "rejected_candidates": [],
                "replacement_policy": {
                    "trajectory_reused": True,
                    "semantic_contract_must_match": True,
                    "generation_order": "original -> semantic -> asr",
                    "asr_parent_must_be_accepted_semantic": True,
                    "asr_must_be_uniquely_recoverable": True,
                },
            }
            augmented_file.write(_canonical({
                "schema_version": VARIANT_SCHEMA, "case_id": root_id,
                "root_case_id": root_id, "variant_type": "original",
                "prompt": root_case.get("prompt"), "turns": root_case.get("turns"),
                "trajectory_hash": root_hash, "semantic_signature": contract["semantic_signature"],
            }) + "\n")
            for sem_index, item in enumerate(_manual_semantic_candidates(root_case, variants_per_root), 1):
                candidate = {
                    "root_case_id": root_id, "semantic_variant_id": sem_index,
                    "prompt": item["prompt"], "style_tags": item.get("style_tags", []),
                    "turns": _turn_variant(root_case, item["prompt"]),
                    "semantic_signature": contract["semantic_signature"],
                }
                candidates_file.write(_canonical(candidate) + "\n")
                valid, reasons = validate_semantic_candidate(
                    root_case, contract, candidate["prompt"], candidate["turns"], candidate["style_tags"]
                )
                candidate["decision"] = "accept" if valid else "reject"
                candidate["reasons"] = reasons
                (sem_ok_file if valid else sem_bad_file).write(_canonical(candidate) + "\n")
                if not valid:
                    rejected_semantic.append(candidate)
                    mapping_rows[root_id]["rejected_candidates"].append({
                        "variant_type": "semantic",
                        "prompt": candidate["prompt"],
                        "semantic_variant_id": sem_index,
                        "reasons": reasons,
                    })
                    continue
                accepted_semantic.append(candidate)
                semantic_id = f"{_safe(root_id)}__sem{sem_index:02d}"
                sem_lineage = {
                    "case_id": semantic_id, "root_case_id": root_id,
                    "parent_variant": root_id, "variant_type": "semantic",
                    "semantic_variant_id": sem_index, "asr_variant_id": None,
                    "trajectory_hash": root_hash,
                    "semantic_signature": contract["semantic_signature"],
                    "trajectory_changed": False, "critical_slots_preserved": True,
                    "semantic_validation": "PASS", "style_tags": item.get("style_tags", []),
                }
                lineages.append(sem_lineage)
                mapping_rows[root_id]["replacements"].append({
                    "case_id": semantic_id,
                    "variant_type": "semantic",
                    "parent_variant": root_id,
                    "semantic_variant_id": sem_index,
                    "asr_variant_id": None,
                    "prompt": candidate["prompt"],
                    "turns": candidate["turns"],
                    "trajectory_hash": root_hash,
                    "semantic_signature": contract["semantic_signature"],
                    "validation": {"semantic": "PASS", "asr_recoverability": None},
                    "style_tags": item.get("style_tags", []),
                    "trajectory_changed": False,
                })
                augmented_file.write(_canonical({
                    "schema_version": VARIANT_SCHEMA, "case_id": semantic_id,
                    "root_case_id": root_id, "variant_type": "semantic",
                    "semantic_variant_id": sem_index, "prompt": candidate["prompt"],
                    "turns": candidate["turns"], "trajectory_hash": root_hash,
                    "semantic_signature": contract["semantic_signature"],
                }) + "\n")
                lineage_file.write(_canonical(sem_lineage) + "\n")
                for asr_index, noise in enumerate(_asr_noise(root_case, candidate["prompt"])[:max(0, asr_per_variant)], 1):
                    asr_candidate = {
                        "root_case_id": root_id, "parent_variant": semantic_id,
                        "semantic_variant_id": sem_index, "asr_variant_id": asr_index,
                        "prompt": noise["prompt"], "noise": noise["noise"],
                        "severity": noise["severity"], "trajectory_hash": root_hash,
                    }
                    asr_candidates_file.write(_canonical(asr_candidate) + "\n")
                    asr_valid, asr_reasons = validate_asr_candidate(root_case, candidate["prompt"], noise["prompt"])
                    asr_candidate["decision"] = "accept" if asr_valid else "reject"
                    asr_candidate["reasons"] = asr_reasons
                    (asr_ok_file if asr_valid else asr_bad_file).write(_canonical(asr_candidate) + "\n")
                    if not asr_valid:
                        rejected_asr.append(asr_candidate)
                        mapping_rows[root_id]["rejected_candidates"].append({
                            "variant_type": "asr",
                            "parent_variant": semantic_id,
                            "semantic_variant_id": sem_index,
                            "asr_variant_id": asr_index,
                            "prompt": noise["prompt"],
                            "noise": noise["noise"],
                            "reasons": asr_reasons,
                        })
                        continue
                    accepted_asr.append(asr_candidate)
                    asr_id = f"{semantic_id}__asr{asr_index:02d}"
                    asr_lineage = {
                        "case_id": asr_id, "root_case_id": root_id,
                        "parent_variant": semantic_id, "variant_type": "asr",
                        "semantic_variant_id": sem_index, "asr_variant_id": asr_index,
                        "trajectory_hash": root_hash,
                        "semantic_signature": contract["semantic_signature"],
                        "trajectory_changed": False, "critical_slots_preserved": True,
                        "semantic_validation": "PASS", "asr_recoverability": "PASS",
                        "asr_noise": noise["noise"],
                    }
                    lineages.append(asr_lineage)
                    mapping_rows[root_id]["replacements"].append({
                        "case_id": asr_id,
                        "variant_type": "asr",
                        "parent_variant": semantic_id,
                        "semantic_variant_id": sem_index,
                        "asr_variant_id": asr_index,
                        "prompt": noise["prompt"],
                        "turns": [noise["prompt"]] + candidate["turns"][1:],
                        "trajectory_hash": root_hash,
                        "semantic_signature": contract["semantic_signature"],
                        "validation": {"semantic": "PASS", "asr_recoverability": "PASS"},
                        "noise": noise["noise"],
                        "severity": noise["severity"],
                        "trajectory_changed": False,
                    })
                    augmented_file.write(_canonical({
                        "schema_version": VARIANT_SCHEMA, "case_id": asr_id,
                        "root_case_id": root_id, "parent_variant": semantic_id,
                        "variant_type": "asr", "prompt": noise["prompt"],
                        "turns": [noise["prompt"]] + candidate["turns"][1:],
                        "trajectory_hash": root_hash,
                        "semantic_signature": contract["semantic_signature"],
                    }) + "\n")
                    lineage_file.write(_canonical(asr_lineage) + "\n")
        for row in lineages:
            if row["variant_type"] == "original":
                lineage_file.write(_canonical(row) + "\n")
    with (output_dir / "prompt_mapping.jsonl").open("w", encoding="utf-8") as mapping_file:
        for root_id in sorted(mapping_rows):
            row = mapping_rows[root_id]
            row["replacement_count"] = len(row["replacements"])
            row["rejected_count"] = len(row["rejected_candidates"])
            mapping_file.write(_canonical(row) + "\n")
    metadata = {
        "schema_version": SCHEMA, "root_schema": ROOT_SCHEMA,
        "root_dir": str(root_dir.resolve()), "root_case_count": len(root_rows),
        "root_snapshot_sha256": _tree_hash(root_dir),
        "semantic_candidates": _line_count(output_dir / "semantic_candidates.jsonl"),
        "semantic_accepted": len(accepted_semantic), "semantic_rejected": len(rejected_semantic),
        "asr_candidates": _line_count(output_dir / "asr_candidates.jsonl"),
        "asr_accepted": len(accepted_asr), "asr_rejected": len(rejected_asr),
        "variant_counts": dict(Counter(row["variant_type"] for row in lineages)),
        "generator": "deterministic_pilot_v1",
        "generation_order": ["original", "semantic", "asr_from_accepted_semantic"],
        "trajectory_reuse": "root trajectory hash shared; rendered SFT surface is compiled separately",
        "sampling_policy": "root_case_uniform_then_variant",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="运行 Root case 级语义/ASR 扩增 pilot")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-limit", type=int, default=8)
    parser.add_argument("--variants-per-root", type=int, default=2)
    parser.add_argument("--asr-per-variant", type=int, default=1, help="保留参数；pilot 当前输出所有安全噪声")
    args = parser.parse_args()
    metadata = run_pipeline(
        Path(args.root_dir).resolve(), Path(args.output_dir).resolve(),
        args.root_limit, args.variants_per_root, args.asr_per_variant,
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
