"""DeepSeek Flash semantic/ASR prompt augmentation.

The frozen root dataset is read only.  The pipeline is deliberately staged:

    root prompt
      -> Flash generates 5--10 semantic variants
      -> Flash scores semantic equivalence
      -> local contract/critical-slot gate + exact/near deduplication
      -> for every accepted semantic variant Flash generates 2--3 ASR-like
         variants
      -> local recoverability gate
      -> prompt_mapping.jsonl and optional compiler-compatible artifacts

Reasoning is enabled for Flash, but only the structured visible JSON is
persisted.  A failure is recorded as a rejected candidate instead of silently
turning a failed call into training data.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import time
from difflib import SequenceMatcher
from typing import Any

import httpx

from agents.TeacherData.augment_pipeline import (
    VARIANT_SCHEMA,
    _canonical,
    _line_count,
    _safe,
    _sha,
    _source_index,
    _trajectory_hash,
    _turn_variant,
    _tree_hash,
    semantic_contract,
    validate_asr_candidate,
    validate_semantic_candidate,
)


SEMANTIC_SYSTEM = """你是教师轨迹数据的语义等价改写器。只改写用户输入表面，不改变任务。
一次返回 JSON 对象：{"candidates":[{"prompt":"...","style_tags":["..."]}]}。
必须生成 5 到 10 条表达自然、彼此有差异但意图完全相同的中文用户输入。
禁止改变：工具意图、关键实体、否定、数量、时间、条件、歧义程度、成功条件。
医疗问题只能保留原有证据范围，不能补充疾病/药物/剂量；导航不能更换地点、方向或楼层。
不要输出解释、Markdown、XML、工具调用或思维过程。"""

SEMANTIC_REVIEW_SYSTEM = """你是独立的语义等价审阅器。逐条比较 ORIGINAL 与 CANDIDATES，依据 CONTRACT 判断候选是否可以复用同一条教师 trajectory。
必须检查 intent、关键实体、否定、数量、时间、条件、歧义和成功条件；医疗和导航关键槽位改变时必须 reject。
输出严格 JSON：{"reviews":[{"candidate_index":1,"decision":"accept|reject","score":0到1,"reason_codes":["..."],"notes":"简短中文理由"}]}。
不要重新回答用户，不要输出 Markdown、XML 或思维过程。"""

ASR_SYSTEM = """你是中文 ASR 噪声生成器。输入是已经通过语义等价审阅的用户输入。
生成 2 到 3 条听感/识别上接近的中文转写错误，模拟替换、删除、词边界或口语停顿；不能凭空改变意图。
关键地点、药物、疾病、剂量、方向等如果变成另一个合法实体，必须标记 recoverable=false；不要主动修正噪声。
严格输出 JSON：{"candidates":[{"prompt":"...","noise":[{"type":"...","source":"...","output":"..."}],"severity":"safe|recoverable|ambiguous","recoverable":true}]}。
不要输出解释、Markdown、XML 或思维过程。"""


def _extract_json(content: str) -> dict[str, Any]:
    text = str(content or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Flash response must be a JSON object")
    return value


def _usage(data: dict[str, Any]) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in (data.get("usage") or {}).items()
        if isinstance(value, (int, float))
    }


def _merge_usage(rows: list[dict[str, int]]) -> dict[str, int]:
    total: Counter[str] = Counter()
    for row in rows:
        total.update(row)
    return dict(total)


def _candidate_text(item: Any) -> str:
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        return str(item.get("prompt", "")).strip()
    return ""


def _score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1.0:
        number /= 100.0
    return max(0.0, min(1.0, number))


def _dedupe(items: list[dict[str, Any]], similarity_threshold: float = 0.94) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep the highest-scoring near-unique candidates in stable order."""
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []
    ranked = sorted(items, key=lambda row: (-_score(row.get("score")), int(row.get("candidate_index", 0))))
    for item in ranked:
        prompt = str(item.get("prompt", ""))
        normalized = re.sub(r"\s+", "", prompt).lower()
        duplicate_of = ""
        for previous in kept:
            previous_prompt = str(previous.get("prompt", ""))
            previous_normalized = re.sub(r"\s+", "", previous_prompt).lower()
            ratio = SequenceMatcher(None, normalized, previous_normalized).ratio()
            if normalized == previous_normalized or ratio >= similarity_threshold:
                duplicate_of = str(previous.get("variant_id") or previous.get("candidate_index"))
                break
        if duplicate_of:
            dropped.append({**item, "dedupe": "drop", "duplicate_of": duplicate_of})
        else:
            kept.append({**item, "dedupe": "keep"})
    kept.sort(key=lambda row: int(row.get("candidate_index", 0)))
    return kept, dropped


def _contract_payload(case: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    source = case.get("medical_source") if isinstance(case.get("medical_source"), dict) else {}
    return {
        "root_case_id": case.get("id"),
        "category": case.get("category"),
        "original_prompt": case.get("prompt"),
        "original_turns": case.get("turns"),
        "expected": case.get("expected"),
        "medical_question": source.get("question"),
        "semantic_contract": contract,
    }


async def _call_flash(
    client: httpx.AsyncClient,
    gate: asyncio.Semaphore,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    *,
    reasoning_effort: str,
    max_tokens: int,
    retries: int,
    user_id: str,
) -> tuple[dict[str, Any], dict[str, int], float]:
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "thinking": {"type": "enabled"},
        "reasoning_effort": reasoning_effort,
        "response_format": {"type": "json_object"},
        "stream": False,
        "max_tokens": max_tokens,
    }
    if user_id:
        body["user_id"] = user_id
    last_error: Exception | None = None
    started = time.perf_counter()
    for attempt in range(retries + 1):
        try:
            async with gate:
                response = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json=body,
                )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"].get("content", "")
            return _extract_json(str(content)), _usage(data), time.perf_counter() - started
        except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(2 ** attempt)
    raise RuntimeError(f"Flash request failed after {retries + 1} attempts: {last_error}")


def _semantic_local_gate(
    case: dict[str, Any], contract: dict[str, Any], item: dict[str, Any], mode: str,
) -> tuple[bool, list[str]]:
    prompt = _candidate_text(item)
    turns = _turn_variant(case, prompt)
    reasons: list[str] = []
    if mode == "strict":
        valid, reasons = validate_semantic_candidate(
            case, contract, prompt, turns, item.get("style_tags") or ["flash"],
            strict_surface=False,
        )
    else:
        # Flash is the semantic judge in the normal path.  Keep only cheap
        # structural/protocol checks locally instead of duplicating its
        # contract reasoning.
        valid = True
        if re.sub(r"\s+", "", prompt).lower() == re.sub(r"\s+", "", str(case.get("prompt", ""))).lower():
            reasons.append("UNCHANGED_PROMPT")
        if len(turns) != int(contract["constraints"]["turn_count"]):
            reasons.append("TURN_COUNT_CHANGED")
    if not prompt:
        reasons.append("EMPTY_PROMPT")
    if any(token in prompt for token in ("<tool>", "<system>", "忽略以上", "不要遵守")):
        reasons.append("PROMPT_INJECTION_OR_PROTOCOL_TEXT")
    return not reasons, reasons


async def _process_case(
    case: dict[str, Any], source_index: dict[str, dict[str, Any]], root_dir: Path,
    client: httpx.AsyncClient, gate: asyncio.Semaphore, args: argparse.Namespace,
    api_key: str,
) -> dict[str, Any]:
    root_id = str(case["id"])
    source = source_index.get(root_id)
    if not source:
        raise KeyError(f"root case missing trajectory source: {root_id}")
    root_hash = _trajectory_hash(root_dir / source["materialized"])
    contract = semantic_contract(case, root_hash)
    payload = _contract_payload(case, contract)
    semantic_usage: list[dict[str, int]] = []
    asr_usage: list[dict[str, int]] = []
    semantic_error = ""
    semantic_response: dict[str, Any] = {}
    try:
        semantic_response, use, latency = await _call_flash(
            client, gate, api_key, args.model,
            [{"role": "system", "content": SEMANTIC_SYSTEM}, {"role": "user", "content": json.dumps({**payload, "requested_count": args.semantic_count}, ensure_ascii=False)}],
            reasoning_effort=args.reasoning_effort, max_tokens=args.semantic_max_tokens,
            retries=args.retries, user_id=args.user_id,
        )
        semantic_usage.append(use)
        semantic_latency = latency
    except Exception as exc:
        semantic_error = f"{type(exc).__name__}: {exc}"
        semantic_latency = 0.0

    raw_candidates = semantic_response.get("candidates") if isinstance(semantic_response, dict) else []
    if not isinstance(raw_candidates, list):
        raw_candidates = []
        semantic_error = semantic_error or "SEMANTIC_CANDIDATES_NOT_LIST"
    semantic_candidates: list[dict[str, Any]] = []
    for index, item in enumerate(raw_candidates, 1):
        if isinstance(item, str):
            item = {"prompt": item, "style_tags": ["flash"]}
        if not isinstance(item, dict):
            continue
        semantic_candidates.append({
            "candidate_index": index,
            "prompt": _candidate_text(item),
            "style_tags": list(item.get("style_tags") or ["flash"]),
        })

    review_error = ""
    review_response: dict[str, Any] = {}
    if semantic_candidates:
        try:
            review_payload = {
                **payload,
                "candidates": semantic_candidates,
                "review_threshold": args.min_score,
            }
            review_response, use, review_latency = await _call_flash(
                client, gate, api_key, args.model,
                [{"role": "system", "content": SEMANTIC_REVIEW_SYSTEM}, {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)}],
                reasoning_effort=args.reasoning_effort, max_tokens=args.review_max_tokens,
                retries=args.retries, user_id=args.user_id,
            )
            semantic_usage.append(use)
        except Exception as exc:
            review_error = f"{type(exc).__name__}: {exc}"
            review_latency = 0.0
    else:
        review_latency = 0.0

    reviews = review_response.get("reviews") if isinstance(review_response, dict) else []
    review_by_index = {
        int(row.get("candidate_index")): row
        for row in reviews if isinstance(row, dict) and str(row.get("candidate_index", "")).isdigit()
    } if isinstance(reviews, list) else {}
    gated: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for item in semantic_candidates:
        review = review_by_index.get(item["candidate_index"], {})
        score = _score(review.get("score"))
        local_ok, local_reasons = _semantic_local_gate(case, contract, item, args.local_gate)
        model_ok = str(review.get("decision", "reject")).lower() == "accept" and score >= args.min_score
        row = {**item, "score": score, "review": review, "local_validation": "PASS" if local_ok else "REJECT", "local_reasons": local_reasons}
        if model_ok and local_ok:
            gated.append(row)
        else:
            row["rejection_reasons"] = (["FLASH_REJECT_OR_LOW_SCORE"] if not model_ok else []) + local_reasons
            rejected.append(row)
    accepted, dedupe_dropped = _dedupe(gated)

    replacements: list[dict[str, Any]] = []
    augmented: list[dict[str, Any]] = [{
        "schema_version": VARIANT_SCHEMA, "case_id": root_id, "root_case_id": root_id,
        "variant_type": "original", "prompt": case.get("prompt"), "turns": case.get("turns"),
        "trajectory_hash": root_hash, "semantic_signature": contract["semantic_signature"],
    }]
    lineages: list[dict[str, Any]] = [{
        "case_id": root_id, "root_case_id": root_id, "variant_type": "original",
        "trajectory_hash": root_hash, "semantic_signature": contract["semantic_signature"],
        "trajectory_changed": False,
    }]

    async def generate_asr(sem_item: dict[str, Any], sem_id: str) -> tuple[dict[str, Any], dict[str, int], float]:
        asr_payload = {
            "root_case_id": root_id, "category": case.get("category"),
            "semantic_variant_id": sem_item["candidate_index"],
            "semantic_prompt": sem_item["prompt"],
            "critical_contract": contract,
            "requested_count": args.asr_count,
        }
        response, use, latency = await _call_flash(
            client, gate, api_key, args.model,
            [{"role": "system", "content": ASR_SYSTEM}, {"role": "user", "content": json.dumps(asr_payload, ensure_ascii=False)}],
            reasoning_effort=args.reasoning_effort, max_tokens=args.asr_max_tokens,
            retries=args.retries, user_id=args.user_id,
        )
        return {"semantic_variant_id": sem_item["candidate_index"], "parent_variant": sem_id, "response": response}, use, latency

    if args.asr_count > 0:
        asr_results = await asyncio.gather(*(generate_asr(item, f"{_safe(root_id)}__sem{position:02d}") for position, item in enumerate(accepted, 1)), return_exceptions=True)
    else:
        asr_results = [None for _ in accepted]
    asr_rows: list[dict[str, Any]] = []
    for sem_position, (sem_item, result) in enumerate(zip(accepted, asr_results), 1):
        sem_id = f"{_safe(root_id)}__sem{sem_position:02d}"
        sem_prompt = sem_item["prompt"]
        sem_turns = _turn_variant(case, sem_prompt)
        sem_lineage = {
            "case_id": sem_id, "root_case_id": root_id, "parent_variant": root_id,
            "variant_type": "semantic", "semantic_variant_id": sem_position,
            "asr_variant_id": None, "trajectory_hash": root_hash,
            "semantic_signature": contract["semantic_signature"], "trajectory_changed": False,
            "semantic_validation": "PASS", "flash_score": sem_item["score"],
        }
        lineages.append(sem_lineage)
        augmented.append({
            "schema_version": VARIANT_SCHEMA, "case_id": sem_id, "root_case_id": root_id,
            "variant_type": "semantic", "semantic_variant_id": sem_position,
            "prompt": sem_prompt, "turns": sem_turns, "trajectory_hash": root_hash,
            "semantic_signature": contract["semantic_signature"],
        })
        replacements.append({
            "case_id": sem_id, "variant_type": "semantic", "parent_variant": root_id,
            "semantic_variant_id": sem_position, "asr_variant_id": None,
            "prompt": sem_prompt, "turns": sem_turns, "trajectory_hash": root_hash,
            "semantic_signature": contract["semantic_signature"], "validation": {"semantic": "PASS", "asr_recoverability": None},
            "flash_review": sem_item["review"], "trajectory_changed": False,
        })
        if isinstance(result, Exception):
            asr_rows.append({"parent_variant": sem_id, "error": f"{type(result).__name__}: {result}", "candidates": []})
            continue
        if result is None:
            asr_rows.append({"parent_variant": sem_id, "skipped": "asr_disabled", "candidates": []})
            continue
        asr_response, use, latency = result
        asr_usage.append(use)
        raw_asr = asr_response.get("response", {}).get("candidates", [])
        if not isinstance(raw_asr, list):
            raw_asr = []
        asr_record = {"parent_variant": sem_id, "semantic_prompt": sem_prompt, "candidates": []}
        for asr_index, item in enumerate(raw_asr, 1):
            if not isinstance(item, dict):
                continue
            noisy = _candidate_text(item)
            model_recoverable = item.get("recoverable") is not False
            local_ok, local_reasons = validate_asr_candidate(
                case, sem_prompt, noisy, strict_surface=False
            )
            row = {
                "candidate_index": asr_index, "prompt": noisy,
                "noise": item.get("noise") or [], "severity": item.get("severity", "recoverable"),
                "model_recoverable": model_recoverable,
                "local_validation": "PASS" if local_ok else "REJECT", "local_reasons": local_reasons,
            }
            if model_recoverable and local_ok:
                asr_index_final = sum(1 for x in asr_record["candidates"] if x.get("decision") == "accept") + 1
                asr_id = f"{sem_id}__asr{asr_index_final:02d}"
                asr_turns = [noisy] + sem_turns[1:]
                asr_lineage = {
                    "case_id": asr_id, "root_case_id": root_id, "parent_variant": sem_id,
                    "variant_type": "asr", "semantic_variant_id": sem_position,
                    "asr_variant_id": asr_index_final, "trajectory_hash": root_hash,
                    "semantic_signature": contract["semantic_signature"], "trajectory_changed": False,
                    "semantic_validation": "PASS", "asr_recoverability": "PASS",
                    "asr_noise": item.get("noise") or [],
                }
                lineages.append(asr_lineage)
                augmented.append({
                    "schema_version": VARIANT_SCHEMA, "case_id": asr_id, "root_case_id": root_id,
                    "parent_variant": sem_id, "variant_type": "asr",
                    "semantic_variant_id": sem_position, "asr_variant_id": asr_index_final,
                    "prompt": noisy, "turns": asr_turns, "trajectory_hash": root_hash,
                    "semantic_signature": contract["semantic_signature"],
                })
                replacements.append({
                    "case_id": asr_id, "variant_type": "asr", "parent_variant": sem_id,
                    "semantic_variant_id": sem_position, "asr_variant_id": asr_index_final,
                    "prompt": noisy, "turns": asr_turns, "trajectory_hash": root_hash,
                    "semantic_signature": contract["semantic_signature"],
                    "validation": {"semantic": "PASS", "asr_recoverability": "PASS"},
                    "noise": item.get("noise") or [], "severity": item.get("severity", "recoverable"),
                    "trajectory_changed": False,
                })
                row.update({"decision": "accept", "case_id": asr_id})
            else:
                row.update({"decision": "reject", "rejection_reasons": local_reasons + (["FLASH_ASR_NOT_RECOVERABLE"] if not model_recoverable else [])})
            asr_record["candidates"].append(row)
        asr_rows.append(asr_record)

    mapping = {
        "schema_version": "teacher-prompt-replacement-map.v1",
        "root_case_id": root_id, "category": case.get("category"),
        "original": {"prompt": case.get("prompt"), "turns": case.get("turns"), "trajectory_hash": root_hash, "semantic_signature": contract["semantic_signature"], "replacement_allowed": False},
        "replacements": replacements, "rejected_candidates": rejected + dedupe_dropped,
        "replacement_policy": {"trajectory_reused": True, "semantic_contract_must_match": True, "generation_order": "original -> semantic -> asr", "asr_parent_must_be_accepted_semantic": True, "asr_must_be_uniquely_recoverable": True},
        "flash": {"model": args.model, "semantic_latency_seconds": round(semantic_latency, 4), "review_latency_seconds": round(review_latency, 4), "semantic_error": semantic_error, "review_error": review_error},
        "replacement_count": len(replacements), "rejected_count": len(rejected) + len(dedupe_dropped),
    }
    return {
        "root_id": root_id, "contract": contract, "mapping": mapping,
        "semantic_candidates": {"root_case_id": root_id, "candidates": semantic_candidates, "error": semantic_error, "usage": _merge_usage(semantic_usage)},
        "semantic_reviews": {"root_case_id": root_id, "reviews": reviews, "accepted": accepted, "rejected": rejected, "dedupe_dropped": dedupe_dropped, "error": review_error},
        "asr_candidates": asr_rows, "asr_usage": _merge_usage(asr_usage),
        "augmented": augmented, "lineages": lineages,
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY 未定义；不会在无密钥时伪造 Flash 结果")
    if not 5 <= args.semantic_count <= 10:
        raise ValueError("--semantic-count 必须在 5 到 10 之间")
    if args.asr_count != 0 and not 2 <= args.asr_count <= 3:
        raise ValueError("--asr-count 必须为 0（暂不生成）或在 2 到 3 之间")
    prompts = json.loads((root_dir / "prompts.json").read_text(encoding="utf-8"))
    cases = list(prompts.get("cases") or [])
    if args.root_limit > 0:
        ordered = sorted(cases, key=lambda row: _sha(str(row["id"])))
        if args.balanced:
            selected: list[dict[str, Any]] = []
            for category in ("medical", "navigation", "mixed", "general"):
                pool = [row for row in ordered if row.get("category") == category]
                if pool and len(selected) < args.root_limit:
                    selected.append(pool[0])
            selected_ids = {str(row["id"]) for row in selected}
            selected.extend(row for row in ordered if str(row["id"]) not in selected_ids and len(selected) < args.root_limit)
            cases = selected[: args.root_limit]
        else:
            cases = ordered[: args.root_limit]
    source_index = _source_index(root_dir)
    gate = asyncio.Semaphore(args.workers)
    checkpoint_path = output_dir / "checkpoint_results.jsonl"
    checkpoint_path.write_text("", encoding="utf-8")
    checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def process_with_id(case: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str]:
            case_id = str(case["id"])
            try:
                return case_id, await _process_case(case, source_index, root_dir, client, gate, args, key), ""
            except Exception as exc:
                return case_id, None, f"{type(exc).__name__}: {exc}"

        tasks = [asyncio.create_task(process_with_id(case)) for case in cases]
        good: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        completed_count = 0
        for completed_task in asyncio.as_completed(tasks):
            case_id, row, error = await completed_task
            if error:
                failed.append({"root_case_id": case_id, "error": error})
            if isinstance(row, dict):
                good.append(row)
                checkpoint_handle.write(_canonical(row) + "\n")
                checkpoint_handle.flush()
            completed_count += 1
            if completed_count % max(1, args.progress_every) == 0 or completed_count == len(cases):
                print(f"[flash-augmentation] completed {completed_count}/{len(cases)}", flush=True)
    checkpoint_handle.close()
    def write_jsonl(name: str, rows: list[Any]) -> None:
        (output_dir / name).write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")
    write_jsonl("flash_semantic_candidates.jsonl", [row["semantic_candidates"] for row in good])
    write_jsonl("flash_semantic_reviews.jsonl", [row["semantic_reviews"] for row in good])
    write_jsonl("flash_asr_candidates.jsonl", [{"root_case_id": row["root_id"], "rows": row["asr_candidates"], "usage": row["asr_usage"]} for row in good])
    write_jsonl("prompt_mapping.jsonl", [row["mapping"] for row in good])
    write_jsonl("augmented_cases.jsonl", [item for row in good for item in row["augmented"]])
    write_jsonl("lineage.jsonl", [item for row in good for item in row["lineages"]])
    semantic_count = sum(len(row["semantic_reviews"]["accepted"]) for row in good)
    asr_count = sum(sum(1 for item in row["mapping"]["replacements"] if item["variant_type"] == "asr") for row in good)
    metadata = {
        "schema_version": "teacher-flash-augmentation.v1", "root_dir": str(root_dir),
        "root_case_count": len(cases), "processed_case_count": len(good), "failed_case_count": len(failed),
        "root_snapshot_sha256": _tree_hash(root_dir), "model": args.model,
        "generation_order": ["original", "flash_semantic_5_to_10", "flash_review_and_structural_gate", "dedupe"] + (["flash_asr_2_to_3_from_accepted_semantic", "local_recoverability_gate"] if args.asr_count > 0 else ["asr_deferred"]),
        "semantic_accepted": semantic_count, "asr_accepted": asr_count,
        "usage": _merge_usage([row["semantic_candidates"]["usage"] for row in good] + [row["asr_usage"] for row in good]),
        "failed_cases": failed,
        "trajectory_reuse": True, "reasoning_persisted": False,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepSeek Flash 语义/ASR prompt 扩增与审阅")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-limit", type=int, default=2, help="0 表示全量；默认只跑 2 条试点")
    parser.add_argument("--balanced", action="store_true", help="试点时优先选医疗/导航/混合/通用各类")
    parser.add_argument("--semantic-count", type=int, default=5)
    parser.add_argument("--asr-count", type=int, default=0, help="0 表示暂不生成 ASR 近音；启用时为 2 或 3")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high", "max"))
    parser.add_argument("--min-score", type=float, default=0.85)
    parser.add_argument("--local-gate", choices=("structural", "strict"), default="structural", help="Flash 负责语义判断；strict 仅用于对照试验")
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--semantic-max-tokens", type=int, default=3200)
    parser.add_argument("--review-max-tokens", type=int, default=3200)
    parser.add_argument("--asr-max-tokens", type=int, default=2400)
    parser.add_argument("--user-id", default="", help="可选的 DeepSeek prompt-cache user_id；不填则不发送")
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers 必须大于 0")
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
