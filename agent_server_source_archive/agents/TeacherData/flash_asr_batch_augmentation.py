"""Batchable, resumable Flash ASR augmentation.

One remote generation call handles five accepted semantic prompts and emits
two candidates for each (ten candidates).  A second remote call reviews the
same ten candidates.  Completed batches are persisted before the next batch,
so interruption/resume never needs to redo completed batches.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
from collections import Counter
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import time
from typing import Any

import httpx

from agents.TeacherData.augment_pipeline import (
    VARIANT_SCHEMA,
    _canonical,
    _tree_hash,
)
from agents.TeacherData.flash_prompt_augmentation import (
    _call_flash,
    _candidate_text,
    _merge_usage,
    _score,
)


ASR_BATCH_GENERATE_SYSTEM = """你是中文 ASR 噪声数据生成器。输入包含 5 条已经通过语义等价审阅的用户输入。
请一次生成 JSON：{"candidates":[{"parent_semantic_case_id":"...","variant_index":1,"prompt":"...","noise":[{"type":"homophone|substitution|deletion|segmentation|filler","source":"...","output":"..."}],"severity":"safe|recoverable","recoverable":true}]}。
每个 parent_semantic_case_id 必须恰好生成 2 条候选，总共 10 条；每条必须修改至少 2 个有意义的汉字/词片（语气助词不计入变化），最多 3 个，可再随机加入 1 到 2 个无意义语气助词（嗯、啊、呃、那个、哎）。禁止只添加语气助词或只做同义改写而没有近音/漏字变化。
关键意图、实体、数字、否定、方向、医疗主体和导航目的地必须仍可唯一恢复；若会变成另一合法实体，不要生成。候选必须与原文不同。
不要输出解释、Markdown、XML 或思维过程。"""

ASR_BATCH_REVIEW_SYSTEM = """你是独立的 ASR 可恢复性审阅器。一次审阅一批 5 条输入及其 10 条候选。
只有在 noisy prompt 仍能唯一恢复原意、可复用同一条教师 trajectory，且至少有 2 个有意义汉字/词片发生可恢复的近音/漏字变化时 approve。只添加语气词、只做同义改写或原文完全不变必须 reject。关键地点、药物、疾病、剂量、数字、否定、方向或意图变成另一合法含义必须 reject。
严格输出 JSON：{"reviews":[{"parent_semantic_case_id":"...","variant_index":1,"decision":"accept|reject","score":0到1,"recoverable":true,"reason_codes":["..."],"notes":"简短中文理由"}]}。
必须覆盖收到的每个候选，不要输出解释、Markdown、XML 或思维过程。"""

ASR_BATCH_REPAIR_GENERATE_SYSTEM = """你是中文 ASR 噪声数据补偿生成器。输入只包含仍不足 2 条合格变体的用户输入。
请严格按每个输入的 missing_count 生成候选，总数等于所有 missing_count 之和；每条必须与原文不同，并让至少两个不同词片各发生一个可恢复的近音/漏字变化，或让同一词片的两个字发生变化（语气助词不计入），最多 3 处，可加入 1 到 2 个无意义语气助词。请在 noise 中列出每一处真实发生的 source/output，并逐字核对 output 确实出现在 prompt 中。禁止只添加语气助词、只做同义改写或复制原文，也不要重复 accepted_variants 或 previous_rejections 中的文本。
关键意图、实体、数字、否定、方向、医疗主体和导航目的地必须仍可唯一恢复；变成另一合法实体就不要生成。
严格输出 JSON：{"candidates":[{"parent_semantic_case_id":"...","variant_index":101,"prompt":"...","noise":[],"severity":"safe|recoverable","recoverable":true}]}。
不要输出解释、Markdown、XML 或思维过程。"""

ASR_BATCH_REPAIR_REVIEW_SYSTEM = """你是独立的 ASR 可恢复性审阅器，审阅补偿候选。只有仍能唯一恢复原意、可复用同一条教师 trajectory、且至少有两个不同词片各发生一个可恢复近音/漏字变化，或同一词片两个字发生变化时才 approve。
只添加语气词、同义改写、原文不变、关键实体变成另一合法含义必须 reject。
严格输出 JSON：{"reviews":[{"parent_semantic_case_id":"...","variant_index":101,"decision":"accept|reject","score":0到1,"recoverable":true,"reason_codes":[],"notes":"简短中文理由"}]}。
必须覆盖收到的每个候选，不要输出解释、Markdown、XML 或思维过程。"""


def _structural_gate(original: str, candidate: str) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    normalize = lambda value: re.sub(r"\s+", "", str(value or "")).lower()
    if not candidate:
        reasons.append("EMPTY_PROMPT")
    if normalize(original) == normalize(candidate):
        reasons.append("UNCHANGED_ASR")
    filler_words = {"嗯", "啊", "呃", "那个", "哎", "呀", "呢", "吧", "诶"}
    def without_fillers(value: str) -> str:
        return "".join(char for char in normalize(value) if char not in filler_words)

    original_surface = without_fillers(original)
    candidate_surface = without_fillers(candidate)
    # Lightweight surface gate only: reject exact/speech-filler-only rows;
    # semantic equivalence and critical-slot judgement remain Flash's job.
    if normalize(original) != normalize(candidate):
        # Chinese has no whitespace word boundaries.  Count changed surface
        # units after removing filler particles; this intentionally stays a
        # lightweight gate, while Flash remains responsible for semantics.
        changed_units = 0
        for tag, i1, i2, j1, j2 in SequenceMatcher(None, original_surface, candidate_surface).get_opcodes():
            if tag != "equal":
                changed_units += max(i2 - i1, j2 - j1)
        if changed_units < 2:
            reasons.append("LESS_THAN_TWO_MEANINGFUL_CHANGES")
    if any(token in candidate for token in ("<tool>", "<system>", "忽略以上", "不要遵守")):
        reasons.append("PROMPT_INJECTION_OR_PROTOCOL_TEXT")
    return not reasons, reasons


def _load_items(mapping_path: Path, prompts_path: Path) -> list[dict[str, Any]]:
    prompts = json.loads(prompts_path.read_text(encoding="utf-8"))
    roots = {str(row["id"]): row for row in prompts.get("cases", [])}
    items: list[dict[str, Any]] = []
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        mapping = json.loads(line)
        root_id = str(mapping["root_case_id"])
        root = roots.get(root_id)
        if not root:
            continue
        for variant in mapping.get("replacements", []):
            if variant.get("variant_type") != "semantic":
                continue
            items.append({
                "root": root,
                "root_case_id": root_id,
                "parent_semantic_case_id": str(variant["case_id"]),
                "category": mapping.get("category", root.get("category")),
                "prompt": str(variant.get("prompt", "")),
                "turns": list(variant.get("turns") or [variant.get("prompt", "")]),
                "trajectory_hash": str(variant.get("trajectory_hash", "")),
                "semantic_signature": str(variant.get("semantic_signature", "")),
            })
    return sorted(items, key=lambda row: row["parent_semantic_case_id"])


def _batch_payload(items: list[dict[str, Any]]) -> dict[str, Any]:
    cases = []
    for item in items:
        root = item["root"]
        source = root.get("medical_source") if isinstance(root.get("medical_source"), dict) else {}
        expected = root.get("expected") if isinstance(root.get("expected"), dict) else {}
        cases.append({
            "parent_semantic_case_id": item["parent_semantic_case_id"],
            "root_case_id": item["root_case_id"], "category": item["category"],
            "semantic_prompt": item["prompt"], "semantic_turns": item["turns"],
            "critical_entities": {
                "navigation_target": expected.get("navigation_target"),
                "medical_question": source.get("question"),
                "required_tools": expected.get("required_tools", []),
            },
        })
    return {"batch_size": len(cases), "requested_candidates_per_case": 2, "cases": cases}


def _build_batch_result(
    items: list[dict[str, Any]], generated: list[dict[str, Any]], reviews: list[dict[str, Any]],
    generation_usage: dict[str, int], review_usage: dict[str, int], batch_index: int,
    session_id: str = "",
) -> dict[str, Any]:
    generated_by_parent: dict[str, list[dict[str, Any]]] = {}
    for row in generated:
        parent = str(row.get("parent_semantic_case_id", ""))
        if parent in {item["parent_semantic_case_id"] for item in items}:
            generated_by_parent.setdefault(parent, []).append(row)
    review_by_key = {
        (str(row.get("parent_semantic_case_id", "")), int(row.get("variant_index", 0))): row
        for row in reviews if isinstance(row, dict) and str(row.get("variant_index", "")).isdigit()
    }
    results: list[dict[str, Any]] = []
    for item in items:
        parent = item["parent_semantic_case_id"]
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        for candidate in generated_by_parent.get(parent, []):
            index = int(candidate.get("variant_index", 0))
            review = review_by_key.get((parent, index), {})
            prompt = _candidate_text(candidate)
            structural_ok, structural_reasons = _structural_gate(item["prompt"], prompt)
            remote_ok = (
                str(review.get("decision", "reject")).lower() == "accept"
                and bool(review.get("recoverable", True))
                and _score(review.get("score")) >= 0.85
                and candidate.get("recoverable") is not False
            )
            row = {
                "candidate_index": index, "prompt": prompt,
                "noise": candidate.get("noise") or [], "severity": candidate.get("severity", "recoverable"),
                "score": _score(review.get("score")), "review": review,
                "structural_validation": "PASS" if structural_ok else "REJECT",
                "structural_reasons": structural_reasons,
            }
            if remote_ok and structural_ok and len(accepted) < 2:
                accepted.append(row)
            else:
                row["rejection_reasons"] = (["FLASH_REJECT_OR_LOW_SCORE"] if not remote_ok else []) + structural_reasons
                rejected.append(row)
        replacements: list[dict[str, Any]] = []
        augmented = [{
            "schema_version": VARIANT_SCHEMA,
            "case_id": parent, "root_case_id": item["root_case_id"],
            "variant_type": "semantic", "prompt": item["prompt"], "turns": item["turns"],
            "trajectory_hash": item["trajectory_hash"], "semantic_signature": item["semantic_signature"],
        }]
        lineages: list[dict[str, Any]] = []
        for asr_index, candidate in enumerate(accepted, 1):
            asr_id = f"{parent}__asr{asr_index:02d}"
            turns = [candidate["prompt"]] + item["turns"][1:]
            lineage = {
                "case_id": asr_id, "root_case_id": item["root_case_id"],
                "parent_variant": parent, "variant_type": "asr",
                "semantic_variant_id": None, "asr_variant_id": asr_index,
                "trajectory_hash": item["trajectory_hash"], "semantic_signature": item["semantic_signature"],
                "trajectory_changed": False, "semantic_validation": "PASS",
                "asr_recoverability": "PASS", "asr_noise": candidate["noise"],
                "batch_index": batch_index,
            }
            lineages.append(lineage)
            augmented.append({
                "schema_version": VARIANT_SCHEMA, "case_id": asr_id,
                "root_case_id": item["root_case_id"], "parent_variant": parent,
                "variant_type": "asr", "prompt": candidate["prompt"], "turns": turns,
                "trajectory_hash": item["trajectory_hash"], "semantic_signature": item["semantic_signature"],
            })
            replacements.append({
                "case_id": asr_id, "variant_type": "asr", "parent_variant": parent,
                "prompt": candidate["prompt"], "turns": turns,
                "trajectory_hash": item["trajectory_hash"], "semantic_signature": item["semantic_signature"],
                "validation": {"semantic": "PASS", "asr_recoverability": "PASS"},
                "flash_review": candidate["review"], "noise": candidate["noise"],
                "severity": candidate["severity"], "trajectory_changed": False,
            })
        results.append({
            "mapping": {
                "schema_version": "teacher-asr-prompt-replacement-map.v1",
                "root_case_id": item["root_case_id"], "parent_semantic_case_id": parent,
                "category": item["category"],
                "original": {"prompt": item["prompt"], "turns": item["turns"], "trajectory_hash": item["trajectory_hash"], "semantic_signature": item["semantic_signature"], "variant_type": "semantic"},
                "replacements": replacements, "rejected_candidates": rejected,
                "batch_index": batch_index, "generation_policy": "5 semantic -> 10 ASR -> batch review",
                "session_id": session_id,
                "replacement_count": len(replacements), "rejected_count": len(rejected),
            },
            "augmented": augmented, "lineages": lineages,
        })
    return {"batch_index": batch_index, "session_id": session_id, "results": results, "generation_usage": generation_usage, "review_usage": review_usage}


async def _process_batch(
    batch_index: int, items: list[dict[str, Any]], client: httpx.AsyncClient,
    gate: asyncio.Semaphore, args: argparse.Namespace, key: str, session_id: str,
) -> dict[str, Any]:
    all_generated: list[dict[str, Any]] = []
    all_reviews: list[dict[str, Any]] = []
    generation_usages: list[dict[str, int]] = []
    review_usages: list[dict[str, int]] = []
    missing_items = list(items)
    accepted_by_parent: dict[str, int] = {item["parent_semantic_case_id"]: 0 for item in items}
    for round_index in range(args.compensation_rounds + 1):
        if not missing_items:
            break
        payload = _batch_payload(missing_items)
        if round_index:
            payload["repair_round"] = round_index
            payload["missing_count_by_parent"] = {
                item["parent_semantic_case_id"]: max(
                    1, args.min_asr_per_semantic - accepted_by_parent.get(item["parent_semantic_case_id"], 0)
                )
                for item in missing_items
            }
            payload["accepted_variants"] = {
                item["parent_semantic_case_id"]: [
                    row.get("prompt", "") for row in all_generated
                    if str(row.get("parent_semantic_case_id")) == item["parent_semantic_case_id"]
                    and any(
                        str(review.get("parent_semantic_case_id")) == item["parent_semantic_case_id"]
                        and int(review.get("variant_index", -1)) == int(row.get("variant_index", -2))
                        and str(review.get("decision", "reject")).lower() == "accept"
                        and bool(review.get("recoverable", True))
                        and _score(review.get("score")) >= 0.85
                        for review in all_reviews
                    )
                ] for item in missing_items
            }
            payload["previous_rejections"] = {
                item["parent_semantic_case_id"]: [
                    row.get("prompt", "") for row in all_generated
                    if str(row.get("parent_semantic_case_id")) == item["parent_semantic_case_id"]
                    and row.get("prompt") not in payload["accepted_variants"].get(item["parent_semantic_case_id"], [])
                ] for item in missing_items
            }
        generate_system = ASR_BATCH_GENERATE_SYSTEM if round_index == 0 else ASR_BATCH_REPAIR_GENERATE_SYSTEM
        review_system = ASR_BATCH_REVIEW_SYSTEM if round_index == 0 else ASR_BATCH_REPAIR_REVIEW_SYSTEM
        try:
            generated_response, generation_usage, _ = await _call_flash(
                client, gate, key, args.model,
                [{"role": "system", "content": generate_system}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                reasoning_effort=args.reasoning_effort, max_tokens=args.generate_max_tokens,
                retries=args.retries, user_id=session_id,
            )
            generated = generated_response.get("candidates") if isinstance(generated_response, dict) else []
            if not isinstance(generated, list):
                raise ValueError("batch generation candidates is not a list")
            # Candidate indices are remapped locally so repeated repair rounds
            # cannot overwrite an earlier review with the same (parent,index).
            used_indices = {
                (str(row.get("parent_semantic_case_id")), int(row.get("variant_index", -1)))
                for row in all_generated
                if str(row.get("variant_index", "")).lstrip("-").isdigit()
            }
            next_index: dict[str, int] = Counter()
            for row in generated:
                if not isinstance(row, dict):
                    continue
                parent = str(row.get("parent_semantic_case_id", ""))
                current = int(row.get("variant_index", 0) or 0)
                while (parent, current) in used_indices or current <= 0:
                    next_index[parent] += 1
                    current = round_index * 100 + next_index[parent]
                row["variant_index"] = current
                used_indices.add((parent, current))
            review_payload = {**payload, "generated_candidates": generated}
            reviewed_response, review_usage, _ = await _call_flash(
                client, gate, key, args.model,
                [{"role": "system", "content": review_system}, {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)}],
                reasoning_effort=args.reasoning_effort, max_tokens=args.review_max_tokens,
                retries=args.retries, user_id=session_id,
            )
            reviews = reviewed_response.get("reviews") if isinstance(reviewed_response, dict) else []
            if not isinstance(reviews, list):
                raise ValueError("batch review reviews is not a list")
            all_generated.extend(generated)
            all_reviews.extend(reviews)
            generation_usages.append(generation_usage)
            review_usages.append(review_usage)
        except Exception as exc:
            if round_index >= args.compensation_rounds:
                raise
            # A failed repair round is retried as the next round; no partial
            # candidate is considered accepted without a paired review.
            print(f"[flash-asr-batch] batch {batch_index} compensation round {round_index} failed: {type(exc).__name__}: {exc}", flush=True)
            continue
        interim = _build_batch_result(
            items, all_generated, all_reviews, _merge_usage(generation_usages),
            _merge_usage(review_usages), batch_index, session_id,
        )
        accepted_by_parent = {
            row["mapping"]["parent_semantic_case_id"]: row["mapping"].get("replacement_count", 0)
            for row in interim["results"]
        }
        missing_items = [
            item for item in items
            if accepted_by_parent.get(item["parent_semantic_case_id"], 0) < args.min_asr_per_semantic
        ]
    result = _build_batch_result(
        items, all_generated, all_reviews, _merge_usage(generation_usages),
        _merge_usage(review_usages), batch_index, session_id,
    )
    for row in result["results"]:
        row["mapping"]["target_replacement_count"] = args.min_asr_per_semantic
        row["mapping"]["replacement_shortfall"] = max(
            0, args.min_asr_per_semantic - row["mapping"].get("replacement_count", 0)
        )
    return result


def _write_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    root_dir = Path(args.root_dir).resolve()
    semantic_mapping = Path(args.semantic_mapping).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY 未定义")
    items = _load_items(semantic_mapping, root_dir / "prompts.json")
    batches = [items[index:index + 5] for index in range(0, len(items), 5)]
    if args.max_batches > 0:
        batches = batches[:args.max_batches]
    completed: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, list[dict[str, Any]]]] = []
    for index, batch in enumerate(batches):
        checkpoint = checkpoint_dir / f"batch_{index:05d}.json"
        if checkpoint.is_file():
            try:
                payload = json.loads(checkpoint.read_text(encoding="utf-8"))
                result = payload.get("result") or {}
                result_rows = result.get("results") or []
                complete = all(
                    row.get("mapping", {}).get("replacement_count", 0) >= args.min_asr_per_semantic
                    for row in result_rows
                ) and len(result_rows) == len(batch)
                checkpoint_usable = complete or args.accept_shortfall_checkpoints
                if payload.get("batch_index") == index and payload.get("item_ids") == [item["parent_semantic_case_id"] for item in batch] and checkpoint_usable:
                    completed[index] = payload["result"]
                    continue
            except (OSError, ValueError, KeyError, TypeError):
                pass
        pending.append((index, batch))
    gate = asyncio.Semaphore(args.workers)
    def batch_session_id(index: int) -> str:
        return f"{args.session_prefix}-batch-{index:05d}"
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def wrapped(index: int, batch: list[dict[str, Any]]) -> tuple[int, dict[str, Any] | None, str]:
            try:
                return index, await _process_batch(index, batch, client, gate, args, key, batch_session_id(index)), ""
            except Exception as exc:
                return index, None, f"{type(exc).__name__}: {exc}"
        tasks = [asyncio.create_task(wrapped(index, batch)) for index, batch in pending]
        for number, task in enumerate(asyncio.as_completed(tasks), 1):
            index, result, error = await task
            if result is not None:
                completed[index] = result
                checkpoint = checkpoint_dir / f"batch_{index:05d}.json"
                _write_atomic(checkpoint, {
                    "schema_version": "teacher-asr-batch-checkpoint.v1",
                    "batch_index": index,
                    "session_id": result.get("session_id", batch_session_id(index)),
                    "item_ids": [item["parent_semantic_case_id"] for item in batches[index]],
                    "result": result,
                })
            else:
                print(f"[flash-asr-batch] batch {index} failed: {error}", flush=True)
            if number % max(1, args.progress_every) == 0 or number == len(tasks):
                print(f"[flash-asr-batch] pending completed {number}/{len(tasks)}; total checkpointed {len(completed)}/{len(batches)}", flush=True)
    good_results = [completed[index] for index in sorted(completed)]
    def write_jsonl(name: str, rows: list[Any]) -> None:
        (output_dir / name).write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")
    mapping_rows = [row["mapping"] for batch in good_results for row in batch["results"]]
    write_jsonl("asr_prompt_mapping.jsonl", mapping_rows)
    write_jsonl("augmented_cases.jsonl", [item for batch in good_results for row in batch["results"] for item in row["augmented"]])
    write_jsonl("lineage.jsonl", [item for batch in good_results for row in batch["results"] for item in row["lineages"]])
    semantic_rows = [json.loads(line) for line in semantic_mapping.read_text(encoding="utf-8").splitlines() if line.strip()]
    asr_by_parent = {str(row["parent_semantic_case_id"]): row.get("replacements", []) for row in mapping_rows}
    integrated: list[dict[str, Any]] = []
    for semantic_row in semantic_rows:
        merged = copy.deepcopy(semantic_row)
        asr_replacements: list[dict[str, Any]] = []
        for semantic_variant in semantic_row.get("replacements", []):
            if semantic_variant.get("variant_type") == "semantic":
                asr_replacements.extend(asr_by_parent.get(str(semantic_variant["case_id"]), []))
        merged["schema_version"] = "teacher-prompt-replacement-map.v2"
        merged["generation_order"] = "original -> semantic -> asr"
        merged["asr_replacements"] = asr_replacements
        merged["integrated_replacements"] = list(merged.get("replacements", [])) + asr_replacements
        merged["semantic_replacement_count"] = sum(1 for item in merged.get("replacements", []) if item.get("variant_type") == "semantic")
        merged["asr_replacement_count"] = len(asr_replacements)
        merged["integrated_replacement_count"] = len(merged["integrated_replacements"])
        integrated.append(merged)
    write_jsonl("integrated_prompt_mapping.jsonl", integrated)
    summary = {
        "schema_version": "teacher-asr-batch-augmentation.v1",
        "root_dir": str(root_dir), "semantic_mapping": str(semantic_mapping),
        "batch_size": 5, "requested_asr_per_semantic": 2,
        "input_item_count": len(items), "total_batch_count": len(batches),
        "completed_batch_count": len(completed), "pending_batch_count": len(batches) - len(completed),
        "processed_semantic_count": len(mapping_rows),
        "accepted_asr_count": sum(row.get("replacement_count", 0) for row in mapping_rows),
        "rejected_asr_count": sum(row.get("rejected_count", 0) for row in mapping_rows),
        "target_asr_per_semantic": args.min_asr_per_semantic,
        "shortfall_semantic_count": sum(
            1 for row in mapping_rows if row.get("replacement_count", 0) < args.min_asr_per_semantic
        ),
        "shortfall_asr_count": sum(
            max(0, args.min_asr_per_semantic - row.get("replacement_count", 0)) for row in mapping_rows
        ),
        "integrated_root_count": len(integrated),
        "integrated_surface_count": sum(1 + row.get("integrated_replacement_count", 0) for row in integrated),
        "root_snapshot_sha256": _tree_hash(root_dir),
        "trajectory_reuse": True, "reasoning_persisted": False,
        "checkpoint_dir": str(checkpoint_dir),
        "session_prefix": args.session_prefix,
    }
    (output_dir / "metadata.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "integrated_summary.json").write_text(json.dumps({
        "schema_version": "teacher-integrated-prompt-map.v1",
        "root_case_count": len(integrated),
        "original_count": len(integrated),
        "semantic_count": sum(row.get("semantic_replacement_count", 0) for row in integrated),
        "asr_count": sum(row.get("asr_replacement_count", 0) for row in integrated),
        "integrated_surface_count": summary["integrated_surface_count"],
        "mapping": str(output_dir / "integrated_prompt_mapping.jsonl"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="每批 5 条 semantic，一次生成 10 条 ASR 并支持 checkpoint resume")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--semantic-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-batches", type=int, default=0, help="冒烟使用；0 表示全量")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high", "max"))
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--generate-max-tokens", type=int, default=2600)
    parser.add_argument("--review-max-tokens", type=int, default=2400)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--session-prefix", default="flash-asr-v4")
    parser.add_argument("--min-asr-per-semantic", type=int, default=2)
    parser.add_argument("--compensation-rounds", type=int, default=6)
    parser.add_argument(
        "--accept-shortfall-checkpoints", action="store_true",
        help="第一阶段允许复用少于目标数量的 checkpoint；后续补偿阶段不应开启",
    )
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
