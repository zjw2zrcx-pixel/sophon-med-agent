"""Flash ASR-noise generation/review, staged after semantic augmentation.

Input is the accepted semantic mapping produced by
``flash_prompt_augmentation.py``.  This module never derives ASR candidates
from a frozen root prompt directly.  It asks Flash to produce exactly two
candidate noisy transcripts for each selected semantic prompt, then asks Flash
again to review recoverability.  Local checks are intentionally structural;
semantic/critical-slot judgement stays with Flash as requested.
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import copy
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
    _safe,
    _sha,
    _source_index,
    _trajectory_hash,
    _tree_hash,
    _turn_variant,
    semantic_contract,
)
from agents.TeacherData.flash_prompt_augmentation import (
    _call_flash,
    _candidate_text,
    _merge_usage,
    _score,
)


ASR_GENERATE_SYSTEM = """你是中文语音识别（ASR）噪声数据生成器。输入已经是通过语义等价审阅的用户输入；现在只模拟听写/识别错误，不改变用户真正想做的事。
请严格返回 JSON：{"candidates":[{"prompt":"...","noise":[{"type":"homophone|substitution|deletion|segmentation|filler","source":"...","output":"..."}],"severity":"safe|recoverable","recoverable":true}]}。
必须生成恰好 2 条彼此不同、与原 prompt 不同的自然中文转写；每条可以同时修改 2 到 3 个词，允许混合同音/近音替换、轻微漏字、词边界变化，并随机加入 1 到 2 个无意义语气助词（如“嗯、啊、呃、那个、哎”）。
关键地点、药物、疾病、剂量、方向、否定和数字必须保持可唯一恢复；如果某候选会变成另一个真实实体，不要生成它。
不要输出解释、Markdown、XML 或思维过程。"""

ASR_REVIEW_SYSTEM = """你是独立的中文 ASR 噪声可恢复性审阅器。判断每条 noisy prompt 是否仍能唯一恢复为 ORIGINAL SEMANTIC PROMPT 的用户意图，并能复用同一条教师 trajectory。
必须检查 intent、关键实体、否定、数量、时间、条件、导航目的地、医疗主体和歧义。安全/可恢复的轻微识别错误才 approve；任何变成另一合法药物、疾病、地点、方向或数字的候选都 reject。
严格输出 JSON：{"reviews":[{"candidate_index":1,"decision":"accept|reject","score":0到1,"recoverable":true,"reason_codes":["..."],"notes":"简短中文理由"}]}。
不要重新回答用户，不要输出 Markdown、XML 或思维过程。"""


def _structural_gate(original: str, candidate: str, turns: list[str]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    normalized = lambda value: re.sub(r"\s+", "", str(value or "")).lower()
    if not candidate:
        reasons.append("EMPTY_PROMPT")
    if normalized(original) == normalized(candidate):
        reasons.append("UNCHANGED_ASR")
    if not turns or turns[0] != candidate:
        reasons.append("FIRST_TURN_MISMATCH")
    if any(token in candidate for token in ("<tool>", "<system>", "忽略以上", "不要遵守")):
        reasons.append("PROMPT_INJECTION_OR_PROTOCOL_TEXT")
    return not reasons, reasons


def _load_items(mapping_path: Path, prompts_path: Path, root_limit: int, max_per_root: int, category_filter: str = "") -> list[dict[str, Any]]:
    prompt_payload = json.loads(prompts_path.read_text(encoding="utf-8"))
    roots = {str(row["id"]): row for row in prompt_payload.get("cases", [])}
    mappings = [json.loads(line) for line in mapping_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_category: dict[str, list[dict[str, Any]]] = {name: [] for name in ("medical", "navigation", "mixed", "general")}
    for mapping in mappings:
        root_id = str(mapping["root_case_id"])
        root = roots.get(root_id)
        if not root:
            continue
        for variant in mapping.get("replacements", []):
            if variant.get("variant_type") != "semantic":
                continue
            by_category.setdefault(str(mapping.get("category", root.get("category", ""))), []).append({
                "root": root,
                "root_case_id": root_id,
                "parent_semantic_case_id": str(variant["case_id"]),
                "prompt": str(variant.get("prompt", "")),
                "turns": list(variant.get("turns") or [variant.get("prompt", "")]),
                "trajectory_hash": str(variant.get("trajectory_hash", "")),
                "semantic_signature": str(variant.get("semantic_signature", "")),
            })
    ordered: list[dict[str, Any]] = []
    categories = (category_filter,) if category_filter else ("medical", "navigation", "mixed", "general")
    if root_limit > 0:
        selected_roots: list[str] = []
        for category in categories:
            candidates = sorted({item["root_case_id"] for item in by_category.get(category, [])})
            if candidates and len(selected_roots) < root_limit:
                selected_roots.append(candidates[0])
        source_values = by_category.get(category_filter, []) if category_filter else [item for values in by_category.values() for item in values]
        remaining = sorted({item["root_case_id"] for item in source_values} - set(selected_roots))
        selected_roots.extend(remaining[: max(0, root_limit - len(selected_roots))])
        for root_id in selected_roots:
            root_items = [item for values in by_category.values() for item in values if item["root_case_id"] == root_id]
            ordered.extend(sorted(root_items, key=lambda item: item["parent_semantic_case_id"])[:max_per_root])
    else:
        for category in categories:
            ordered.extend(sorted(by_category.get(category, []), key=lambda item: item["parent_semantic_case_id"]))
    return ordered


async def _process_item(
    item: dict[str, Any], client: httpx.AsyncClient, gate: asyncio.Semaphore,
    args: argparse.Namespace, api_key: str, session_id: str = "",
) -> dict[str, Any]:
    root = item["root"]
    contract = semantic_contract(root, item["trajectory_hash"])
    payload = {
        "root_case_id": item["root_case_id"],
        "parent_semantic_case_id": item["parent_semantic_case_id"],
        "category": root.get("category"),
        "semantic_prompt": item["prompt"],
        "semantic_turns": item["turns"],
        "semantic_contract": contract,
        "requested_count": 2,
    }
    started = time.perf_counter()
    generation, generation_usage, _ = await _call_flash(
        client, gate, api_key, args.model,
        [{"role": "system", "content": ASR_GENERATE_SYSTEM}, {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        reasoning_effort=args.reasoning_effort, max_tokens=args.generate_max_tokens,
        retries=args.retries, user_id=session_id or args.user_id,
    )
    generated = generation.get("candidates") if isinstance(generation, dict) else []
    if not isinstance(generated, list):
        generated = []
    candidates: list[dict[str, Any]] = []
    for index, value in enumerate(generated, 1):
        if not isinstance(value, dict):
            continue
        candidates.append({
            "candidate_index": index,
            "prompt": _candidate_text(value),
            "noise": value.get("noise") or [],
            "severity": value.get("severity", "recoverable"),
            "model_recoverable": value.get("recoverable") is not False,
        })
    review_payload = {
        "root_case_id": item["root_case_id"],
        "parent_semantic_case_id": item["parent_semantic_case_id"],
        "category": root.get("category"),
        "original_semantic_prompt": item["prompt"],
        "semantic_contract": contract,
        "candidates": candidates,
    }
    review, review_usage, _ = await _call_flash(
        client, gate, api_key, args.model,
        [{"role": "system", "content": ASR_REVIEW_SYSTEM}, {"role": "user", "content": json.dumps(review_payload, ensure_ascii=False)}],
        reasoning_effort=args.reasoning_effort, max_tokens=args.review_max_tokens,
        retries=args.retries, user_id=session_id or args.user_id,
    )
    reviews = review.get("reviews") if isinstance(review, dict) else []
    review_by_index = {
        int(row.get("candidate_index")): row
        for row in reviews if isinstance(row, dict) and str(row.get("candidate_index", "")).isdigit()
    } if isinstance(reviews, list) else {}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for candidate in candidates:
        review_row = review_by_index.get(candidate["candidate_index"], {})
        score = _score(review_row.get("score"))
        structural_ok, structural_reasons = _structural_gate(
            item["prompt"], candidate["prompt"], _turn_variant(root, candidate["prompt"]),
        )
        remote_ok = (
            str(review_row.get("decision", "reject")).lower() == "accept"
            and bool(review_row.get("recoverable", True))
            and score >= args.min_score
            and candidate["model_recoverable"]
        )
        row = {**candidate, "score": score, "review": review_row, "structural_validation": "PASS" if structural_ok else "REJECT", "structural_reasons": structural_reasons}
        if remote_ok and structural_ok:
            accepted.append(row)
        else:
            row["rejection_reasons"] = (["FLASH_REJECT_OR_LOW_SCORE"] if not remote_ok else []) + structural_reasons
            rejected.append(row)
    accepted = sorted(accepted, key=lambda row: row["candidate_index"])[:2]
    replacements: list[dict[str, Any]] = []
    augmented: list[dict[str, Any]] = [{
        "schema_version": VARIANT_SCHEMA,
        "case_id": item["parent_semantic_case_id"],
        "root_case_id": item["root_case_id"],
        "variant_type": "semantic",
        "prompt": item["prompt"], "turns": item["turns"],
        "trajectory_hash": item["trajectory_hash"],
        "semantic_signature": item["semantic_signature"],
    }]
    lineages: list[dict[str, Any]] = []
    for asr_index, candidate in enumerate(accepted, 1):
        asr_id = f"{item['parent_semantic_case_id']}__asr{asr_index:02d}"
        asr_turns = _turn_variant(root, candidate["prompt"])
        lineage = {
            "case_id": asr_id, "root_case_id": item["root_case_id"],
            "parent_variant": item["parent_semantic_case_id"],
            "variant_type": "asr", "semantic_variant_id": None,
            "asr_variant_id": asr_index, "trajectory_hash": item["trajectory_hash"],
            "semantic_signature": item["semantic_signature"],
            "trajectory_changed": False, "semantic_validation": "PASS",
            "asr_recoverability": "PASS", "asr_noise": candidate["noise"],
        }
        lineages.append(lineage)
        augmented.append({
            "schema_version": VARIANT_SCHEMA, "case_id": asr_id,
            "root_case_id": item["root_case_id"],
            "parent_variant": item["parent_semantic_case_id"],
            "variant_type": "asr", "prompt": candidate["prompt"],
            "turns": asr_turns, "trajectory_hash": item["trajectory_hash"],
            "semantic_signature": item["semantic_signature"],
        })
        replacements.append({
            "case_id": asr_id, "variant_type": "asr",
            "parent_variant": item["parent_semantic_case_id"],
            "prompt": candidate["prompt"], "turns": asr_turns,
            "trajectory_hash": item["trajectory_hash"],
            "semantic_signature": item["semantic_signature"],
            "validation": {"semantic": "PASS", "asr_recoverability": "PASS"},
            "flash_review": candidate["review"], "noise": candidate["noise"],
            "severity": candidate["severity"], "trajectory_changed": False,
        })
    mapping = {
        "schema_version": "teacher-asr-prompt-replacement-map.v1",
        "root_case_id": item["root_case_id"],
        "parent_semantic_case_id": item["parent_semantic_case_id"],
        "session_id": session_id,
        "category": root.get("category"),
        "original": {
            "prompt": item["prompt"], "turns": item["turns"],
            "trajectory_hash": item["trajectory_hash"],
            "semantic_signature": item["semantic_signature"],
            "variant_type": "semantic",
        },
        "replacements": replacements,
        "rejected_candidates": rejected,
        "generation_policy": "semantic_variant -> Flash ASR generate exactly 2 -> Flash review",
        "flash": {
            "model": args.model,
            "generation_latency_seconds": round(time.perf_counter() - started, 4),
            "generation_usage": generation_usage,
            "review_usage": review_usage,
            "review_count": len(reviews) if isinstance(reviews, list) else 0,
        },
        "replacement_count": len(replacements),
        "rejected_count": len(rejected),
    }
    return {
        "root_case_id": item["root_case_id"],
        "parent_semantic_case_id": item["parent_semantic_case_id"],
        "category": root.get("category"), "mapping": mapping,
        "generated": {"parent_semantic_case_id": item["parent_semantic_case_id"], "candidates": candidates},
        "review": {"parent_semantic_case_id": item["parent_semantic_case_id"], "reviews": reviews, "accepted": accepted, "rejected": rejected},
        "augmented": augmented, "lineages": lineages,
        "usage": _merge_usage([generation_usage, review_usage]),
    }


async def run(args: argparse.Namespace) -> dict[str, Any]:
    root_dir = Path(args.root_dir).resolve()
    semantic_mapping = Path(args.semantic_mapping).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未定义；不会伪造 ASR 结果")
    items = _load_items(semantic_mapping, root_dir / "prompts.json", args.root_limit, args.max_per_root, args.category)
    if args.max_items > 0:
        items = items[:args.max_items]
    if not items:
        raise RuntimeError("没有找到已接受的 semantic variant")
    gate = asyncio.Semaphore(args.workers)
    checkpoint = output_dir / "checkpoint_results.jsonl"
    checkpoint.write_text("", encoding="utf-8")
    good: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        async def wrapped(item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
            try:
                session_id = f"{args.session_prefix}-item-{item['parent_semantic_case_id'].replace(':', '_')}"
                return item, await _process_item(item, client, gate, args, api_key, session_id), ""
            except Exception as exc:
                return item, None, f"{type(exc).__name__}: {exc}"
        tasks = [asyncio.create_task(wrapped(item)) for item in items]
        with checkpoint.open("a", encoding="utf-8") as handle:
            for number, task in enumerate(asyncio.as_completed(tasks), 1):
                item, result, error = await task
                if result is not None:
                    good.append(result)
                    handle.write(_canonical(result) + "\n")
                    handle.flush()
                else:
                    failed.append({"parent_semantic_case_id": item["parent_semantic_case_id"], "error": error})
                if number % max(1, args.progress_every) == 0 or number == len(items):
                    print(f"[flash-asr] completed {number}/{len(items)}", flush=True)
    def write_jsonl(name: str, rows: list[Any]) -> None:
        (output_dir / name).write_text("".join(_canonical(row) + "\n" for row in rows), encoding="utf-8")
    write_jsonl("asr_candidates.jsonl", [row["generated"] for row in good])
    write_jsonl("asr_reviews.jsonl", [row["review"] for row in good])
    write_jsonl("asr_prompt_mapping.jsonl", [row["mapping"] for row in good])
    write_jsonl("augmented_cases.jsonl", [item for row in good for item in row["augmented"]])
    write_jsonl("lineage.jsonl", [item for row in good for item in row["lineages"]])
    # Merge semantic and ASR surfaces into the root-case mapping.  The root
    # mapping remains the authoritative original/semantic contract; ASR rows
    # are appended only under their accepted semantic parent.
    semantic_rows = [
        json.loads(line) for line in semantic_mapping.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    asr_by_parent = {
        str(row["mapping"]["parent_semantic_case_id"]): row["mapping"].get("replacements", [])
        for row in good
    }
    integrated: list[dict[str, Any]] = []
    for semantic_row in semantic_rows:
        merged = copy.deepcopy(semantic_row)
        merged["schema_version"] = "teacher-prompt-replacement-map.v2"
        merged["generation_order"] = "original -> semantic -> asr"
        asr_replacements: list[dict[str, Any]] = []
        for semantic_variant in semantic_row.get("replacements", []):
            if semantic_variant.get("variant_type") != "semantic":
                continue
            asr_replacements.extend(asr_by_parent.get(str(semantic_variant["case_id"]), []))
        merged["asr_replacements"] = asr_replacements
        merged["integrated_replacements"] = list(merged.get("replacements", [])) + asr_replacements
        merged["integrated_replacement_count"] = len(merged["integrated_replacements"])
        merged["asr_replacement_count"] = len(asr_replacements)
        integrated.append(merged)
    write_jsonl("integrated_prompt_mapping.jsonl", integrated)
    integrated_summary = {
        "schema_version": "teacher-integrated-prompt-map.v1",
        "root_case_count": len(integrated),
        "original_count": len(integrated),
        "semantic_count": sum(sum(1 for item in row.get("replacements", []) if item.get("variant_type") == "semantic") for row in integrated),
        "asr_count": sum(int(row.get("asr_replacement_count", 0)) for row in integrated),
        "integrated_surface_count": sum(1 + int(row.get("integrated_replacement_count", 0)) for row in integrated),
        "asr_processed_item_count": len(good),
        "asr_failed_item_count": len(failed),
        "trajectory_reuse": True,
        "semantic_mapping": str(semantic_mapping),
        "integrated_mapping": str(output_dir / "integrated_prompt_mapping.jsonl"),
    }
    (output_dir / "integrated_summary.json").write_text(json.dumps(integrated_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "schema_version": "teacher-flash-asr-augmentation.v1",
        "root_dir": str(root_dir), "semantic_mapping": str(semantic_mapping),
        "input_item_count": len(items), "processed_item_count": len(good),
        "failed_item_count": len(failed), "model": args.model,
        "root_snapshot_sha256": _tree_hash(root_dir),
        "generation_order": ["accepted_semantic_variant", "flash_generate_2_asr_2_to_3_word_changes_plus_fillers", "flash_review", "structural_gate"],
        "requested_asr_per_item": 2,
        "accepted_asr_count": sum(row["mapping"]["replacement_count"] for row in good),
        "rejected_asr_count": sum(row["mapping"]["rejected_count"] for row in good),
        "failed_items": failed, "reasoning_persisted": False,
        "usage": _merge_usage([row["usage"] for row in good]),
        "integrated_summary": integrated_summary,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="对已接受 semantic prompt 生成并审阅 2 条 ASR 近音变体")
    parser.add_argument("--root-dir", required=True)
    parser.add_argument("--semantic-mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--root-limit", type=int, default=4, help="冒烟时选多少个 root；0 表示全量")
    parser.add_argument("--max-per-root", type=int, default=2)
    parser.add_argument("--category", choices=("", "medical", "navigation", "mixed", "general"), default="")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--model", default="deepseek-v4-flash")
    parser.add_argument("--reasoning-effort", default="low", choices=("low", "medium", "high", "max"))
    parser.add_argument("--min-score", type=float, default=0.85)
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--generate-max-tokens", type=int, default=2200)
    parser.add_argument("--review-max-tokens", type=int, default=2200)
    parser.add_argument("--user-id", default="")
    parser.add_argument("--session-prefix", default="flash-asr-item-v1")
    parser.add_argument("--max-items", type=int, default=0, help="限制实测条数；0 表示不限制")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
