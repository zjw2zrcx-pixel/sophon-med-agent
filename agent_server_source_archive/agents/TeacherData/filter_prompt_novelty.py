"""Remove cross-version prompt/source overlap before executing Teacher runs."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re


def _normalize(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", text.lower())


def _grams(text: str) -> set[str]:
    value = _normalize(text)
    return {value[i:i + 2] for i in range(max(0, len(value) - 1))}


def _similar(left: str, right: str) -> float:
    a, b = _grams(left), _grams(right)
    return len(a & b) / len(a | b) if a and b else float(_normalize(left) == _normalize(right))


def main() -> None:
    parser = argparse.ArgumentParser(description="剔除与历史固定 bank 重复或高度相似的提示词")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--prior-split-dir", action="append", default=[])
    parser.add_argument(
        "--prior-prompts-file", action="append", default=[],
        help="历史完整 prompts.json；无需拆成 train/benchmark",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--threshold", type=float, default=0.82)
    parser.add_argument(
        "--medical-threshold", type=float, default=None,
        help="medical/mixed 的文本相似度阈值；留空沿用 threshold。医疗题通常共享句式，"
             "建议用 0.95，并仍以 answer_sha256/source 去重。",
    )
    args = parser.parse_args()
    source_path = Path(args.prompts_file).resolve()
    source = json.loads(source_path.read_text("utf-8"))
    prior = []
    prior_medical_hashes = set()
    for root_value in args.prior_split_dir:
        root = Path(root_value).resolve()
        for split in ("train", "benchmark"):
            for case in json.loads((root / split / "prompts.json").read_text("utf-8"))["cases"]:
                prior.append(case)
                answer_hash = str((case.get("medical_source") or {}).get("answer_sha256", ""))
                if answer_hash:
                    prior_medical_hashes.add(answer_hash)
    for file_value in args.prior_prompts_file:
        payload = json.loads(Path(file_value).resolve().read_text("utf-8"))
        for case in payload.get("cases", []):
            prior.append(case)
            answer_hash = str((case.get("medical_source") or {}).get("answer_sha256", ""))
            if answer_hash:
                prior_medical_hashes.add(answer_hash)
    kept, rejected = [], []
    seen_new_medical_hashes = set()
    for case in source.get("cases", []):
        answer_hash = str((case.get("medical_source") or {}).get("answer_sha256", ""))
        reason = ""
        matched = ""
        if answer_hash and answer_hash in prior_medical_hashes:
            reason = "PRIOR_MEDICAL_SOURCE"
        elif answer_hash and answer_hash in seen_new_medical_hashes:
            reason = "DUPLICATE_NEW_MEDICAL_SOURCE"
        else:
            # Masked medical first turns intentionally share a short template
            # (for example “还没说具体名称”); provenance hash is the stronger
            # independent-intent key.  A non-positive medical threshold means
            # do not reject such cases by wording similarity.
            skip_medical_text = (
                case.get("category") in {"medical", "mixed"}
                and answer_hash
                and args.medical_threshold is not None
                and args.medical_threshold <= 0
            )
            if not skip_medical_text:
                for old in prior:
                    if old.get("category") != case.get("category"):
                        continue
                    score = _similar(str(case.get("prompt", "")), str(old.get("prompt", "")))
                    threshold = (
                        args.medical_threshold
                        if args.medical_threshold is not None
                        and case.get("category") in {"medical", "mixed"}
                        else args.threshold
                    )
                    if score >= threshold:
                        reason, matched = f"PRIOR_PROMPT_SIMILARITY:{score:.3f}", str(old.get("id"))
                        break
        if reason:
            rejected.append({"id": case.get("id"), "reason": reason, "matched": matched})
        else:
            kept.append(copy.deepcopy(case))
            if answer_hash:
                seen_new_medical_hashes.add(answer_hash)
    generation = dict(source.get("generation") or {})
    generation["novelty_filter"] = {
        "source": str(source_path), "prior_split_dirs": args.prior_split_dir,
        "prior_prompts_files": args.prior_prompts_file,
        "threshold": args.threshold, "input": len(source.get("cases", [])),
        "medical_threshold": args.medical_threshold,
        "kept": len(kept), "rejected": len(rejected),
        "reasons": dict(Counter(row["reason"].split(":")[0] for row in rejected)),
        "rejected_cases": rejected,
    }
    Path(args.output).write_text(json.dumps({
        "schema_version": source.get("schema_version", "teacher-prompts.v1"),
        "generation": generation, "cases": kept,
    }, ensure_ascii=False, indent=2), "utf-8")


if __name__ == "__main__":
    main()
