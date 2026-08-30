"""Select a balanced, deterministic independent-intent prompt bank."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


DEFAULT_RATIOS = {"medical": 0.5, "mixed": 0.2, "navigation": 0.2, "general": 0.1}


def _shuffle(rows: list[dict[str, Any]], seed: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{seed}:{row.get('id')}:{row.get('prompt')}".encode("utf-8")
    ).hexdigest())


def _counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    raw = {key: total * value for key, value in ratios.items()}
    result = {key: int(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda value: raw[value] - result[value], reverse=True):
        if sum(result.values()) >= total:
            break
        result[key] += 1
    return result


def select(
    cases: list[dict[str, Any]], *, total: int, multi_ratio: float,
    category_ratios: dict[str, float], seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if total < 1 or not 0 <= multi_ratio <= 1:
        raise ValueError("total 必须大于 0，multi_ratio 必须在 0 和 1 之间")
    expected_categories = _counts(total, category_ratios)
    expected_multi = round(total * multi_ratio)
    grouped = {
        "multi_turn": [row for row in cases if len(row.get("turns") or []) > 1],
        "single_turn": [row for row in cases if len(row.get("turns") or []) <= 1],
    }
    target_by_mode = {"multi_turn": expected_multi, "single_turn": total - expected_multi}
    selected: list[dict[str, Any]] = []
    remaining = dict(expected_categories)
    for mode in ("multi_turn", "single_turn"):
        pool = {category: _shuffle(
            [row for row in grouped[mode] if row.get("category") == category],
            f"{seed}:{mode}:{category}",
        ) for category in expected_categories}
        mode_target = target_by_mode[mode]
        # The desired category allocation is applied to each mode only as a
        # proportional target; remaining slots are filled from available
        # categories deterministically.
        mode_counts = _counts(mode_target, expected_categories)
        for category in expected_categories:
            take = min(mode_counts[category], len(pool[category]), remaining[category])
            selected.extend(pool[category][:take])
            remaining[category] -= take
        current_mode_count = sum(
            1 for row in selected
            if (len(row.get("turns") or []) > 1) == (mode == "multi_turn")
        )
        need = mode_target - current_mode_count
        if need > 0:
            candidates = [
                row for category in expected_categories for row in pool[category]
                if row not in selected and remaining[category] > 0
            ]
            for row in candidates[:need]:
                selected.append(row)
                remaining[str(row.get("category"))] -= 1
    if len(selected) < total:
        # A malformed upstream filter must not silently claim a 1K bank.
        raise ValueError(f"可选 case 不足: requested={total}, selected={len(selected)}")
    selected = _shuffle(selected, seed)[:total]
    if len({str(row.get("id")) for row in selected}) != total:
        raise ValueError("选择结果 case id 不唯一")
    stats = {
        "requested_total": total,
        "selected_total": len(selected),
        "requested_multi_ratio": multi_ratio,
        "category_counts": dict(Counter(str(row.get("category")) for row in selected)),
        "dialogue_counts": dict(Counter(
            "multi_turn" if len(row.get("turns") or []) > 1 else "single_turn"
            for row in selected
        )),
        "seed": seed,
    }
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description="从独立意图候选中确定性选择平衡的约千条 bank")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--total", type=int, default=1000)
    parser.add_argument("--multi-ratio", type=float, default=0.5)
    parser.add_argument("--category-ratios", default="medical=0.5,mixed=0.2,navigation=0.2,general=0.1")
    parser.add_argument("--seed", default="independent-bank-v7")
    args = parser.parse_args()
    payload = json.loads(Path(args.prompts_file).read_text(encoding="utf-8"))
    cases = list(payload.get("cases") or [])
    ratios = {}
    for item in args.category_ratios.split(","):
        name, separator, value = item.partition("=")
        if not separator:
            raise ValueError("category-ratios 格式错误")
        ratios[name.strip()] = float(value)
    selected, stats = select(
        cases, total=args.total, multi_ratio=args.multi_ratio,
        category_ratios=ratios, seed=args.seed,
    )
    generation = dict(payload.get("generation") or {})
    generation["independent_selection"] = stats
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": "teacher-prompts.v1",
        "generation": generation,
        "cases": selected,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
