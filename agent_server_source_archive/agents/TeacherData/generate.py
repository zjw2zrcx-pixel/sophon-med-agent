"""Generate prompts with DeepSeek V4 Pro and trajectories with V4 Flash."""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import httpx

from agents.API.session import Session
from agents.agent import Agent, AgentConfig
from agents.CallRoute.router import CallRouter
from agents.Modes.benchmark import BenchmarkMode
from agents.MCP.tools.navigate import HOSPITAL_LOCATIONS, LOCATION_ALIASES
from agents.Skill.manager import SkillManager
from agents.TeacherData.medical_prompts import MedicalPromptSampler
from agents.TeacherData.audit import audit_runs
from agents.API.trajectory import summarize_token_usage


PROMPT_SYSTEM = """你是医院机器人Agent训练集设计专家。请生成中文用户请求，覆盖医疗问答、医院导航、通用问答、医疗+导航混合任务。请求要像真实口语，包含省略、否定、指代、ASR同音误差、隐含意图或多约束，但不能故意制造无法回答的谜语。

当前工具规则：
1. 医疗问题必须调用 medical_consult，参数 query 必须保留用户完整原话；工具会返回真实本地混合检索结果，不能预设它一定能给出诊断。
2. navigation_target 必须逐字使用已注册名称：医院大门、门诊大厅、挂号处、收费处、服务台、急诊科、药房、检验科、影像科、超声科、输液室、住院部、体检中心、卫生间、呼吸内科、心血管内科、消化内科、神经内科、内分泌科、肾内科、血液科、风湿免疫科、普通外科、骨科、泌尿外科、神经外科、胸外科、妇科、产科、儿科、眼科、耳鼻喉科、口腔科、皮肤科、精神心理科、康复医学科、中医科。禁止使用泛称“内科”“外科”。
3. 通用问答必须能由稳定常识直接回答，或明确使用 get_time/get_system_stats。禁止生成需要联网或医院私有实时数据的问题，例如天气、新闻、附近商家、停车费、开放时间、WiFi密码、院内设施、窗口排队、实时库存和排班。
4. 混合请求需要区分“询问去哪一科”和“明确要求机器人带路”。只有后者需要 navigate。
5. 危险症状应优先医疗检索并建议急诊；如果用户明确要求带路，可导航到急诊科。

只输出 JSON object，格式为：
{"cases":[{"id":"case-01","category":"medical|navigation|general|mixed","prompt":"第一轮用户输入","turns":["第一轮用户输入","可选的第二轮回答","可选的第三轮回答"],"difficulty":"hard","expected":{"required_tools":["..."],"forbidden_tools":["..."],"navigation_target":"可选","notes":"期望行为，不写具体诊断"},"risk_tags":["..."]}]}
医疗文本会在生成后由 Harness 使用本地 train 数据库替换；你只需生成类别、导航目标和期望工具结构，不得为医疗 case 提供具体诊断作为 expected。绝大多数样本应为单轮，少量医疗样本可留作多轮结构。
不要输出思维过程。"""

# Keep the beginning of every Pro prompt byte/token stable.  DeepSeek context
# caching is exact-prefix based; putting batch number/category distribution at
# the beginning would invalidate the rest of this otherwise reusable frame.
PROMPT_BATCH_FRAME = """请严格按照上面的 JSON 契约生成结构化候选。先检查每条 case 的类别、工具标签和目的地合法性，再输出结果。不要输出解释、Markdown 或思维过程；只输出一个 JSON object，且 cases 数量必须与本批要求完全一致。医疗 case 只写用户表达和结构标签，具体医疗事实由本地数据库后处理。"""
PROMPT_CACHE_LANE_PREFIX = "teacher-prompt-lane"

PROMPT_CONTRACT_VERSION = "teacher-prompt-contract.v5-flash"
GENERAL_FOCUS = (
    "当前时间（required_tools=get_time）",
    "机器人CPU/内存/磁盘状态（required_tools=get_system_stats）",
    "一步或两步口算（不调用业务工具）",
    "常见词语解释或近义词辨析（不调用业务工具）",
    "简单中英翻译（不调用业务工具）",
    "长度、重量或温度单位换算（不调用业务工具）",
    "稳定的日常科学常识（不调用业务工具）",
    "日期与星期表达；只有询问当前日期时使用get_time",
)
INDEPENDENT_INTENT_FOCUS = (
    "少见但合法的院内科室或服务场景，避免‘挂号处怎么走’等基准短句",
    "带有顺序、否定或限制条件的单一任务，但不要拼接第二个工具意图",
    "用户不完整但可由工具 schema 判断的真实口语请求，避免套用旧句式",
    "医院不同流程节点之间的独立目标，明确区分咨询、查询和带路",
    "通用问答中的单位、词义、时间表达和稳定常识，避免重复常见示例",
    "医疗数据库问题的不同知识方面（检查、科室、病因、表现、预防），不写诊断",
)
_UNSUPPORTED_GENERAL = re.compile(
    r"天气|下雨|新闻|附近|好吃|停车费|收费标准|几点关门|开放时间|营业时间|"
    r"WiFi|wifi|无线网|微波炉|充电宝|排队|库存|排班|今天有什么号"
)


def _category_counts(
    count: int, ratios: dict[str, float] | None = None,
) -> dict[str, int]:
    if count < 1:
        raise ValueError("count 必须大于 0")
    ratios = ratios or {"medical": 0.3, "navigation": 0.2, "general": 0.2, "mixed": 0.3}
    if set(ratios) != {"medical", "navigation", "general", "mixed"}:
        raise ValueError("category ratios 必须包含 medical/navigation/general/mixed")
    if any(value < 0 for value in ratios.values()) or not 0.999 <= sum(ratios.values()) <= 1.001:
        raise ValueError("category ratios 必须为非负且总和为 1")
    result = {name: int(count * ratio) for name, ratio in ratios.items()}
    remainder = count - sum(result.values())
    ranking = sorted(
        ratios,
        key=lambda name: (count * ratios[name] - result[name], ratios[name]),
        reverse=True,
    )
    for index in range(remainder):
        result[ranking[index % len(ranking)]] += 1
    return result


def _category_batches(
    count: int, batch_size: int, ratios: dict[str, float] | None = None,
) -> list[dict[str, int]]:
    if batch_size < 1:
        raise ValueError("prompt_batch_size 必须大于 0")
    remaining = _category_counts(count, ratios)
    sequence: list[str] = []
    while any(remaining.values()):
        for name in ("medical", "navigation", "general", "mixed"):
            if remaining[name]:
                sequence.append(name)
                remaining[name] -= 1
    return [
        dict(Counter(sequence[start:start + batch_size]))
        for start in range(0, len(sequence), batch_size)
    ]


def _extract_json(text: str) -> dict:
    value = text.strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.I | re.S)
    return json.loads(value)


def _validate_generated_batch(cases: list[dict], batch_index: int) -> None:
    """Reject raw Pro outputs that grounding cannot safely repair."""
    for case in cases:
        case_id = str(case.get("id", ""))
        category = str(case.get("category", ""))
        prompt = str(case.get("prompt", "")).strip()
        if str(case.get("difficulty", "")).lower() not in {"medium", "hard"}:
            raise ValueError(f"DeepSeek 第 {batch_index} 批难度非法: {case_id}")
        expected = case.get("expected") if isinstance(case.get("expected"), dict) else {}
        required = set(expected.get("required_tools", []))
        if category == "general":
            if not required.issubset({"get_time", "get_system_stats"}):
                raise ValueError(
                    f"DeepSeek 第 {batch_index} 批 general 工具非法: {case_id}"
                )
            if _UNSUPPORTED_GENERAL.search(prompt):
                raise ValueError(
                    f"DeepSeek 第 {batch_index} 批 general 不可核实: {case_id}"
                )
        if category in {"navigation", "mixed"}:
            target = str(expected.get("navigation_target", "")).strip()
            if target not in HOSPITAL_LOCATIONS:
                raise ValueError(
                    f"DeepSeek 第 {batch_index} 批目的地非法: {case_id}={target}"
                )
        if category == "medical" and "medical_consult" not in required:
            raise ValueError(f"DeepSeek 第 {batch_index} 批 medical 工具非法: {case_id}")
        if category == "navigation" and required != {"navigate"}:
            raise ValueError(f"DeepSeek 第 {batch_index} 批 navigation 工具非法: {case_id}")
        if category == "mixed" and not {"medical_consult", "navigate"}.issubset(required):
            raise ValueError(f"DeepSeek 第 {batch_index} 批 mixed 工具非法: {case_id}")


def _validate_cases(payload: dict, count: int) -> list[dict]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != count:
        raise ValueError(f"DeepSeek 必须返回 {count} 条 cases")
    allowed = {"medical", "navigation", "general", "mixed"}
    seen = set()
    seen_turns: set[tuple[str, ...]] = set()
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} 不是 object")
        case_id = str(case.get("id", "")).strip()
        prompt = str(case.get("prompt", "")).strip()
        turns = case.get("turns") or [prompt]
        if not isinstance(turns, list) or not 1 <= len(turns) <= 3:
            raise ValueError(f"case {case_id} turns 必须包含 1 至 3 条用户输入")
        turns = [str(item).strip() for item in turns]
        if any(not item for item in turns) or turns[0] != prompt:
            raise ValueError(f"case {case_id} turns[0] 必须等于 prompt 且不得为空")
        case["turns"] = turns
        turn_signature = tuple(turns)
        if turn_signature in seen_turns:
            raise ValueError(f"case {case_id} 与已有 case 对话文本重复")
        seen_turns.add(turn_signature)
        if not case_id or case_id in seen or not prompt:
            raise ValueError(f"case {index} 缺少唯一 id 或 prompt")
        if case.get("category") not in allowed:
            raise ValueError(f"case {case_id} category 非法")
        expected = case.get("expected")
        if not isinstance(expected, dict):
            raise ValueError(f"case {case_id} 缺少 expected")
        required = set(expected.get("required_tools", []))
        if len(turns) > 1 and "query" not in required:
            raise ValueError(f"case {case_id} 多轮样本 required_tools 必须包含 query")
        category = case.get("category")
        if category == "medical" and "medical_consult" not in required:
            raise ValueError(f"case {case_id} medical 缺少 medical_consult")
        if category == "navigation" and required != {"navigate"}:
            raise ValueError(f"case {case_id} navigation 工具标签错误")
        if category == "general":
            if not required.issubset({"get_time", "get_system_stats"}):
                raise ValueError(f"case {case_id} general 工具标签错误")
            if _UNSUPPORTED_GENERAL.search(prompt):
                raise ValueError(f"case {case_id} general 依赖不可用的实时/院内私有信息")
        if category == "mixed" and not {"medical_consult", "navigate"}.issubset(required):
            raise ValueError(f"case {case_id} mixed 必须同时需要医疗和导航")
        if category in {"navigation", "mixed"}:
            target = str(expected.get("navigation_target", "")).strip()
            if target not in HOSPITAL_LOCATIONS:
                raise ValueError(f"case {case_id} navigation_target 非法: {target}")
        if str(case.get("difficulty", "")).lower() not in {"medium", "hard"}:
            raise ValueError(f"case {case_id} pilot 难度必须为 medium/hard")
        seen.add(case_id)
    return cases


def _dedupe_full_dialogues(cases: list[dict]) -> tuple[list[dict], list[dict]]:
    """Remove exact cross-batch duplicates while retaining masked families.

    Same first-turn masking text is valid when follow-up turns differ.  Only a
    complete ``category + turns`` collision is removed.  If an exact text is
    assigned conflicting expected actions, fail loudly instead of silently
    choosing one label.
    """
    unique: list[dict] = []
    seen: dict[tuple[str, tuple[str, ...]], dict] = {}
    duplicates: list[dict] = []
    for case in cases:
        turns = tuple(str(value).strip() for value in case.get("turns") or [])
        key = (str(case.get("category", "")), turns)
        previous = seen.get(key)
        if previous is None:
            seen[key] = case
            unique.append(case)
            continue
        def action_signature(item: dict) -> tuple:
            expected = item.get("expected") or {}
            return (
                tuple(sorted(str(value) for value in expected.get("required_tools", []))),
                str(expected.get("navigation_target") or ""),
            )

        if action_signature(previous) != action_signature(case):
            raise ValueError(
                "重复对话文本对应了冲突 expected: "
                f"{previous.get('id')} vs {case.get('id')}"
            )
        duplicates.append({
            "duplicate_id": case.get("id"),
            "kept_id": previous.get("id"),
            "category": case.get("category"),
            "turns": list(turns),
            "expected_variants": [
                previous.get("expected") or {}, case.get("expected") or {}
            ],
        })
    return unique, duplicates


def _attach_case_support(cases: list[dict]) -> None:
    for case in cases:
        category = str(case.get("category", ""))
        expected = case.get("expected") or {}
        support = {
            "schema_version": "teacher-case-support.v1",
            "category": category,
            "required_tools": list(expected.get("required_tools", [])),
            "forbidden_tools": list(expected.get("forbidden_tools", [])),
            "evaluation_notes": str(expected.get("notes", "")),
        }
        if category in {"navigation", "mixed"}:
            target = str(expected.get("navigation_target") or "")
            aliases = sorted(alias for alias, canonical in LOCATION_ALIASES.items()
                             if canonical == target)
            distractors = [
                location for location in HOSPITAL_LOCATIONS
                if location != target and location in str(case.get("prompt", ""))
            ]
            support["navigation"] = {
                "canonical_target": target,
                "registered": target in HOSPITAL_LOCATIONS,
                "known_aliases": aliases,
                "mentioned_distractors": distractors,
            }
        if category in {"medical", "mixed"}:
            support["medical"] = {
                "source": dict(case.get("medical_source") or {}),
                "followup_support": dict(case.get("followup_support") or {}),
                "must_use_retrieved_evidence": True,
            }
        if category == "general":
            required = set(expected.get("required_tools", []))
            support["general"] = {
                "verification": (
                    "dynamic_tool_result" if required else "stable_reference_answer"
                ),
                "reference_answer": expected.get("reference_answer"),
                "evaluation_criteria": expected.get("evaluation_criteria")
                or expected.get("notes", ""),
            }
        case["support_data"] = support


def _attach_intent_origin(cases: list[dict], origin: str) -> None:
    """Tag prompt provenance without changing the executable expected action."""
    value = str(origin or "").strip()
    if not value:
        return
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,80}", value):
        raise ValueError("intent_origin 只允许字母、数字、._:-")
    for case in cases:
        prompt = str(case.get("prompt", "")).strip()
        category = str(case.get("category", ""))
        turns = list(case.get("turns") or [prompt])
        family_hash = hashlib.sha256(
            f"{category}\0{prompt}".encode("utf-8")
        ).hexdigest()[:20]
        case["intent_metadata"] = {
            "schema_version": "teacher-intent-origin.v1",
            "origin": value,
            "independent_intent": True,
            "semantic_family_id": f"{value}:{family_hash}",
            "category": category,
            "turn_count": len(turns),
            "dialogue_mode": "multi_turn" if len(turns) > 1 else "single_turn",
            "is_paraphrase": False,
            "is_pronunciation_variant": False,
        }


async def generate_prompts(
    count: int, *, medical_database: Path, medical_multiturn_ratio: float,
    seed: int, prompt_batch_size: int = 10, prompt_workers: int = 2,
    prompt_retries: int = 2, checkpoint_dir: Path | None = None,
    prompt_review_dir: Path | None = None,
    prompt_cache_user_ids: tuple[str, ...] = (),
    prompt_model: str = "deepseek-v4-flash",
    prompt_reasoning_effort: str = "low",
    category_ratios: dict[str, float] | None = None,
) -> tuple[list[dict], dict]:
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未定义")
    category_counts = _category_counts(count, category_ratios)
    batches = _category_batches(count, prompt_batch_size, category_ratios)
    cases: list[dict] = []
    usage_total: Counter = Counter()
    model_name = str(prompt_model).strip() or "deepseek-v4-flash"
    if model_name not in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        raise ValueError("prompt_model 仅支持 deepseek-v4-flash 或 deepseek-v4-pro")
    if prompt_reasoning_effort not in {"low", "high", "max"}:
        raise ValueError("prompt_reasoning_effort 仅支持 low/high/max")
    if prompt_workers < 1:
        raise ValueError("prompt_workers 必须大于 0")
    if prompt_retries < 0:
        raise ValueError("prompt_retries 不能小于 0")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir).resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if prompt_review_dir is not None:
        prompt_review_dir = Path(prompt_review_dir).resolve()
        prompt_review_dir.mkdir(parents=True, exist_ok=True)
    cache_lanes = tuple(
        str(value).strip() for value in prompt_cache_user_ids if str(value).strip()
    )
    invalid_lanes = [
        value for value in cache_lanes
        if not re.fullmatch(r"[a-zA-Z0-9\-_]{1,512}", value)
    ]
    if invalid_lanes:
        raise ValueError(f"prompt_cache_user_ids 包含非法 user_id: {invalid_lanes}")
    prompt_gate = asyncio.Semaphore(prompt_workers)

    def write_batch_review(
        batch_index: int, batch_counts: dict[str, int], batch_cases: list[dict],
        usage: dict, lane_id: str, source: str,
    ) -> None:
        """Persist a cheap structural review as soon as a batch is ready.

        This is intentionally deterministic and does not claim medical
        semantic approval.  The later database grounding/preflight and
        evidence-only review remain authoritative for medical trajectories.
        """
        if prompt_review_dir is None:
            return
        hit = int(usage.get("prompt_cache_hit_tokens", 0) or 0)
        miss = int(usage.get("prompt_cache_miss_tokens", 0) or 0)
        review = {
            "schema_version": "teacher-prompt-batch-review.v1",
            "batch_index": batch_index,
            "source": source,
            "status": "pass",
            "case_count": len(batch_cases),
            "category_counts": dict(Counter(
                str(case.get("category", "")) for case in batch_cases
            )),
            "expected_category_counts": dict(batch_counts),
            "cache_lane_user_id": lane_id,
            "usage": dict(usage),
            "prompt_cache_hit_ratio": round(hit / (hit + miss), 6)
            if hit + miss else None,
            "reviewed_at": time.time(),
            "notes": [
                "仅结构/工具标签审核通过；医疗事实仍须本地数据库预检和证据审阅。"
            ],
        }
        (prompt_review_dir / f"batch_{batch_index:04d}.json").write_text(
            json.dumps(review, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    async def generate_batch(
        client: httpx.AsyncClient, batch_index: int, batch_counts: dict[str, int],
    ) -> tuple[int, list[dict], str, dict]:
        batch_count = sum(batch_counts.values())
        distribution = "、".join(
            f"{name} {value}条" for name, value in batch_counts.items()
        )
        # Empty means the provider's account-level/default cache partition.
        # Explicit lanes are opt-in because each user_id may isolate KV cache.
        lane_id = (
            cache_lanes[(batch_index - 1) % len(cache_lanes)]
            if cache_lanes else ""
        )
        user = (
            PROMPT_BATCH_FRAME
            + "\n\n以下是本次请求的批次参数（仅本段随批次变化）：\n"
            + f"批次={batch_index}/{len(batches)}；数量={batch_count}；类别配比={distribution}。"
            + "尽量覆盖否定、指代、口语别名和多约束，并避免同一批内重复。"
            + "mixed 必须同时需要真实医疗查询和明确的物理带路；不得假设前文已有诊断。"
            + "难度只能是 medium 或 hard。医疗文本随后会由本地数据库覆盖，当前只需保证"
            + "工具标签和 mixed 导航目标合法。"
            + "本批是 original_independent_intent 新集：每条只表达一个独立意图，"
            + "避免复用旧集常见的固定问法和短句，优先使用下面的主题方向："
            + INDEPENDENT_INTENT_FOCUS[(batch_index - 1) % len(INDEPENDENT_INTENT_FOCUS)]
        )
        if batch_counts.get("general"):
            user += (
                " 本批 general 的指定主题是："
                f"{GENERAL_FOCUS[(batch_index - 1) % len(GENERAL_FOCUS)]}；"
                "不得换成其他实时或医院私有信息。"
            )
        body = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": PROMPT_SYSTEM},
                {"role": "user", "content": user},
            ],
            "thinking": {"type": "enabled"},
            "reasoning_effort": prompt_reasoning_effort,
            "response_format": {"type": "json_object"},
            "stream": False,
            # DeepSeek documents user_id as the account-side KV-cache and
            # scheduling isolation key.  It is not a local session-id and is
            # never used to merge Agent histories.
            # V4 Pro may spend a large portion of this budget on hidden
            # reasoning.  Keep the full budget even for small batches so the
            # structured content is not truncated to an empty string.
            "max_tokens": 6000,
        }
        if lane_id:
            body["user_id"] = lane_id
        checkpoint = (
            checkpoint_dir / f"batch_{batch_index:04d}.json"
            if checkpoint_dir is not None else None
        )
        if checkpoint is not None and checkpoint.is_file():
            cached = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                cached.get("contract_version") == PROMPT_CONTRACT_VERSION
                and cached.get("category_counts") == batch_counts
            ):
                cached_cases = cached.get("cases")
                if isinstance(cached_cases, list) and len(cached_cases) == batch_count:
                    try:
                        _validate_generated_batch(cached_cases, batch_index)
                    except ValueError:
                        # A prior run may have checkpointed a structurally
                        # invalid Flash response.  Re-request only this batch.
                        cached_cases = None
                    if cached_cases is not None:
                        write_batch_review(
                            batch_index, batch_counts, cached_cases,
                            dict(cached.get("usage") or {}),
                            str(cached.get("cache_lane_user_id") or lane_id),
                            "checkpoint",
                        )
                        return (
                            batch_index, cached_cases,
                            str(cached.get("model", model_name)),
                            dict(cached.get("usage") or {}),
                        )

        last_error: Exception | None = None
        for attempt in range(prompt_retries + 1):
            try:
                async with prompt_gate:
                    response = await client.post(
                        "https://api.deepseek.com/chat/completions",
                        headers={"Authorization": f"Bearer {api_key}",
                                 "Content-Type": "application/json"},
                        json=body,
                    )
                response.raise_for_status()
                data = response.json()
                content = str(data["choices"][0]["message"].get("content", ""))
                batch_cases = _extract_json(content).get("cases")
                if not isinstance(batch_cases, list) or len(batch_cases) != batch_count:
                    raise ValueError(
                        f"DeepSeek 第 {batch_index} 批必须返回 {batch_count} 条 cases"
                    )
                actual = Counter(str(case.get("category", "")) for case in batch_cases)
                if any(actual[name] != expected for name, expected in batch_counts.items()):
                    raise ValueError(
                        f"DeepSeek 第 {batch_index} 批类别错误: "
                        f"actual={dict(actual)}, expected={batch_counts}"
                    )
                _validate_generated_batch(batch_cases, batch_index)
                usage = {
                    key: int(value) for key, value in (data.get("usage") or {}).items()
                    if isinstance(value, (int, float))
                }
                returned_model = str(data.get("model", model_name))
                if checkpoint is not None:
                    checkpoint.write_text(json.dumps({
                        "schema_version": "teacher-prompt-batch.v1",
                        "contract_version": PROMPT_CONTRACT_VERSION,
                        "batch_index": batch_index,
                        "category_counts": batch_counts,
                        "model": returned_model,
                        "cache_lane_user_id": lane_id,
                        "usage": usage,
                        "cases": batch_cases,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")
                write_batch_review(
                    batch_index, batch_counts, batch_cases, usage, lane_id, "generated"
                )
                return batch_index, batch_cases, returned_model, usage
            except (httpx.HTTPError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
                if attempt < prompt_retries:
                    await asyncio.sleep(min(4.0, 2.0 ** attempt))
        raise RuntimeError(
            f"DeepSeek 第 {batch_index} 批在 {prompt_retries + 1} 次尝试后失败: "
            f"{type(last_error).__name__}: {last_error}"
        )

    async with httpx.AsyncClient(timeout=300.0) as client:
        generated = await asyncio.gather(*(
            generate_batch(client, batch_index, batch_counts)
            for batch_index, batch_counts in enumerate(batches, 1)
        ))
        for _batch_index, batch_cases, returned_model, usage in sorted(generated):
            model_name = returned_model
            usage_total.update(usage)
            for case in batch_cases:
                case["id"] = f"case-{len(cases) + 1:06d}"
                cases.append(case)
    MedicalPromptSampler(medical_database, seed=seed).ground_cases(
        cases, multiturn_ratio=medical_multiturn_ratio
    )
    _attach_case_support(cases)
    raw_case_count = len(cases)
    cases, duplicate_dialogues_removed = _dedupe_full_dialogues(cases)
    cases = _validate_cases({"cases": cases}, len(cases))
    actual_counts = Counter(str(case["category"]) for case in cases)
    if any(actual_counts[name] > expected for name, expected in category_counts.items()):
        raise ValueError(
            f"DeepSeek 类别配比错误: actual={dict(actual_counts)}, expected={category_counts}"
        )
    first_prompt_groups: dict[str, list[str]] = {}
    full_dialogue_groups: dict[tuple[str, ...], list[str]] = {}
    for case in cases:
        first_prompt_groups.setdefault(str(case["prompt"]).strip(), []).append(
            str(case["id"])
        )
        full_dialogue_groups.setdefault(
            tuple(str(value).strip() for value in case.get("turns") or []),
            [],
        ).append(str(case["id"]))
    duplicate_first_prompt_groups = [
        {"prompt": prompt, "case_ids": ids}
        for prompt, ids in first_prompt_groups.items() if len(ids) > 1
    ]
    if prompt_review_dir is not None:
        (prompt_review_dir / "summary.json").write_text(json.dumps({
            "schema_version": "teacher-prompt-global-review.v1",
            "status": "pass_with_masked_prompt_families"
            if duplicate_first_prompt_groups else "pass",
            "case_count": len(cases),
            "raw_case_count": raw_case_count,
            "duplicates_removed": len(duplicate_dialogues_removed),
            "duplicate_dialogues_removed": duplicate_dialogues_removed,
            "unique_first_prompts": len(first_prompt_groups),
            "duplicate_first_prompt_groups": duplicate_first_prompt_groups,
            "duplicate_full_dialogues": [
                {"turns": list(turns), "case_ids": ids}
                for turns, ids in full_dialogue_groups.items() if len(ids) > 1
            ],
            "notes": [
                "同一首轮遮蔽句可出现在不同多轮样本中；完整 turns 仍必须唯一。",
                "这不是医疗语义通过证明，医疗事实仍须本地预检和证据审阅。",
            ],
            "reviewed_at": time.time(),
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "model": model_name,
        "prompt_reasoning_effort": prompt_reasoning_effort,
        "usage": dict(usage_total),
        "prompt_batches": len(batches),
        "raw_case_count": raw_case_count,
        "duplicates_removed": len(duplicate_dialogues_removed),
        "prompt_batch_size": prompt_batch_size,
        "prompt_workers": prompt_workers,
        "prompt_cache_user_ids": list(cache_lanes),
        "prompt_cache_layout": "stable-frame-then-batch-parameters-v1",
        "prompt_review_dir": str(prompt_review_dir) if prompt_review_dir else None,
        "unique_first_prompts": len(first_prompt_groups),
        "duplicate_first_prompt_group_count": len(duplicate_first_prompt_groups),
        "duplicate_full_dialogue_count": sum(
            len(ids) > 1 for ids in full_dialogue_groups.values()
        ),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "prompt_system_sha256": hashlib.sha256(PROMPT_SYSTEM.encode("utf-8")).hexdigest(),
        "generated_at": time.time(),
        "medical_prompt_source": str(medical_database.resolve()),
        "medical_grounding": "train rows only; no LLM-authored medical facts",
        "medical_multiturn_ratio": medical_multiturn_ratio,
        "seed": seed,
        "category_ratios": dict(category_ratios or {
            "medical": 0.3, "navigation": 0.2, "general": 0.2, "mixed": 0.3,
        }),
    }
    return cases, metadata


def _command_summary(result) -> list[dict]:
    rows = []
    for command, tool_result in result.commands:
        row: dict[str, Any] = {
            "name": command.name,
            "type": command.type,
            "params": command.params,
            "success": tool_result.success,
            "error": tool_result.error,
        }
        if command.name == "medical_consult" and tool_result.data:
            try:
                medical = json.loads(tool_result.data)
                row["medical"] = {
                    "status": medical.get("status"),
                    "intent": medical.get("intent"),
                    "normalized_terms": medical.get("normalized_terms"),
                    "red_flags": medical.get("red_flags"),
                    "questions": medical.get("questions"),
                    "recommended_destination": medical.get("recommended_destination"),
                    "departments": medical.get("departments"),
                    "message": medical.get("message"),
                    "retrieval": medical.get("retrieval"),
                }
            except json.JSONDecodeError:
                row["medical_parse_error"] = True
        rows.append(row)
    return rows


async def run_teacher(
    cases: list[dict], output_dir: Path, *, workers: int = 3,
) -> list[dict]:
    if workers < 1:
        raise ValueError("workers 必须大于 0")
    trajectory_dir = output_dir / "trajectories"
    agent = Agent(AgentConfig(
        server_url="http://127.0.0.1:8000",
        model_name="deepseek-v4-flash",
        default_mode="Benchmark",
        benchmark_enabled=True,
        benchmark_max_tokens=8192,
        emit_agent_events=False,
        trajectory_enabled=True,
        trajectory_dir=str(trajectory_dir),
        camera_backend="dummy",
        navigation_location_profile="hospital",
        medical_dense_enabled=True,
        online_max_concurrency=workers,
    ))
    agent.initialize()
    case_gate = asyncio.Semaphore(workers)
    case_checkpoint_dir = output_dir / "case_checkpoints"
    case_checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_row(row: dict[str, Any]) -> dict[str, Any]:
        safe_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", str(row.get("id", "case")))
        (case_checkpoint_dir / f"{safe_id}.json").write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return row

    def make_mode() -> BenchmarkMode:
        skills = SkillManager(agent.skill_loader)
        router = CallRouter(mcp=agent.mcp, skill_manager=skills)
        router.update_registered_names()
        mode = BenchmarkMode(
            api=agent.api,
            mcp=agent.mcp,
            skill_manager=skills,
            call_router=router,
        )
        mode.config.history_visible_entries = agent.config.history_visible_entries
        return mode

    async def run_case(case: dict) -> dict:
        async with case_gate:
            mode = make_mode()
            session = Session(mode="Benchmark", need_clear=True)
            session.id = str(case["id"])
            session.benchmark_context = {
                "teacher_case_id": str(case["id"]),
                "teacher_category": str(case["category"]),
                "medical_source": dict(case.get("medical_source") or {}),
                "teacher_workers": workers,
            }
            mode.session = session
            started = time.monotonic()
            try:
                scripted_turns = list(case.get("turns") or [case["prompt"]])
                next_index = 1

                def followup_provider(_question: str) -> str:
                    nonlocal next_index
                    if next_index >= len(scripted_turns):
                        return ""
                    value = scripted_turns[next_index]
                    next_index += 1
                    return value

                turn_rows = []
                current = scripted_turns[0]
                result = None
                while current and len(turn_rows) < 3:
                    result = await mode.loop(
                        user_input=current,
                        tool_context_extra={
                            "query_followup_provider": followup_provider,
                            "input_metadata": {
                                "source": "benchmark_text" if not turn_rows else "query_mock",
                                "asr_text": current,
                            },
                        },
                    )
                    turn_rows.append({
                        "turn": len(turn_rows) + 1,
                        "input": current,
                        "status": "completed" if result.text else "error",
                        "end_reason": result.turn_end_reason,
                        "output": result.text,
                        "commands": _command_summary(result),
                    })
                    if result.turn_end_reason == "query":
                        current = str(result.continuation_audio.get("followup_text", ""))
                        continue
                    break
                assert result is not None
                return checkpoint_row({
                    "id": case["id"], "category": case["category"],
                    "prompt": case["prompt"], "expected": case["expected"],
                    "medical_source": case.get("medical_source"),
                    "status": "completed" if result.session_ended else "error",
                    "session_end_reason": result.turn_end_reason,
                    "final": result.text,
                    "commands": _command_summary(result),
                    "turns": turn_rows,
                    "token_usage": summarize_token_usage(
                        session.model_call_records, agent.config.context_window_tokens
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                })
            except Exception as exc:
                return checkpoint_row({
                    "id": case["id"], "category": case["category"],
                    "prompt": case["prompt"], "expected": case["expected"],
                    "medical_source": case.get("medical_source"),
                    "status": "error", "error": f"{type(exc).__name__}: {exc}",
                    "token_usage": summarize_token_usage(
                        session.model_call_records, agent.config.context_window_tokens
                    ),
                    "elapsed_seconds": time.monotonic() - started,
                })

    try:
        # gather preserves input order while cases execute concurrently.
        rows = list(await asyncio.gather(*(run_case(case) for case in cases)))
    finally:
        await agent.shutdown()
    return rows


async def async_main(args) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.prompts_file:
        source = json.loads(Path(args.prompts_file).read_text(encoding="utf-8"))
        cases = source.get("cases")
        if not isinstance(cases, list) or len(cases) != args.count:
            raise ValueError(f"prompts_file 必须包含 {args.count} 条 cases")
        medical_cases = [
            case for case in cases
            if case.get("category") in {"medical", "mixed"}
        ]
        sampler = MedicalPromptSampler(Path(args.medical_database), seed=args.seed)
        regrounded = any(not case.get("medical_source") for case in medical_cases)
        if regrounded:
            sampler.ground_cases(
                cases, multiturn_ratio=args.medical_multiturn_ratio
            )
        sampler.hydrate_existing_cases(cases)
        MedicalPromptSampler.normalize_existing_cases(cases)
        _attach_case_support(cases)
        cases = _validate_cases({"cases": cases}, args.count)
        generation = dict(source.get("generation") or {})
        generation["reused_from"] = str(Path(args.prompts_file).resolve())
        generation["medical_regrounded"] = regrounded
        generation["medical_prompt_source"] = str(Path(args.medical_database).resolve())
    else:
        cases, generation = await generate_prompts(
            args.count,
            medical_database=Path(args.medical_database),
            medical_multiturn_ratio=args.medical_multiturn_ratio,
            seed=args.seed,
            prompt_batch_size=args.prompt_batch_size,
            prompt_workers=args.prompt_workers,
            prompt_retries=args.prompt_retries,
            checkpoint_dir=output_dir / "prompt_batches",
            prompt_review_dir=output_dir / "prompt_batch_reviews",
            prompt_cache_user_ids=tuple(
                value.strip() for value in str(args.prompt_cache_user_ids).split(",")
                if value.strip()
            ),
            prompt_model=args.prompt_model,
            prompt_reasoning_effort=args.prompt_reasoning_effort,
            category_ratios=args.category_ratios,
        )
    selected_ids = {
        value.strip()
        for value in ([args.case_id] if args.case_id else [])
        + str(args.case_ids or "").split(",")
        if value.strip()
    }
    if selected_ids:
        available = {str(case["id"]) for case in cases}
        missing = sorted(selected_ids - available)
        if missing:
            raise ValueError(f"未找到 case: {', '.join(missing)}")
        cases = [case for case in cases if str(case["id"]) in selected_ids]
    _attach_intent_origin(cases, args.intent_origin)
    if args.intent_origin:
        generation["intent_origin"] = args.intent_origin
        generation["intent_origin_schema"] = "teacher-intent-origin.v1"
        generation["intent_origin_counts"] = dict(Counter(
            str(case.get("category", "")) for case in cases
        ))
        generation["intent_dialogue_counts"] = dict(Counter(
            "multi_turn" if len(case.get("turns") or []) > 1 else "single_turn"
            for case in cases
        ))
    (output_dir / "prompts.json").write_text(json.dumps({
        "schema_version": "teacher-prompts.v1", "generation": generation, "cases": cases,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.prompts_only:
        return
    existing_rows: dict[str, dict] = {}
    runs_path = output_dir / "runs.json"
    if args.resume and runs_path.is_file():
        previous = json.loads(runs_path.read_text(encoding="utf-8"))
        existing_rows = {
            str(row.get("id")): row for row in previous.get("runs", [])
            if row.get("status") == "completed"
        }
    if args.resume:
        for checkpoint in (output_dir / "case_checkpoints").glob("*.json"):
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            if row.get("status") == "completed":
                existing_rows[str(row.get("id"))] = row
    pending = [case for case in cases if str(case["id"]) not in existing_rows]
    teacher_started = time.monotonic()
    generated_rows = (
        await run_teacher(pending, output_dir, workers=args.workers) if pending else []
    )
    teacher_elapsed = time.monotonic() - teacher_started
    generated_by_id = {str(row["id"]): row for row in generated_rows}
    rows = [
        generated_by_id.get(str(case["id"])) or existing_rows[str(case["id"])]
        for case in cases
    ]
    runs_payload = {
        "schema_version": "teacher-pilot.v1", "teacher_model": "deepseek-v4-flash",
        "prompt_model": str(generation.get("model", args.prompt_model)),
        "execution": {
            "workers": args.workers,
            "case_count": len(cases),
            "executed_case_count": len(pending),
            "resumed_completed_count": len(cases) - len(pending),
            "elapsed_seconds": teacher_elapsed,
            "completed": sum(row.get("status") == "completed" for row in rows),
            "errors": sum(row.get("status") != "completed" for row in rows),
            "token_usage": {
                "prompt_tokens_sum": sum(
                    int((row.get("token_usage") or {}).get("prompt_tokens_sum", 0))
                    for row in rows
                ),
                "completion_tokens_sum": sum(
                    int((row.get("token_usage") or {}).get("completion_tokens_sum", 0))
                    for row in rows
                ),
                "total_tokens_sum": sum(
                    int((row.get("token_usage") or {}).get("total_tokens_sum", 0))
                    for row in rows
                ),
                "peak_prompt_tokens": max((
                    int((row.get("token_usage") or {}).get("peak_prompt_tokens", 0))
                    for row in rows
                ), default=0),
                "context_overflow_cases": sum(
                    bool((row.get("token_usage") or {}).get("context_overflow"))
                    for row in rows
                ),
                "provider_total_overflow_cases": sum(
                    int((row.get("token_usage") or {}).get(
                        "provider_total_overflow_calls", 0
                    ) or 0) > 0 for row in rows
                ),
            },
        },
        "runs": rows,
    }
    (output_dir / "runs.json").write_text(
        json.dumps(runs_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_dir / "audit.json").write_text(
        json.dumps(audit_runs(output_dir, runs_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 DeepSeek Teacher trajectory")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--output-dir", default="/data/structure/teacher_trajectories/pilot_10")
    parser.add_argument("--prompts-file", default="", help="复用已审核的 prompts.json")
    parser.add_argument("--case-id", default="", help="仅重跑指定 case")
    parser.add_argument(
        "--case-ids", default="",
        help="逗号分隔的 case id 子集；按原提示词顺序输出",
    )
    parser.add_argument(
        "--medical-database",
        default="/data/structure/med_database/med_search.sqlite",
        help="医疗 prompt 的本地 SQLite 来源",
    )
    parser.add_argument(
        "--medical-multiturn-ratio", type=float, default=0.10,
        help="医疗/混合样本中掩盖字段构造多轮的比例；默认 10%，单轮是主要数据",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--prompt-batch-size", type=int, default=10,
        help="DeepSeek V4 Pro 每次生成的结构 case 数，批量预训练时自动分批",
    )
    parser.add_argument(
        "--prompt-workers", type=int, default=2,
        help="并行生成提示词结构的 DeepSeek V4 Pro 请求数",
    )
    parser.add_argument(
        "--prompt-retries", type=int, default=2,
        help="每个提示词批次失败后的额外重试次数；成功批次会即时 checkpoint",
    )
    parser.add_argument(
        "--prompt-cache-user-ids",
        default="",
        help=(
            "DeepSeek 上游 user_id cache lane，逗号分隔。它只用于 KV-cache/"
            "调度隔离，不共享本地 Agent session 历史；留空使用账号默认缓存分区。"
        ),
    )
    parser.add_argument(
        "--prompt-model", default="deepseek-v4-flash",
        choices=("deepseek-v4-flash", "deepseek-v4-pro"),
        help="提示词候选生成模型；默认 Flash，Pro 仅用于质量对照。",
    )
    parser.add_argument(
        "--prompt-reasoning-effort", default="low",
        choices=("low", "high", "max"),
        help="提示词候选生成的推理强度；Flash 默认 low 以降低延迟和费用。",
    )
    parser.add_argument(
        "--category-ratios", default="",
        help=(
            "可选类别配比，例如 medical=0.5,navigation=0.2,general=0.1,mixed=0.2；"
            "默认 0.3/0.2/0.2/0.3。"
        ),
    )
    parser.add_argument(
        "--intent-origin", default="",
        help="为本批 case 写入 intent_metadata，例如 original_independent；留空不添加。",
    )
    parser.add_argument(
        "--prompts-only", action="store_true",
        help="只生成并落盘 prompts.json，不启动 Router 或执行 Teacher trajectory",
    )
    parser.add_argument(
        "--workers", type=int, default=3,
        help="并行执行的独立 Teacher session 数；每个 session 内仍严格串行",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="复用输出目录 runs.json 中已完成 case，仅重跑失败或缺失项",
    )
    args = parser.parse_args()
    if args.category_ratios:
        parsed_ratios = {}
        for item in args.category_ratios.split(","):
            name, separator, raw_value = item.partition("=")
            if not separator:
                raise ValueError("category-ratios 格式应为 name=value,name=value")
            parsed_ratios[name.strip()] = float(raw_value)
        args.category_ratios = parsed_ratios
    else:
        args.category_ratios = None
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
