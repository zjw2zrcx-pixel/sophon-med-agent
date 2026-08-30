"""Shared deterministic evidence policy for medical spoken responses."""
from __future__ import annotations

import json
import re
from typing import Any, Iterable


CLINICAL_DEPARTMENTS = {
    "急诊科", "呼吸内科", "心血管内科", "消化内科", "神经内科", "内分泌科",
    "肾内科", "血液科", "风湿免疫科", "普通外科", "骨科", "泌尿外科",
    "神经外科", "胸外科", "妇科", "产科", "儿科", "眼科", "耳鼻喉科",
    "口腔科", "皮肤科", "精神心理科", "康复医学科", "中医科",
}


def department_names(values: Iterable[Any]) -> set[str]:
    result: set[str] = set()
    for value in values:
        if isinstance(value, str):
            result.add(value.strip())
        elif isinstance(value, dict):
            for key in ("department", "name", "canonical", "destination"):
                text = str(value.get(key, "")).strip()
                if text:
                    result.add(text)
    return result


def consultation_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def unsupported_departments(
    text: str, consultation: Any, additional_allowed: Iterable[str] = (),
) -> list[str]:
    medical = consultation_dict(consultation)
    allowed = department_names(medical.get("departments", []))
    destination = str(medical.get("recommended_destination", "")).strip()
    if destination:
        allowed.add(destination)
    allowed.update(str(value).strip() for value in additional_allowed if str(value).strip())
    mentioned = {name for name in CLINICAL_DEPARTMENTS if name in str(text)}
    return sorted(mentioned - allowed)


def _spoken_text(value: Any, max_chars: int = 100) -> str:
    """Normalize a trusted tool string for short TTS playback."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    # The deployed VITS lexicon does not accept every typography character.
    text = text.replace("、", "，").replace("；", "，")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip("，。；：、 ") + "。"


def grounded_medical_speech(
    consultation: Any, *, navigation_target: str = "", max_chars: int = 180,
) -> str:
    """Build a short, evidence-bounded medical response without an LLM turn.

    This is intentionally conservative.  It only verbalizes fields emitted by
    ``medical_consult`` and never turns drug associations into personalized
    medication instructions.  A follow-up question remains owned by ``query``
    and therefore returns an empty string here.
    """
    medical = consultation_dict(consultation)
    status = str(medical.get("status", "") or "").strip()
    if not medical or status == "out_of_scope":
        return ""
    questions = [
        _spoken_text(item, 120)
        for item in medical.get("questions", [])
        if str(item or "").strip()
    ]
    if status in {"need_more_info", "ambiguous"} and questions:
        return ""

    message = _spoken_text(medical.get("message"), 110)
    urgency = str(medical.get("urgency", "") or "").strip()
    red_flags = [
        _spoken_text(item, 45)
        for item in medical.get("red_flags", [])[:2]
        if str(item or "").strip()
    ]
    destination = _spoken_text(medical.get("recommended_destination"), 30)
    navigation = _spoken_text(navigation_target, 30)
    intent = str(medical.get("intent", "") or "").strip()

    if status == "urgent" or urgency == "emergency" or red_flags:
        lead = message or "检测到可能需要紧急处理的危险信号。"
        if red_flags:
            lead += "相关表现包括" + "，".join(red_flags) + "。"
        if destination:
            lead += f"请立即前往{destination}或联系当地急救服务。"
        return _spoken_text(lead, max_chars)

    if intent == "medication":
        # The retriever may return prescription/antibiotic graph edges.  Those
        # are evidence associations, not a patient-specific prescription.
        diagnosis = ""
        for item in medical.get("normalized_terms", []):
            if isinstance(item, dict) and "疾病" in str(item.get("type", "")):
                diagnosis = _spoken_text(item.get("canonical"), 30)
                if diagnosis:
                    break
        if bool(medical.get("medication_allowed", False)):
            lead = (
                (f"已记录您说明的{diagnosis}。" if diagnosis else "已记录您说明的诊断情况。")
                + "当前资料不能判断具体药物的剂量、禁忌和是否适合您。"
                + "请以已有医嘱为准，不要自行叠加或更换药物。"
            )
            suffix = "如尚未拿到处方，可携带年龄、过敏史和正在使用的药物到药房核对。"
        else:
            lead = "仅凭当前描述不能安全确定具体药物。"
            suffix = "请先补充医生诊断、症状持续时间和过敏或既往用药情况。"
    else:
        names = sorted(department_names(medical.get("departments", [])))[:2]
        if intent == "department_recommendation" and names:
            lead = (
                "本地分诊资料建议优先咨询" + "或".join(names)
                + "，仅供分诊参考，不代表已经确诊。"
            )
        else:
            evidence = [
                item for item in medical.get("evidence", [])
                if isinstance(item, dict) and str(item.get("text", "")).strip()
            ]
            associations = [
                item for item in medical.get("associations", [])
                if isinstance(item, dict)
            ]
            if evidence:
                lead = "本地医疗资料显示，" + _spoken_text(
                    evidence[0].get("text"), 105
                )
            elif associations:
                item = associations[0]
                matched = _spoken_text(item.get("matched"), 35)
                relation = _spoken_text(item.get("relation"), 25)
                related = _spoken_text(item.get("related"), 45)
                if not (matched and relation and related):
                    lead = message
                else:
                    lead = f"本地资料显示，{matched}与{related}存在{relation}关联，这不代表诊断。"
            else:
                lead = message
        suffix = "请咨询专业医生。"

    if navigation:
        lead = lead.rstrip("。") + f"。已开始导航至{navigation}，请跟紧我。"
    if not lead:
        return ""
    # Keep the safety qualification intact when bounding speech duration.
    body_limit = max(40, max_chars - len(suffix))
    return _spoken_text(lead, body_limit).rstrip("。") + "。" + suffix


__all__ = [
    "CLINICAL_DEPARTMENTS", "consultation_dict", "department_names",
    "grounded_medical_speech", "unsupported_departments",
]
