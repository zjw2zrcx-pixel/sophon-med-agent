"""Build evidence-grounded medical prompts from the local training database."""
from __future__ import annotations

import hashlib
import random
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

MIXED_SAFE_TARGETS = ("挂号处", "服务台", "门诊大厅")


@dataclass(frozen=True)
class MedicalSource:
    table: str
    row_id: int
    question: str
    answer: str
    source: str

    def metadata(self) -> Dict[str, Any]:
        return {
            "database_table": self.table,
            "row_id": self.row_id,
            "source": self.source,
            "question": self.question,
            "reference_answer": self.answer[:600],
            "answer_sha256": hashlib.sha256(self.answer.encode("utf-8")).hexdigest(),
        }


class MedicalPromptSampler:
    """Sample only train-provenance rows without asking an LLM to invent facts."""

    def __init__(self, database: Path, seed: int = 20260812) -> None:
        self.database = Path(database).resolve()
        if not self.database.is_file():
            raise FileNotFoundError(f"医疗数据库不存在: {self.database}")
        self.random = random.Random(seed)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.database}?mode=ro", uri=True)

    def hydrate_existing_cases(self, cases: List[dict]) -> List[dict]:
        """Restore evidence text from SQLite; never trust prompt-file copies."""
        with self._connect() as connection:
            for case in cases:
                if case.get("category") not in {"medical", "mixed"}:
                    continue
                source = case.get("medical_source")
                if not isinstance(source, dict):
                    continue
                table = str(source.get("database_table", ""))
                if table not in {"documents", "facts"}:
                    raise ValueError(f"case {case.get('id')} 医疗来源表非法: {table}")
                row_id = int(source.get("row_id", 0) or 0)
                row = connection.execute(
                    f"SELECT answer,source FROM {table} WHERE id=?", (row_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"case {case.get('id')} 医疗来源行不存在: {table}/{row_id}")
                answer, dataset = str(row[0]), str(row[1])
                answer_hash = hashlib.sha256(answer.encode("utf-8")).hexdigest()
                declared_hash = str(source.get("answer_sha256", ""))
                if declared_hash and declared_hash != answer_hash:
                    raise ValueError(f"case {case.get('id')} 医疗来源 hash 不匹配")
                if not dataset.endswith("/train"):
                    raise ValueError(f"case {case.get('id')} 不能使用非 train 医疗来源")
                source["source"] = dataset
                source["answer_sha256"] = answer_hash
                source["reference_answer"] = answer[:600]
        return cases

    @staticmethod
    def _followup_support(source: MedicalSource, turns: List[str]) -> Dict[str, Any]:
        snippets = [
            item.strip() for item in re.split(r"[。！？；\n]+", source.answer)
            if item.strip()
        ][:6]
        general_reply = (
            "这是一般医学知识咨询，不是针对我个人做诊断。"
            f"请只根据知识库介绍这个问题：{source.question}"
        )
        return {
            "schema_version": "medical-followup-support.v1",
            "source": {
                "table": source.table, "row_id": source.row_id,
                "dataset": source.source,
                "answer_sha256": hashlib.sha256(source.answer.encode("utf-8")).hexdigest(),
            },
            "knowledge_snippets": snippets,
            "clarification_fields": [
                "consultation_scope", "duration", "severity", "location",
                "associated_symptoms", "age_group", "medical_history",
                "allergy_and_current_medication",
            ],
            "user_replies": turns[1:] if len(turns) > 1 else [general_reply],
            "max_session_turns": 3,
            "safety": "补充数据只帮助澄清或核对证据，不是诊断结论",
        }

    def _document(self, used: set[int]) -> MedicalSource:
        with self._connect() as connection:
            maximum = int(connection.execute("SELECT max(id) FROM documents").fetchone()[0])
            for _ in range(300):
                row_id = self.random.randint(1, maximum)
                row = connection.execute(
                    "SELECT id,question,answer,source FROM documents WHERE id>=? "
                    "AND length(question) BETWEEN 6 AND 80 "
                    "AND length(answer)>=30 ORDER BY id LIMIT 1",
                    (row_id,),
                ).fetchone()
                if row and int(row[0]) not in used and str(row[3]).endswith("/train"):
                    used.add(int(row[0]))
                    return MedicalSource("documents", int(row[0]), str(row[1]).strip(),
                                         str(row[2]).strip(), str(row[3]))
        raise RuntimeError("无法从 documents 抽取足够的 train 医疗文本")

    def _masked_fact(self) -> tuple[MedicalSource, List[str]]:
        with self._connect() as connection:
            maximum = int(connection.execute("SELECT max(id) FROM facts").fetchone()[0])
            for _ in range(300):
                row_id = self.random.randint(1, maximum)
                row = connection.execute(
                    "SELECT id,subject,aspect,answer,source FROM facts WHERE id>=? "
                    "AND quality>=0.8 AND length(subject) BETWEEN 2 AND 18 "
                    "AND length(aspect) BETWEEN 2 AND 18 AND length(answer)>=8 "
                    "ORDER BY id LIMIT 1",
                    (row_id,),
                ).fetchone()
                if row and str(row[4]).endswith("/train"):
                    subject, aspect = str(row[1]).strip(), str(row[2]).strip()
                    full_question = f"{subject}的{aspect}是什么？"
                    source = MedicalSource(
                        "facts", int(row[0]), full_question,
                        str(row[3]).strip(), str(row[4]),
                    )
                    # The first turn hides the database subject; the next turn
                    # restores it and repeats the complete grounded question.
                    turns = [
                        f"我想咨询一种疾病的{aspect}，但还没说具体名称。",
                        f"具体是{subject}。完整问题是：{full_question}",
                    ]
                    return source, turns
        raise RuntimeError("无法从 facts 抽取可掩盖字段的 train 医疗文本")

    def ground_cases(self, cases: List[dict], multiturn_ratio: float = 0.15) -> List[dict]:
        if not 0.0 <= multiturn_ratio <= 1.0:
            raise ValueError("medical_multiturn_ratio 必须在 0 到 1 之间")
        medical = [case for case in cases if case.get("category") in {"medical", "mixed"}]
        multi_count = min(
            len([case for case in medical if case.get("category") == "medical"]),
            round(len(medical) * multiturn_ratio),
        )
        multi_ids = {
            id(case) for case in [
                case for case in medical if case.get("category") == "medical"
            ][:multi_count]
        }
        used: set[int] = set()
        for case in medical:
            if id(case) in multi_ids:
                source, turns = self._masked_fact()
                case["prompt"] = turns[0]
                case["turns"] = turns
                required = list(case["expected"].get("required_tools", []))
                if "query" not in required:
                    required.append("query")
                case["expected"]["required_tools"] = required
                case["expected"]["notes"] = (
                    "来自本地 train 数据的受控多轮医疗问题；第一轮必须 medical_consult，"
                    "按工具追问状态调用 query，补充后再次检索并只依据证据回答。"
                )
                case["medical_source"] = {**source.metadata(), "mask_strategy": "subject"}
                case["followup_support"] = self._followup_support(source, turns)
                continue

            source = self._document(used)
            prompt = source.question
            if case.get("category") == "mixed":
                target = MIXED_SAFE_TARGETS[source.row_id % len(MIXED_SAFE_TARGETS)]
                case["expected"]["navigation_target"] = target
                prompt = f"{prompt}，另外请带我去{target}。"
            case["prompt"] = prompt
            case["turns"] = [prompt]
            case["expected"]["required_tools"] = [
                name for name in case["expected"].get("required_tools", [])
                if name != "query"
            ]
            if case.get("category") == "mixed":
                case["expected"]["notes"] = (
                    "医疗问题来自本地 train 数据；先用完整原话调用 medical_consult，"
                    f"再按用户明确要求导航至{case['expected']['navigation_target']}，"
                    "最终回答不得超出工具证据。"
                )
            else:
                case["expected"]["notes"] = (
                    "完整问题来自本地 train 数据；调用 medical_consult 后按工具状态"
                    "选择 query 或 speak，结论不得超出检索证据。"
                )
            case["medical_source"] = source.metadata()
            case["followup_support"] = self._followup_support(source, [prompt])
        return cases

    @staticmethod
    def normalize_existing_cases(cases: List[dict]) -> List[dict]:
        """Refresh derived labels when reusing an already grounded prompt set."""
        for case in cases:
            if case.get("category") not in {"medical", "mixed"}:
                continue
            source = case.get("medical_source")
            if not isinstance(source, dict) or not str(source.get("source", "")).endswith("/train"):
                continue
            # Never trust LLM-authored demographic/symptom follow-ups.  Rebuild
            # support deterministically from the audited database metadata.
            source_object = MedicalSource(
                str(source.get("database_table", "documents")),
                int(source.get("row_id", 0) or 0),
                str(source.get("question", "")),
                str(source.get("reference_answer", "")),
                str(source.get("source", "")),
            )
            support = MedicalPromptSampler._followup_support(
                source_object, list(case.get("turns") or [case.get("prompt", "")])
            )
            # reference_answer is intentionally truncated in exported metadata;
            # provenance must keep the verified full database answer hash.
            support["source"]["answer_sha256"] = str(source.get("answer_sha256", ""))
            case["followup_support"] = support
            expected = case.setdefault("expected", {})
            if case.get("category") == "mixed":
                row_id = int(source.get("row_id", 0) or 0)
                target = MIXED_SAFE_TARGETS[row_id % len(MIXED_SAFE_TARGETS)]
                question = str(source.get("question", "")).strip()
                prompt = f"{question}，另外请带我去{target}。"
                existing_turns = list(case.get("turns") or [])
                case["prompt"] = prompt
                # Preflight may have materialized a database-grounded user
                # reply for need_more_info. Normalize only turn 1; never erase
                # later user input when reusing the fixed prompt set.
                case["turns"] = [prompt] + existing_turns[1:3]
                expected["navigation_target"] = target
                expected["notes"] = (
                    "医疗问题来自本地 train 数据；先用完整原话调用 medical_consult，"
                    f"再按用户明确要求导航至{target}，最终回答不得超出工具证据。"
                )
            elif source.get("mask_strategy"):
                expected["notes"] = (
                    "来自本地 train 数据的受控多轮医疗问题；第一轮必须 medical_consult，"
                    "按工具追问状态调用 query，补充后再次检索并只依据证据回答。"
                )
            else:
                expected["notes"] = (
                    "完整问题来自本地 train 数据；调用 medical_consult 后按工具状态"
                    "选择 query 或 speak，结论不得超出检索证据。"
                )
        return cases


__all__ = ["MedicalPromptSampler", "MedicalSource"]
