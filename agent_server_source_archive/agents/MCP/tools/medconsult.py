"""Single-call medical consultation tool backed by the Python retriever."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import re
import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Tuple

from ...Medical.dense import DenseDocumentIndex, DenseRetriever, EmbeddingClient
from ...Medical.retriever import MedicalRetriever
from ..base import Tool, ToolContext, ToolResult


logger = logging.getLogger(__name__)
_DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[3] / "med_database" / "med_search.sqlite"
_DEFAULT_DENSE_MANIFEST = (
    Path(__file__).resolve().parents[3]
    / "med_database"
    / "medical_document_vectors_256.manifest.json"
)
_BUILD_COMMAND = (
    "/data/env310/bin/python -m agents.Medical.build_index "
    "--output /data/structure/med_database/med_search.sqlite"
)


class MedicalConsultTool(Tool):
    """Natural-language facade for the local medical knowledge index."""

    name = "medical_consult"
    description = (
        "用一句自然语言查询本地医疗知识库。工具会自动理解口语症状、识别咨询意图、"
        "识别否定症状和组合危险信号，并返回带来源的科室候选及少量检索证据；"
        "非医疗计算、通用知识或纯导航请求会返回 out_of_scope，不会强行模糊匹配医学实体；"
        "只需原样传入用户问题，不要先猜实体名，"
        "也不要拆成图数据库子命令。"
    )
    param_schema = {"query": "必填；用户原始医疗问题，请保留症状、持续时间和伴随表现"}
    modes = ["Voice", "Benchmark"]
    harness_metadata = {
        "effect": "READ", "idempotent": True,
        "produces": ["medical.consultation"], "invalidates": [],
        "retry": {
            "max_attempts": 2,
            "allowed_errors": ["NOT_FOUND", "TIMEOUT", "TEMPORARY_UNAVAILABLE"],
        },
    }

    def __init__(self, dense_enabled: Optional[bool] = None) -> None:
        self._cache_lock = threading.Lock()
        self._cached: Optional[Tuple[Tuple[Any, ...], MedicalRetriever]] = None
        self._dense_enabled_override = dense_enabled
        self._prewarm_thread: Optional[threading.Thread] = None

    def start_prewarm(self) -> None:
        """Load immutable medical indexes in the background during startup."""
        enabled = os.environ.get("MEDICAL_INDEX_PREWARM", "1").strip().lower()
        if enabled in {"0", "false", "no", "off"}:
            return
        if self._prewarm_thread is not None and self._prewarm_thread.is_alive():
            return

        def run() -> None:
            started = time.perf_counter()
            try:
                path = self._index_path()
                if not path.is_file():
                    logger.warning("跳过医疗索引预热，索引不存在: %s", path)
                    return
                retriever = self._retriever(path)
                # ``DenseDocumentIndex.warm`` only faults the vector matrix.
                # Run one representative consultation as well, so SQLite's
                # sparse/document pages and the embedding inference path are
                # hot before the first real voice request arrives.
                probe = os.environ.get(
                    "MEDICAL_RETRIEVAL_PREWARM_QUERY", "咳嗽怎么办"
                ).strip()
                if probe:
                    probe_started = time.perf_counter()
                    retriever.consult(probe)
                    logger.info(
                        "医疗完整检索预热完成: %.1f ms (query=%s)",
                        (time.perf_counter() - probe_started) * 1000,
                        probe[:30],
                    )
                logger.info(
                    "医疗检索器后台预热完成: %.1f ms",
                    (time.perf_counter() - started) * 1000,
                )
            except Exception as exc:  # request path retains normal diagnostics
                logger.warning("医疗检索器后台预热失败，将在首个请求重试: %s", exc)

        self._prewarm_thread = threading.Thread(
            target=run, name="medical-index-prewarm", daemon=True
        )
        self._prewarm_thread.start()

    @staticmethod
    def _needs_followup(result: Dict[str, Any]) -> bool:
        questions = [
            str(item) for item in result.get("questions", []) if str(item).strip()
        ]
        return result.get("status") in {"need_more_info", "ambiguous"} and bool(questions)

    @staticmethod
    def _explicitly_missing_subject(query: str) -> bool:
        """Recognize an explicit omission instead of fuzzily inventing a disease.

        This is production dialogue behavior, not a Teacher-only override: when
        the user says the disease name has not been provided, retrieval must not
        turn generic words such as “疾病” into a concrete fuzzy match.
        """
        return bool(re.search(
            r"(?:还没|没有|暂时没|尚未)(?:说|告诉|提供|提到).{0,8}"
            r"(?:具体)?(?:疾病|病|药物|药品)?(?:的)?名称|"
            r"没说具体(?:是哪种|什么)?(?:疾病|病|药物|药品)",
            query,
        ))

    @staticmethod
    def _index_path() -> Path:
        configured = os.environ.get("MEDICAL_INDEX_PATH", "").strip()
        return Path(configured).expanduser().resolve() if configured else _DEFAULT_INDEX_PATH.resolve()

    def _dense_enabled(self) -> bool:
        if self._dense_enabled_override is not None:
            return self._dense_enabled_override
        return os.environ.get("MEDICAL_DENSE_ENABLED", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @staticmethod
    def _dense_manifest_path() -> Path:
        configured = os.environ.get("MEDICAL_DENSE_MANIFEST", "").strip()
        return (
            Path(configured).expanduser().resolve()
            if configured
            else _DEFAULT_DENSE_MANIFEST.resolve()
        )

    def _retriever(self, path: Path) -> MedicalRetriever:
        dense_enabled = self._dense_enabled()
        manifest_path = self._dense_manifest_path()
        endpoint = os.environ.get("MEDICAL_EMBEDDING_URL", "http://127.0.0.1:8006").strip()
        model = os.environ.get("MEDICAL_EMBEDDING_MODEL", "qwen3-embedding-0.6b").strip()
        dimensions = int(os.environ.get("MEDICAL_DENSE_DIMENSIONS", "256"))
        timeout = float(os.environ.get("MEDICAL_EMBEDDING_TIMEOUT", "5"))
        top_k = int(os.environ.get("MEDICAL_DENSE_TOP_K", "30"))
        prefix = os.environ.get("MEDICAL_DENSE_QUERY_PREFIX", "")
        signature: Tuple[Any, ...] = (
            path,
            path.stat().st_mtime_ns,
            dense_enabled,
            manifest_path,
            manifest_path.stat().st_mtime_ns if dense_enabled and manifest_path.is_file() else None,
            endpoint,
            model,
            dimensions,
            timeout,
            top_k,
            prefix,
        )
        with self._cache_lock:
            if self._cached and self._cached[0] == signature:
                return self._cached[1]
            dense_retriever = None
            if dense_enabled:
                try:
                    dense_index = DenseDocumentIndex(
                        manifest_path,
                        expected_model=model,
                        expected_dimensions=dimensions,
                        max_concurrent_scans=int(
                            os.environ.get("MEDICAL_DENSE_MAX_CONCURRENT_SCANS", "1")
                        ),
                    )
                    if os.environ.get("MEDICAL_DENSE_PREWARM", "1").strip().lower() not in {
                        "0",
                        "false",
                        "no",
                        "off",
                    }:
                        warm_ms = dense_index.warm()
                        logger.info("医疗 dense 索引预热完成: %.1f ms", warm_ms)
                    dense_retriever = DenseRetriever(
                        EmbeddingClient(
                            endpoint=endpoint,
                            model=model,
                            dimensions=dimensions,
                            timeout=timeout,
                            query_prefix=prefix,
                            require_bf16=True,
                        ),
                        dense_index,
                    )
                except Exception as exc:
                    logger.warning("医疗 dense 索引不可用，将使用 sparse 降级: %s", exc)
            retriever = MedicalRetriever(path, dense_retriever=dense_retriever, dense_top_k=top_k)
            self._cached = (signature, retriever)
            return retriever

    async def call(self, params: Dict[str, str], context: ToolContext) -> ToolResult:
        total_started = time.perf_counter()
        del context  # The local read-only lookup does not require session state.
        query = str(params.get("query", "") or "").strip()
        validated_at = time.perf_counter()

        def timing_diagnostics(
            *, prepare_ms: float = 0.0, consult_ms: float = 0.0,
            retrieval: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            retrieval = retrieval if isinstance(retrieval, dict) else {}
            retrieval_timings = {
                key: retrieval[key]
                for key in (
                    "edge_ms", "fact_ms", "department_ms", "sparse_ms",
                    "dense_ms", "parallel_total_ms",
                )
                if isinstance(retrieval.get(key), (int, float))
            }
            return {
                "schema_version": "medical-consult-timing.v1",
                "total_ms": round((time.perf_counter() - total_started) * 1000, 3),
                "stages": {
                    "input_validation_ms": round(
                        (validated_at - total_started) * 1000, 3
                    ),
                    "retriever_prepare_ms": round(prepare_ms, 3),
                    "consult_ms": round(consult_ms, 3),
                },
                "retrieval": {
                    "mode": str(retrieval.get("mode", "")),
                    "dense_used": bool(retrieval.get("dense_used", False)),
                    **retrieval_timings,
                },
            }
        if not query:
            return ToolResult(
                success=False,
                error="medical_consult 缺少必填参数 query",
                data="请把用户的原始医疗问题放入 query。",
                diagnostics=timing_diagnostics(),
            )

        if self._explicitly_missing_subject(query):
            result = {
                "status": "need_more_info",
                "query": query,
                "intent": "clarification",
                "positive_symptoms": [],
                "negative_symptoms": [],
                "normalized_terms": [],
                "red_flags": [],
                "urgency": "routine",
                "recommended_destination": "",
                "departments": [],
                "medication_allowed": False,
                "medication_notice": "",
                "questions": ["请问您具体想咨询哪种疾病或药物？"],
                "associations": [],
                "evidence": [],
                "retrieval": {"mode": "not_run", "reason": "subject_explicitly_missing"},
                "message": "缺少明确咨询对象，请先补充疾病或药物名称。",
            }
            return ToolResult(
                success=True,
                data=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                facts={
                    "medical.consultation": result,
                    "dialogue.followup_required": True,
                    "dialogue.followup_questions": result["questions"],
                },
                diagnostics=timing_diagnostics(retrieval=result["retrieval"]),
            )

        path = self._index_path()
        if not path.is_file():
            return ToolResult(
                success=False,
                error=f"医疗索引不存在: {path}",
                data=f"请先构建索引：{_BUILD_COMMAND}",
                diagnostics=timing_diagnostics(),
            )

        try:
            prepare_started = time.perf_counter()
            retriever = await asyncio.to_thread(self._retriever, path)
            prepare_ms = (time.perf_counter() - prepare_started) * 1000
            consult_started = time.perf_counter()
            result = await asyncio.to_thread(retriever.consult, query)
            consult_ms = (time.perf_counter() - consult_started) * 1000
            diagnostics = timing_diagnostics(
                prepare_ms=prepare_ms,
                consult_ms=consult_ms,
                retrieval=result.get("retrieval"),
            )
            if result.get("status") == "not_found":
                return ToolResult(
                    success=False,
                    data=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                    error="医疗知识库未返回可用证据",
                    error_type="NOT_FOUND",
                    empty=True,
                    retryable=True,
                    recovery_hint="请保留症状要点并换一种简短句式查询；不要重复原参数。",
                    diagnostics=diagnostics,
                )
            questions = [str(item) for item in result.get("questions", []) if str(item).strip()]
            needs_followup = self._needs_followup(result)
            return ToolResult(
                success=True,
                data=json.dumps(result, ensure_ascii=False, separators=(",", ":")),
                facts={
                    "medical.consultation": result,
                    "dialogue.followup_required": needs_followup,
                    "dialogue.followup_questions": questions[:3],
                },
                diagnostics=diagnostics,
            )
        except (FileNotFoundError, ValueError, sqlite3.DatabaseError) as exc:
            # Imported lazily below to keep module-level dependencies minimal.
            logger.warning("医疗索引不可用: %s", exc)
            return ToolResult(
                success=False,
                error=f"医疗索引不可用: {exc}",
                data=f"可尝试重新构建索引：{_BUILD_COMMAND}",
                diagnostics=timing_diagnostics(),
            )
        except Exception as exc:  # pragma: no cover - manager-level resilience
            logger.exception("medical_consult 查询失败")
            return ToolResult(
                success=False, error=f"医疗查询失败: {exc}", data="",
                diagnostics=timing_diagnostics(),
            )

__all__ = ["MedicalConsultTool"]
