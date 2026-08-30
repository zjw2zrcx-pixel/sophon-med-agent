"""Small-model friendly, read-only medical knowledge retrieval runtime.

The runtime deliberately accepts a natural-language question instead of exposing
the graph database's low-level query primitives.  It is conservative by design:
symptom-only medication requests never return medication evidence, and broad
graph traversals are capped before data is materialised.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
from pathlib import Path
import re
import sqlite3
import time
import unicodedata
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import quote

from .dense import DenseRetriever, reciprocal_rank_fusion


_SPACE_RE = re.compile(r"\s+")
_NON_TERM_RE = re.compile(r"[^0-9a-z\u3400-\u9fff]+")
logger = logging.getLogger(__name__)
_RETRIEVAL_EXECUTOR = ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="medical-retrieval"
)


def _normalise(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).lower()
    return _NON_TERM_RE.sub("", text)


def _short_text(value: object, limit: int) -> str:
    text = _SPACE_RE.sub(" ", str(value or "")).strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + "…"


def _char_ngrams(text: str) -> set[str]:
    text = _normalise(text)
    if not text:
        return set()
    grams = set(text)
    grams.update(text[index : index + 2] for index in range(len(text) - 1))
    return grams


def _like_clause(column: str, values: Sequence[str]) -> Tuple[str, List[str]]:
    if not values:
        return "", []
    return (
        "(" + " OR ".join(f"{column} LIKE ?" for _ in values) + ")",
        [f"%{value}%" for value in values],
    )


class MedicalRetriever:
    """Retrieve compact, sourced medical evidence from a SQLite index.

    Connections are intentionally short lived.  This makes a retriever instance
    safe to share between calls dispatched through ``asyncio.to_thread`` without
    disabling sqlite's thread checks or serialising all readers behind a lock.
    """

    # The graph has a small, closed relation vocabulary.  These are exact
    # allow-lists, not fuzzy ranking hints: an intent must never be padded with
    # another relation merely because the adjacent entity has a plausible type.
    _INTENT_RELATIONS: Mapping[str, Tuple[str, ...]] = {
        "symptoms": ("症状",),
        "causes": (),  # Causes live in structured facts/documents, not graph edges.
        "risk_factors": (),
        "prevention": (),
        "checks": ("诊断检查",),
        "diet": ("推荐食谱", "宜吃", "忌吃"),
        "surgery": (),  # Exact surgery evidence lives in structured facts/documents.
        "treatment": ("治疗方法",),
        "medication": ("常用药品",),
        "complications": ("并发症",),
        "overview": (),
        "department_recommendation": ("所属科室",),
        "symptom_consultation": (),
    }

    _INTENT_ASPECTS: Mapping[str, Tuple[str, ...]] = {
        "symptoms": ("症状", "临床表现"),
        "causes": ("病因", "原因", "为什么", "引起", "导致", "发病机制", "遗传因素", "传播途径"),
        "risk_factors": ("风险因素", "危险因素", "高危因素", "风险评估因素"),
        "prevention": ("预防", "预防措施", "如何预防", "怎样预防", "防止", "避免"),
        "checks": ("检查", "检查结果", "检验", "诊断", "化验", "复查", "随访", "b超", "超声"),
        "diet": ("宜食", "忌食", "饮食", "忌口", "食物", "吃什么"),
        "surgery": ("手术治疗", "手术方案", "手术方法", "手术", "术后"),
        "treatment": ("治疗", "怎么办", "方法", "辅助治疗", "放射治疗", "化疗", "缓解", "处理", "坐浴"),
        "medication": ("推荐药", "常用药", "药物", "用药"),
        "complications": ("并发症", "并发", "危害", "后果", "严重", "影响", "会怎么样"),
        "overview": (),
        "department_recommendation": ("所属科室", "就诊科室", "挂号", "科室"),
        "symptom_consultation": (),
    }

    _FACT_ASPECTS: Mapping[str, Tuple[str, ...]] = {
        "symptoms": ("症状",),
        "causes": ("病因", "发病机制", "遗传因素", "传播途径"),
        "risk_factors": ("风险因素",),
        "prevention": ("预防",),
        "checks": ("检查", "检查结果"),
        "diet": ("宜食", "忌食"),
        "surgery": ("手术治疗",),
        "treatment": ("治疗", "辅助治疗", "放射治疗", "化疗"),
        "medication": ("推荐药",),
        "complications": ("并发症",),
        "overview": (),
        "department_recommendation": ("就诊科室",),
        "symptom_consultation": (),
    }

    _GENERIC_ENTITY_NAMES = frozenset({
        "手术", "治疗", "检查", "疼痛", "医生", "门诊", "挂号", "药物", "疾病",
    })

    _FACT_NOISE_TERMS = (
        "季节指数", "领悟社会支持量表", "社会支持评定量表", "arima",
        "驻站医", "量子共振检测", "肾移植",
    )

    _RED_FLAG_RULES: Tuple[Tuple[str, re.Pattern[str], str], ...] = (
        (
            "breathing_difficulty",
            re.compile(r"呼吸困难|喘不过气|不能呼吸|嘴唇发紫|口唇发紫"),
            "呼吸困难或发绀需要立即评估。",
        ),
        (
            "chest_pain",
            re.compile(r"胸痛|胸口(?:压榨|剧痛)|胸闷.*(?:出汗|呼吸困难)"),
            "胸痛可能涉及心肺急症，请尽快急诊评估。",
        ),
        (
            "bleeding",
            re.compile(r"呕血|吐血|咯血|便血|黑便|大量出血|止不住血"),
            "明显消化道或其他部位出血需要尽快急诊处理。",
        ),
        (
            "altered_consciousness",
            re.compile(r"意识不清|神志不清|昏迷|叫不醒|抽搐|晕厥"),
            "意识改变、晕厥或抽搐属于紧急危险信号。",
        ),
        (
            "stroke_sign",
            re.compile(r"口角歪|一侧(?:无力|麻木)|说话不清|言语不清|突然看不清"),
            "疑似卒中表现需要立即联系急救服务。",
        ),
        (
            "sudden_severe_pain",
            re.compile(r"(?:突然|突发|骤然).{0,6}(?:剧烈|难忍|最严重).{0,4}(?:痛|疼)|剧烈腹痛"),
            "突发或难以忍受的剧痛需要尽快急诊评估。",
        ),
        (
            "pregnancy_abdominal_pain",
            re.compile(r"(?:怀孕|孕期|妊娠).{0,10}(?:腹痛|肚子痛|出血)|(?:腹痛|肚子痛).{0,10}(?:怀孕|孕期|妊娠)"),
            "孕期腹痛或出血需要尽快由产科评估。",
        ),
        (
            "severe_allergy",
            re.compile(r"(?:过敏|皮疹).{0,8}(?:喉咙紧|呼吸困难|脸肿|舌头肿)|喉头水肿"),
            "可能的严重过敏反应需要立即联系急救服务。",
        ),
    )

    _SYMPTOM_TERMS: Tuple[str, ...] = (
        "胸痛", "胸闷", "胸口闷", "呼吸困难", "喘不过气", "冷汗", "出汗",
        "左臂麻木", "左胳膊麻", "左胳膊有点麻", "下颌不适", "头痛", "头晕", "发热",
        "发烧", "高烧", "高热", "咳嗽", "流鼻涕", "腹痛", "肚子痛",
        "呕吐", "恶心", "腹泻", "皮疹", "红疹", "麻木", "乏力", "心慌", "出血",
    )

    _NEGATION_PREFIX = re.compile(r"(?:没有|没出现|未出现|并无|否认|无|不)(?:明显)?$")

    def __init__(
        self,
        index_path: str | Path,
        dense_retriever: Optional[DenseRetriever] = None,
        dense_top_k: int = 30,
    ):
        self.index_path = Path(index_path).expanduser().resolve()
        if not self.index_path.is_file():
            raise FileNotFoundError(f"医疗索引不存在: {self.index_path}")

        self._entities: Dict[Any, Dict[str, Any]] = {}
        self._aliases: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._aliases_by_first: DefaultDict[str, List[str]] = defaultdict(list)
        self._ngram_index: DefaultDict[str, set[str]] = defaultdict(set)
        self._alias_trie: Dict[str, Any] = {}
        self._dense_retriever = dense_retriever
        self._dense_top_k = max(1, dense_top_k)
        self._load_lookup_data()

    def _connect(self) -> sqlite3.Connection:
        encoded = quote(str(self.index_path), safe="/:")
        connection = sqlite3.connect(
            f"file:{encoded}?mode=ro", uri=True, timeout=3.0
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only=ON")
        return connection

    def _load_lookup_data(self) -> None:
        with self._connect() as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
                )
            }
            required = {"entities", "aliases", "edges", "facts", "documents", "document_fts"}
            missing = sorted(required - tables)
            if missing:
                raise ValueError("医疗索引缺少数据表: " + ", ".join(missing))

            if self._dense_retriever is not None:
                document_stats = connection.execute(
                    "SELECT COUNT(*), MIN(id), MAX(id) FROM documents"
                ).fetchone()
                dense_index = self._dense_retriever.index
                expected_count = int(dense_index.manifest["mapped_document_ids"])
                mapping_min = int(dense_index.document_ids.min())
                mapping_max = int(dense_index.document_ids.max())
                observed = (
                    int(document_stats[0]),
                    int(document_stats[1]),
                    int(document_stats[2]),
                )
                expected = (expected_count, mapping_min, mapping_max)
                if observed != expected:
                    logger.warning(
                        "dense document mapping does not match SQLite %s != %s; disabling dense",
                        expected,
                        observed,
                    )
                    self._dense_retriever = None

            for row in connection.execute(
                "SELECT id, name, label, properties_json, source FROM entities"
            ):
                self._entities[row["id"]] = {
                    "id": row["id"],
                    "name": _short_text(row["name"], 80),
                    "label": _short_text(row["label"], 40),
                    "properties_json": row["properties_json"] or "{}",
                    "source": _short_text(row["source"], 100),
                }

            # Canonical entity names participate in exact matching even if the
            # builder did not duplicate them in the aliases table.
            for entity in self._entities.values():
                self._add_alias(entity["name"], entity["id"], 1.0, "entity_name")

            for row in connection.execute(
                "SELECT alias, entity_id, weight, source FROM aliases"
            ):
                if row["entity_id"] not in self._entities:
                    continue
                try:
                    weight = float(row["weight"] if row["weight"] is not None else 0.8)
                except (TypeError, ValueError):
                    weight = 0.8
                self._add_alias(row["alias"], row["entity_id"], weight, row["source"])

        for first, aliases in self._aliases_by_first.items():
            # Longest-match-first is the core protection against mapping "腹痛"
            # as two unrelated one-character entities.
            aliases.sort(key=lambda value: (-len(value), value))

        for alias in self._aliases:
            node = self._alias_trie
            for character in alias:
                node = node.setdefault(character, {})
            node[""] = alias
            for gram in _char_ngrams(alias):
                self._ngram_index[gram].add(alias)

    def _add_alias(self, alias: object, entity_id: Any, weight: float, source: object) -> None:
        normalised = _normalise(alias)
        if not normalised:
            return
        item = {
            "entity_id": entity_id,
            "weight": max(0.0, min(float(weight), 1.0)),
            "source": _short_text(source, 80),
        }
        existing = self._aliases[normalised]
        if any(entry["entity_id"] == entity_id for entry in existing):
            for entry in existing:
                if entry["entity_id"] == entity_id and item["weight"] > entry["weight"]:
                    entry.update(item)
            return
        if not existing:
            self._aliases_by_first[normalised[0]].append(normalised)
        existing.append(item)
        existing.sort(key=lambda entry: entry["weight"], reverse=True)

    @staticmethod
    def _is_disease_label(label: object) -> bool:
        value = _normalise(label)
        return "疾病" in value or value in {"disease", "illness", "disorder"}

    @staticmethod
    def _is_symptom_label(label: object) -> bool:
        value = _normalise(label)
        return "症状" in value or value in {"symptom", "sign"}

    def _exact_matches(self, raw_query: str) -> List[Dict[str, Any]]:
        query = _normalise(raw_query)
        matches: List[Dict[str, Any]] = []
        seen_entities: set[Any] = set()
        position = 0
        while position < len(query):
            chosen = ""
            node = self._alias_trie
            cursor = position
            while cursor < len(query) and query[cursor] in node:
                node = node[query[cursor]]
                cursor += 1
                terminal = node.get("")
                if terminal and (len(terminal) > 1 or query == terminal):
                    chosen = terminal
            if not chosen:
                position += 1
                continue

            # Keep at most two interpretations for an ambiguous surface form.
            for alias_item in self._aliases[chosen][:2]:
                entity_id = alias_item["entity_id"]
                if entity_id in seen_entities:
                    continue
                entity = self._entities[entity_id]
                matches.append(
                    {
                        **entity,
                        "surface": chosen,
                        "confidence": alias_item["weight"],
                        "match": "exact",
                    }
                )
                seen_entities.add(entity_id)
            position += len(chosen)
            if len(matches) >= 8:
                break
        return matches

    @staticmethod
    def _fuzzy_core(raw_query: str) -> str:
        value = _normalise(raw_query)
        for phrase in (
            "麻烦你", "帮我查一下", "帮我看看", "我想问一下", "我想问", "请问",
            "医生", "怎么办", "怎么回事", "是什么", "有哪些", "能不能", "可以吗",
            "最近", "这几天", "今天", "现在", "一下", "请", "吗", "呢", "啊", "呀",
        ):
            value = value.replace(phrase, "")
        return value[:80]

    @staticmethod
    def _window_similarity(query: str, alias: str) -> float:
        if not query or not alias:
            return 0.0
        if alias in query:
            return 0.92
        candidate_lengths = range(max(1, len(alias) - 1), len(alias) + 2)
        best = 0.0
        for size in candidate_lengths:
            if size > len(query):
                windows = (query,)
            else:
                windows = (query[index : index + size] for index in range(len(query) - size + 1))
            for window in windows:
                left = _char_ngrams(window)
                right = _char_ngrams(alias)
                if not left or not right:
                    continue
                dice = 2.0 * len(left & right) / (len(left) + len(right))
                length_penalty = min(len(window), len(alias)) / max(len(window), len(alias))
                best = max(best, dice * (0.85 + 0.15 * length_penalty))
        return best

    def _fuzzy_matches(self, raw_query: str) -> List[Dict[str, Any]]:
        query = self._fuzzy_core(raw_query)
        if not query:
            return []

        # Bigrams are far more discriminative for Chinese.  Unigrams are used
        # only when no bigram can retrieve a candidate (e.g. a one-character term).
        bigrams = {query[index : index + 2] for index in range(len(query) - 1)}
        votes: Counter[str] = Counter()
        for gram in bigrams:
            votes.update(self._ngram_index.get(gram, ()))
        if not votes:
            for gram in set(query):
                votes.update(self._ngram_index.get(gram, ()))

        ranked_aliases: List[Tuple[float, int, str]] = []
        for alias, vote_count in votes.most_common(800):
            score = self._window_similarity(query, alias)
            if score >= 0.56:
                ranked_aliases.append((score, vote_count, alias))
        ranked_aliases.sort(key=lambda item: (-item[0], -item[1], len(item[2]), item[2]))

        results: List[Dict[str, Any]] = []
        seen_entities: set[Any] = set()
        if not ranked_aliases:
            return results
        best_score = ranked_aliases[0][0]
        for score, _, alias in ranked_aliases:
            if score < best_score - 0.08:
                break
            for alias_item in self._aliases[alias][:1]:
                entity_id = alias_item["entity_id"]
                if entity_id in seen_entities:
                    continue
                entity = self._entities[entity_id]
                results.append(
                    {
                        **entity,
                        "surface": alias,
                        "confidence": round(min(0.89, score * alias_item["weight"]), 3),
                        "match": "fuzzy",
                    }
                )
                seen_entities.add(entity_id)
                if len(results) >= 3:
                    return results
        return results

    @staticmethod
    def _medical_subquery(raw_query: str) -> Tuple[str, Optional[str]]:
        """Remove an explicit navigation side-task before medical retrieval."""
        navigation = re.search(
            r"(?:[，,。；;]\s*)?(?:另外|顺便|然后|再)?(?:请)?"
            r"(?:带我|领我|引导我|陪我|送我|导航|带路).{0,24}$",
            raw_query,
            re.IGNORECASE,
        )
        if navigation and navigation.start() > 0:
            cleaned = raw_query[: navigation.start()].rstrip("，,。；; ")
            if cleaned:
                return cleaned, _short_text(raw_query[navigation.start():], 80)
        return raw_query, None

    @staticmethod
    def _non_medical_reason(raw_query: str) -> str:
        """Recognise high-precision requests that must not enter medical fuzzy search."""
        value = _normalise(raw_query)
        if (
            re.search(r"(?:买|花|拿|给).{0,12}(?:元|块).{0,16}(?:找|剩).{0,5}(?:多少|几)", value)
            or re.search(r"\d+(?:元|块).{0,12}\d+(?:元|块).{0,12}(?:找|剩)", value)
        ):
            return "arithmetic_request"
        if "水" in value and "大气压" in value and any(
            term in value for term in ("沸腾", "沸点", "摄氏度")
        ):
            return "general_knowledge_request"
        if re.search(
            r"(?:怎么走|在哪里|在哪儿|什么地方|如何去|带我去|领我去|导航到)",
            value,
        ) and any(
            term in value
            for term in ("抽血", "检验科", "门诊", "挂号", "大厅", "医院", "科室", "药房")
        ):
            return "navigation_request"
        return ""

    @classmethod
    def _focus_matches(
        cls, matches: Sequence[Mapping[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Keep clinical anchors and suppress duplicate/generic graph entities."""
        if not matches:
            return []
        exact = [dict(match) for match in matches if match.get("match") == "exact"]
        candidates = exact or [
            dict(match) for match in matches if float(match.get("confidence", 0)) >= 0.78
        ]
        candidates = [
            match for match in candidates
            if str(match.get("name", "")) not in cls._GENERIC_ENTITY_NAMES
        ]

        # One surface may be duplicated as both a disease and a symptom.  For a
        # named-disease query, prefer the disease interpretation deterministically.
        candidates.sort(key=lambda match: (
            0 if cls._is_disease_label(match.get("label")) else 1,
            -len(str(match.get("surface", ""))),
            -float(match.get("confidence", 0)),
        ))
        focused: List[Dict[str, Any]] = []
        seen_surfaces: set[str] = set()
        seen_ids: set[Any] = set()
        for match in candidates:
            surface = _normalise(match.get("surface"))
            if surface in seen_surfaces or match.get("id") in seen_ids:
                continue
            seen_surfaces.add(surface)
            seen_ids.add(match.get("id"))
            focused.append(match)
        return focused[:5]

    def _detect_intent(self, raw_query: str, matches: Sequence[Mapping[str, Any]]) -> str:
        query = _normalise(raw_query)
        # High-confidence question structure wins over keywords embedded in an
        # entity name, e.g. “新诊断ITP通常如何治疗” is treatment rather than
        # checks, and “阿片类药物成瘾有哪些症状” is not a medication request.
        explicit_question_patterns: Tuple[Tuple[str, str], ...] = (
            ("surgery", r"手术治疗(?:方案|方法)?(?:有吗|吗|是什么|有哪些)?$"),
            ("prevention", r"^(?:如何|怎么|怎样)预防"),
            ("risk_factors", r"(?:有)?哪些(?:风险|危险|高危)因素$"),
            ("complications", r"(?:可能)?(?:有)?哪些并发症$"),
            ("causes", r"(?:的)?病因(?:是什么|有哪些)?$"),
            ("symptoms", r"(?:通常)?(?:有)?哪些症状$"),
            ("checks", r"(?:需要|应该|一般需要)?(?:做)?哪些检查$"),
            ("treatment", r"(?:通常)?如何治疗$"),
        )
        for explicit_intent, pattern in explicit_question_patterns:
            if re.search(pattern, query):
                return explicit_intent

        medication_keywords = (
            "吃什么药", "用什么药", "用药", "药物", "药品", "服什么药", "开药",
        )
        medication_pattern = re.compile(
            r"(?:吃|服|用).{0,6}药|药.{0,4}(?:剂量|几片|几粒|怎么吃|怎么服)"
        )
        if any(keyword in query for keyword in medication_keywords) or medication_pattern.search(query):
            return "medication"

        department_pattern = re.compile(
            r"(?:挂|看|去|找|应该去|该去|建议去|适合去).{0,6}(?:什么|哪个|哪一个)?(?:科|科室)"
            r"|(?:挂什么科|挂哪科|哪科|哪个科|科室推荐|就诊科室)"
            r"|(?:还没|没有|未).{0,4}挂号.{0,8}(?:去哪|去哪里|怎么办)"
            r"|(?:去哪|去哪里).{0,5}(?:看|就诊|挂号)"
        )
        if department_pattern.search(query):
            return "department_recommendation"

        # Explicit aspect wording wins over graph labels.  Entity labels can be
        # ambiguous (the same surface is often both a disease and a symptom).
        if re.search(
            r"(?:能否|能不能|可以|可不可以|是否)?.{0,5}(?:检查|检验|化验|确诊|诊断|复查|随访|"
            r"b超|超声|ct|mri|核磁|肠镜).{0,8}(?:出来|看出|发现|查到|显示)?|"
            r"(?:检查|检验|化验|确诊|诊断|复查|随访).{0,8}(?:什么|哪些|多少钱|费用)",
            query,
        ):
            return "checks"

        # Resolve explicit requested aspects before generic treatment wording.
        # Disease names and source-derived subjects may themselves contain
        # words such as “治疗” or “术后”; those must not override the question's
        # final aspect (for example “……有哪些并发症”).
        if re.search(
            r"如何预防|怎么预防|怎样预防|预防(?:措施|方法|要点)|"
            r"如何防止|怎么防止|怎样防止|如何避免|怎么避免|怎样避免|"
            r"降低.{0,8}(?:发病|患病|复发)?风险",
            query,
        ):
            return "prevention"

        if re.search(r"并发症|后遗症|危害|长期.{0,6}会怎么样|会有什么后果|严重吗|影响.{0,8}(?:吗|么)", query):
            return "complications"

        if re.search(r"风险因素|危险因素|高危因素|风险评估因素|哪些人.{0,6}(?:容易|易得|易患)", query):
            return "risk_factors"

        if re.search(
            r"为什么|什么原因|哪些原因|病因|发病机制|怎么引起|如何引起|"
            r"由什么引起|引起.{1,20}的原因|导致.{1,20}的原因|危险因素|诱因",
            query,
        ):
            return "causes"

        if re.search(r"什么症状|哪些症状|有哪些症状|早期症状|临床表现|有什么表现|是什么样", query):
            return "symptoms"

        surgery_pattern = re.compile(
            r"手术治疗|手术方案|手术方法|需要.{0,4}(?:做|进行)什么手术|"
            r"(?:能否|能不能|是否|可以|可不可以).{0,4}手术|术后.{0,8}(?:处理|治疗|恢复)"
        )
        if surgery_pattern.search(query):
            return "surgery"

        treatment_pattern = re.compile(
            r"怎么办|怎么处理|怎样处理|怎么治|如何治|治疗(?:方案|方法)?|"
            r"怎样缓解|怎么缓解|如何缓解|能治好|治愈|"
            r"为什么(?:要|用|做).{0,10}(?:治疗|坐浴|热敷|药)"
        )
        if treatment_pattern.search(query):
            return "treatment"

        # A phrase such as "腹痛伴腹泻可能是什么病" may itself exist as a
        # symptom node.  It is still a symptom consultation, not a confirmed
        # disease/overview lookup, and must not fan out into arbitrary diseases.
        has_symptom = any(
            self._is_symptom_label(match.get("label")) for match in matches
        )
        has_disease = any(
            self._is_disease_label(match.get("label")) for match in matches
        )
        if has_symptom and not has_disease:
            return "symptom_consultation"

        rules: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
            ("diet", ("饮食", "忌口", "吃什么", "什么食物")),
            ("symptoms", ("症状", "表现")),
            ("overview", ("什么是", "什么叫", "介绍", "了解一下", "是什么病", "历史")),
        )
        for intent, keywords in rules:
            if any(keyword in query for keyword in keywords):
                return intent

        if any(self._is_symptom_label(match.get("label")) for match in matches):
            return "symptom_consultation"
        complaint_words = (
            "疼", "痛", "不舒服", "发烧", "发热", "咳", "吐", "恶心", "腹泻",
            "拉肚子", "头晕", "乏力", "出血", "皮疹", "麻木", "水肿", "心慌",
        )
        if any(word in query for word in complaint_words):
            return "symptom_consultation"
        return "overview"

    @classmethod
    def _term_is_negated(cls, raw_query: str, start: int) -> bool:
        """Return whether a symptom occurrence has a nearby explicit negation."""
        prefix = raw_query[max(0, start - 8) : start]
        # Do not let a preceding, already completed clause negate the next one.
        prefix = re.split(r"[，,。；;但却而]", prefix)[-1]
        return bool(cls._NEGATION_PREFIX.search(prefix))

    @classmethod
    def _symptom_polarity(
        cls, raw_query: str, matches: Sequence[Mapping[str, Any]] = ()
    ) -> Tuple[List[str], List[str]]:
        del matches  # polarity uses a conservative lexicon, not ambiguous graph labels
        terms = list(cls._SYMPTOM_TERMS)
        positive: List[str] = []
        negative: List[str] = []
        # Longest first prevents "麻木" from shadowing "左臂麻木".
        for term in sorted(terms, key=len, reverse=True):
            for occurrence in re.finditer(re.escape(term), raw_query):
                bucket = negative if cls._term_is_negated(raw_query, occurrence.start()) else positive
                if term not in bucket:
                    bucket.append(term)
        # If a longer expression was found, suppress its contained short alias.
        def compact(values: List[str]) -> List[str]:
            return [
                value for value in values
                if not any(value != other and value in other for other in values)
            ]
        negative = compact(negative)
        positive = [value for value in compact(positive) if value not in negative]
        return positive[:12], negative[:12]

    def _red_flags(
        self, raw_query: str, negative_symptoms: Sequence[str] = ()
    ) -> List[Dict[str, str]]:
        flags: List[Dict[str, str]] = []
        for code, pattern, advice in self._RED_FLAG_RULES:
            matched = pattern.search(raw_query)
            if matched and not self._term_is_negated(raw_query, matched.start()):
                flags.append(
                    {"code": code, "matched": _short_text(matched.group(0), 40), "advice": advice}
                )
            if len(flags) >= 5:
                break

        normalised = _normalise(raw_query)
        negative = {_normalise(item) for item in negative_symptoms}
        chest_match = re.search(r"胸(?:口|部)?.{0,2}(?:闷|痛|压榨)", normalised)
        chest = bool(
            chest_match
            and not self._term_is_negated(normalised, chest_match.start())
        )
        companion_patterns = (
            ("左上肢不适", re.compile(r"左(?:臂|胳膊|手臂).{0,5}(?:麻|痛|疼|酸)")),
            ("下颌不适", re.compile(r"下颌.{0,3}(?:麻|痛|疼|不适)")),
            ("呼吸困难", re.compile(r"呼吸困难|喘不过气|憋气")),
            ("冷汗", re.compile(r"冷汗|大汗")),
        )
        companions: List[str] = []
        for label, pattern in companion_patterns:
            companion_match = pattern.search(normalised)
            if companion_match and not self._term_is_negated(
                normalised, companion_match.start()
            ):
                companions.append(label)
        if chest and not ({"胸闷", "胸痛"} & negative) and companions:
            flags.append({
                "code": "cardiac_symptom_combination",
                "matched": _short_text("胸部不适+" + "+".join(companions[:2]), 40),
                "advice": "胸部不适伴上肢放射不适、呼吸困难或冷汗，需要立即急诊评估。",
            })

        pediatric = any(term in normalised for term in ("孩子", "儿童", "小孩", "宝宝", "婴儿"))
        prolonged_high_fever = bool(
            re.search(r"(?:高烧|高热).{0,8}(?:[三四五六七八九十两2-9]\s*天|多天|不退)", raw_query)
            or re.search(r"(?:[三四五六七八九十两2-9]\s*天|多天).{0,8}(?:高烧|高热)", raw_query)
        )
        rash = any(term in normalised for term in ("皮疹", "红疹", "起红点", "出疹"))
        if pediatric and prolonged_high_fever and rash and not ({"发热", "发烧", "皮疹"} & negative):
            flags.append({
                "code": "pediatric_persistent_fever_rash",
                "matched": "儿童持续高热伴皮疹",
                "advice": "儿童持续高热伴皮疹需要尽快由儿科或急诊当面评估。",
            })

        unique: List[Dict[str, str]] = []
        seen_codes: set[str] = set()
        for flag in flags:
            if flag["code"] not in seen_codes:
                seen_codes.add(flag["code"])
                unique.append(flag)
        return unique[:5]

    @staticmethod
    def _entity_ids(matches: Sequence[Mapping[str, Any]]) -> List[Any]:
        result: List[Any] = []
        for match in matches:
            if match["id"] not in result:
                result.append(match["id"])
        return result[:8]

    @classmethod
    def _medication_context_allowed(
        cls, raw_query: str, matches: Sequence[Mapping[str, Any]]
    ) -> bool:
        """Require a clinician-confirmed exact disease before drug evidence.

        A user saying "我感冒了，吃什么药" is self-reported context, not a
        prescription basis.  Returning graph medication facts for it can make
        unsafe antibiotics appear authoritative in an otherwise trusted tool
        result.  Require an explicit clinician-confirmation cue as well.
        """
        # Continuation prompts contain the *question* "是否已有医生明确诊断".
        # It is not user evidence.  For that structured form, only inspect
        # the actual follow-up answer.
        confirmation_text = str(raw_query or "")
        followup = re.search(r"补充回答：\s*(.+?)\s*$", confirmation_text, re.S)
        if followup:
            confirmation_text = followup.group(1)
        confirmed = bool(re.search(
            r"(?:医生|医师|医院|门诊).{0,12}(?:诊断|确诊|说是|判断)|(?:确诊|诊断为|医生开)",
            confirmation_text,
            flags=re.IGNORECASE,
        ))
        if not confirmed:
            return False
        symptom_surfaces = {
            str(match.get("surface", ""))
            for match in matches
            if match.get("match") == "exact" and cls._is_symptom_label(match.get("label"))
        }
        return any(
            match.get("match") == "exact"
            and cls._is_disease_label(match.get("label"))
            and str(match.get("surface", "")) not in symptom_surfaces
            for match in matches
        )

    def _edge_associations(
        self,
        connection: sqlite3.Connection,
        matches: Sequence[Mapping[str, Any]],
        intent: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        disease_matches = [
            match for match in matches
            if self._is_disease_label(match.get("label"))
            and str(match.get("name", "")) not in self._GENERIC_ENTITY_NAMES
        ]
        ids = self._entity_ids(disease_matches)
        if not ids:
            return []
        relation_terms = self._INTENT_RELATIONS[intent]
        if not relation_terms:
            return []
        placeholders = ",".join("?" for _ in ids)
        relation_placeholders = ",".join("?" for _ in relation_terms)
        sql = f"""
            SELECT e.src_id, e.dst_id, e.relation, e.source,
                   s.name AS src_name, s.label AS src_label,
                   d.name AS dst_name, d.label AS dst_label
            FROM edges AS e
            JOIN entities AS s ON s.id = e.src_id
            JOIN entities AS d ON d.id = e.dst_id
            WHERE e.src_id IN ({placeholders})
              AND e.relation IN ({relation_placeholders})
            LIMIT 50
        """
        params: List[Any] = [*ids, *relation_terms]
        rows = connection.execute(sql, params).fetchall()

        id_set = set(ids)
        ranked: List[Tuple[int, sqlite3.Row, bool]] = []
        for row in rows:
            outgoing = True
            other_label = row["dst_label"] if outgoing else row["src_label"]
            relation = str(row["relation"] or "")
            if intent == "symptom_consultation" and not self._is_disease_label(other_label):
                continue
            if intent == "medication" and not (
                "药" in _normalise(other_label) or "药" in relation
            ):
                continue
            if intent == "symptoms" and not (
                self._is_symptom_label(other_label) or "症状" in relation or "临床表现" in relation
            ):
                continue
            score = 0
            for index, term in enumerate(relation_terms):
                if term in relation:
                    score = max(score, 20 - index)
            ranked.append((score, row, outgoing))
        ranked.sort(key=lambda item: (-item[0], str(item[1]["relation"]), str(item[1]["dst_name"])))

        associations: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for _, row, outgoing in ranked:
            matched_name = row["src_name"] if outgoing else row["dst_name"]
            related_name = row["dst_name"] if outgoing else row["src_name"]
            related_label = row["dst_label"] if outgoing else row["src_label"]
            key = (str(matched_name), str(row["relation"]), str(related_name))
            if key in seen:
                continue
            seen.add(key)
            associations.append(
                {
                    "matched": _short_text(matched_name, 60),
                    "relation": _short_text(row["relation"], 50),
                    "related": _short_text(related_name, 80),
                    "related_type": _short_text(related_label, 30),
                    "direction": "outgoing" if outgoing else "incoming",
                    "source": _short_text(row["source"], 100),
                }
            )
            if len(associations) >= min(3, max(1, limit)):
                break
        return associations

    @staticmethod
    def _navigation_department_name(name: object) -> str:
        value = str(name or "").strip()
        return {
            "心内科": "心血管内科",
            "小儿内科": "儿科",
            "小儿外科": "儿科",
            "耳鼻咽喉科": "耳鼻喉科",
            "精神科": "精神心理科",
            "普外科": "普通外科",
        }.get(value, value)

    def _department_candidates(
        self,
        connection: sqlite3.Connection,
        matches: Sequence[Mapping[str, Any]],
        raw_query: str,
        limit: int = 3,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Return sourced routing candidates without presenting a diagnosis.

        Exact disease entities use their direct ``所属科室`` edge.  Symptom
        entities may use a bounded two-hop ``症状 -> 疾病 -> 所属科室`` lookup;
        only the department and the graph path are exposed, never a diagnosis.
        """
        disease_ids = [
            match["id"] for match in matches if self._is_disease_label(match.get("label"))
        ][:8]
        symptom_ids = [
            match["id"] for match in matches if self._is_symptom_label(match.get("label"))
        ][:8]
        rows: List[Mapping[str, Any]] = []
        if disease_ids:
            placeholders = ",".join("?" for _ in disease_ids)
            rows.extend(connection.execute(
                f"""
                SELECT s.name AS matched_name, s.name AS disease_name,
                       d.name AS department_name, e.source AS source,
                       'direct_disease' AS basis
                FROM edges e
                JOIN entities s ON s.id=e.src_id
                JOIN entities d ON d.id=e.dst_id
                WHERE e.src_id IN ({placeholders})
                  AND e.relation='所属科室'
                  AND (d.label LIKE '%科室%' OR d.label LIKE '%科%')
                LIMIT 30
                """,
                disease_ids,
            ).fetchall())

        if symptom_ids:
            placeholders = ",".join("?" for _ in symptom_ids)
            rows.extend(connection.execute(
                f"""
                WITH symptom_diseases AS (
                    SELECT sy.name AS matched_name, di.id AS disease_id,
                           di.name AS disease_name, se.source AS symptom_source
                    FROM edges se
                    JOIN entities sy ON sy.id=se.src_id
                    JOIN entities di ON di.id=se.dst_id
                    WHERE se.src_id IN ({placeholders}) AND se.relation LIKE '%症状%'
                      AND di.label LIKE '%疾病%'
                    UNION ALL
                    SELECT sy.name AS matched_name, di.id AS disease_id,
                           di.name AS disease_name, se.source AS symptom_source
                    FROM edges se
                    JOIN entities di ON di.id=se.src_id
                    JOIN entities sy ON sy.id=se.dst_id
                    WHERE se.dst_id IN ({placeholders}) AND se.relation LIKE '%症状%'
                      AND di.label LIKE '%疾病%'
                )
                SELECT sd.matched_name, sd.disease_name,
                       dep.name AS department_name, de.source AS source,
                       'symptom_two_hop' AS basis
                FROM symptom_diseases sd
                JOIN edges de ON de.src_id=sd.disease_id AND de.relation='所属科室'
                JOIN entities dep ON dep.id=de.dst_id
                WHERE dep.label LIKE '%科室%' OR dep.label LIKE '%科%'
                LIMIT 120
                """,
                [*symptom_ids, *symptom_ids],
            ).fetchall())

        pediatric = any(term in _normalise(raw_query) for term in ("孩子", "儿童", "小孩", "宝宝", "婴儿"))
        grouped: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            raw_name = str(row["department_name"] or "").strip()
            name = self._navigation_department_name(raw_name)
            if not name:
                continue
            if name == "急诊科":
                # Urgent routing is determined by the red-flag gate.  A broad
                # symptom graph path alone must not manufacture an emergency.
                continue
            item = grouped.setdefault(name, {
                "name": name,
                "canonical_name": raw_name,
                "basis": row["basis"],
                "support": 0,
                "matched_terms": [],
                "source": _short_text(row["source"], 100),
                "pediatric": False,
            })
            item["support"] += 1
            matched_name = _short_text(row["matched_name"], 50)
            if matched_name and matched_name not in item["matched_terms"]:
                item["matched_terms"].append(matched_name)
            if "小儿" in str(row["disease_name"]) or name == "儿科":
                item["pediatric"] = True
            if row["basis"] == "direct_disease":
                item["basis"] = "direct_disease"

        ranked = sorted(
            grouped.values(),
            key=lambda item: (
                0 if item["basis"] == "direct_disease" else 1,
                0 if pediatric and item["pediatric"] else 1,
                -int(item["support"]),
                item["name"],
            ),
        )
        departments: List[Dict[str, Any]] = []
        associations: List[Dict[str, Any]] = []
        for item in ranked[: max(1, limit)]:
            direct = item["basis"] == "direct_disease"
            confidence = 0.95 if direct else min(0.70, 0.52 + 0.03 * item["support"])
            departments.append({
                "name": item["name"],
                "canonical_name": item["canonical_name"],
                "confidence": round(confidence, 3),
                "basis": item["basis"],
                "matched_terms": item["matched_terms"][:3],
                "source": item["source"],
            })
            associations.append({
                "matched": "、".join(item["matched_terms"][:3]),
                "relation": "所属科室" if direct else "症状→疾病→所属科室",
                "related": item["name"],
                "related_type": "科室",
                "direction": "outgoing",
                "source": item["source"],
            })
        return departments, associations

    def _fact_evidence(
        self,
        connection: sqlite3.Connection,
        raw_query: str,
        matches: Sequence[Mapping[str, Any]],
        intent: str,
        limit: int = 3,
    ) -> List[Dict[str, Any]]:
        subjects = list(dict.fromkeys(
            str(match["name"])
            for match in matches
            if self._is_disease_label(match.get("label"))
            and str(match.get("name", "")) not in self._GENERIC_ENTITY_NAMES
        ))[:4]
        # Some imported graph entities lost a leading ASCII type marker while
        # the facts table retained it (for example entity “型糖尿病” versus fact
        # subject “1型糖尿病”). Recover only an adjacent alphanumeric prefix
        # explicitly present in the user's query; never fuzzy-expand to another
        # disease. Exact indexed subject lookup remains deterministic and cheap.
        normalized_query = _normalise(raw_query)
        aligned_subjects: List[str] = []
        for subject in subjects:
            normalized_subject = _normalise(subject)
            start = normalized_query.find(normalized_subject)
            if start <= 0:
                continue
            prefix_start = start
            while prefix_start > 0 and re.fullmatch(
                r"[0-9a-z]", normalized_query[prefix_start - 1]
            ):
                prefix_start -= 1
            if prefix_start < start:
                aligned = normalized_query[prefix_start : start] + str(subject)
                if aligned not in subjects and aligned not in aligned_subjects:
                    aligned_subjects.append(aligned)
        subjects = (aligned_subjects + subjects)[:8]
        if not subjects or intent == "symptom_consultation":
            return []
        placeholders = ",".join("?" for _ in subjects)
        aspect_terms = self._FACT_ASPECTS[intent]
        if not aspect_terms:
            return []
        aspect_placeholders = ",".join("?" for _ in aspect_terms)
        sql = f"""
            SELECT subject, aspect, answer, source, source_line, quality
            FROM facts
            WHERE subject IN ({placeholders})
              AND aspect IN ({aspect_placeholders})
            ORDER BY COALESCE(quality, 0) DESC, source_line ASC
            LIMIT 60
        """
        rows = connection.execute(sql, [*subjects, *aspect_terms]).fetchall()
        aspect_rank = {aspect: index for index, aspect in enumerate(aspect_terms)}
        subject_rank = {subject: index for index, subject in enumerate(subjects)}
        rows = sorted(rows, key=lambda row: (
            subject_rank.get(str(row["subject"]), len(subject_rank)),
            aspect_rank.get(str(row["aspect"]), len(aspect_rank)),
            -float(row["quality"] or 0),
            int(row["source_line"] or 0),
        ))
        evidence: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str, str]] = set()
        for row in rows:
            key = (str(row["subject"]), str(row["aspect"]), str(row["answer"]))
            if key in seen:
                continue
            seen.add(key)
            answer = self._clean_fact_answer(row["answer"])
            if not answer:
                continue
            item: Dict[str, Any] = {
                "type": "fact",
                "subject": _short_text(row["subject"], 70),
                "aspect": _short_text(row["aspect"], 40),
                "text": answer,
                "source": _short_text(row["source"], 100),
            }
            if row["source_line"] is not None:
                item["source_line"] = row["source_line"]
            evidence.append(item)
            if len(evidence) >= min(2, max(1, limit)):
                break
        return evidence

    @classmethod
    def _clean_fact_answer(cls, value: object) -> str:
        text = _SPACE_RE.sub(" ", str(value or "")).strip()
        if not text:
            return ""
        parts = [part.strip() for part in re.split(r"[；;]", text) if part.strip()]
        if len(parts) <= 1:
            lowered = text.lower()
            return "" if any(term in lowered for term in cls._FACT_NOISE_TERMS) else _short_text(text, 220)
        cleaned: List[str] = []
        seen: set[str] = set()
        for part in parts:
            normalised = _normalise(part)
            lowered = part.lower()
            if not normalised or normalised in seen:
                continue
            if any(term in lowered for term in cls._FACT_NOISE_TERMS):
                continue
            seen.add(normalised)
            cleaned.append(part)
            if len(cleaned) >= 8:
                break
        return _short_text("；".join(cleaned), 220) if cleaned else ""

    @staticmethod
    def _fts_terms(
        raw_query: str, matches: Sequence[Mapping[str, Any]], intent: str
    ) -> List[str]:
        seeds = [MedicalRetriever._fuzzy_core(raw_query)]
        seeds.extend(str(match["name"]) for match in matches[:4])
        aspect_terms = MedicalRetriever._INTENT_ASPECTS[intent]
        if aspect_terms:
            seeds.append(aspect_terms[0])
        tokens: List[str] = []
        for seed in seeds:
            value = _normalise(seed)
            if not value:
                continue
            if len(value) == 1:
                candidates: Iterable[str] = (value,)
            else:
                candidates = (value[index : index + 2] for index in range(len(value) - 1))
            for token in candidates:
                if token not in tokens:
                    tokens.append(token)
                if len(tokens) >= 12:
                    return tokens
        return tokens

    @staticmethod
    def _unsafe_document_question(question: object) -> bool:
        """Block standalone encyclopedia snippets that invite self-medication."""
        value = _normalise(question)
        return any(
            term in value
            for term in (
                "吃什么药",
                "用什么药",
                "如何用药",
                "怎么用药",
                "服用什么",
                "药物剂量",
                "用药剂量",
                "一次几片",
                "一天几次",
                "降压药",
                "抗生素",
                "抗菌药",
                "处方药",
            )
        )

    def _document_evidence(
        self,
        connection: sqlite3.Connection,
        raw_query: str,
        matches: Sequence[Mapping[str, Any]],
        intent: str,
        limit: int = 2,
        dense_future: Optional[Future[Any]] = None,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        branch_started = time.perf_counter()
        retrieval: Dict[str, Any] = {
            "mode": "sparse",
            "dense_enabled": self._dense_retriever is not None,
            "dense_used": False,
        }
        if limit <= 0:
            return [], retrieval
        tokens = self._fts_terms(raw_query, matches, intent)
        sparse_rows: Sequence[sqlite3.Row] = ()
        if tokens:
            expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
            try:
                sparse_rows = connection.execute(
                    """
                    SELECT d.id, d.question, d.answer, d.source,
                           bm25(document_fts) AS rank
                    FROM document_fts
                    JOIN documents AS d ON d.rowid = document_fts.rowid
                    WHERE document_fts MATCH ?
                    ORDER BY rank
                    LIMIT 30
                    """,
                    (expression,),
                ).fetchall()
            except sqlite3.OperationalError:
                # A malformed/older FTS index may still use dense retrieval.
                sparse_rows = ()

        retrieval["sparse_ms"] = round(
            (time.perf_counter() - branch_started) * 1000, 3
        )

        sparse_ids = [int(row["id"]) for row in sparse_rows]
        dense_hits = ()
        if self._dense_retriever is not None:
            try:
                wait_started = time.perf_counter()
                dense_result = (
                    dense_future.result()
                    if dense_future is not None
                    else self._dense_retriever.search(raw_query, top_k=self._dense_top_k)
                )
                dense_hits = dense_result.hits
                retrieval.update(
                    {
                        "mode": "hybrid" if sparse_ids else "dense",
                        "dense_used": True,
                        "parallel": dense_future is not None,
                        "dense_wait_ms": round(
                            (time.perf_counter() - wait_started) * 1000, 3
                        ),
                        "embedding_ms": round(dense_result.embedding_ms, 3),
                        "dense_search_ms": round(dense_result.search_ms, 3),
                    }
                )
            except Exception as exc:
                # The medical tool must remain useful when the TPU service is
                # unavailable; sparse/structured retrieval is the safe fallback.
                logger.warning("dense medical retrieval unavailable, using sparse fallback: %s", exc)
                retrieval["fallback"] = "sparse"

        fused_ids = reciprocal_rank_fusion(sparse_ids, dense_hits)
        if not fused_ids:
            return [], retrieval
        fetch_ids = fused_ids[:60]
        placeholders = ",".join("?" for _ in fetch_ids)
        fetched = connection.execute(
            f"SELECT id, question, answer, source FROM documents WHERE id IN ({placeholders})",
            fetch_ids,
        ).fetchall()
        rows_by_id = {int(row["id"]): row for row in fetched}
        rows = [rows_by_id[document_id] for document_id in fused_ids if document_id in rows_by_id]
        aspect_terms = self._INTENT_ASPECTS[intent]
        disease_terms = list(dict.fromkeys(
            _normalise(match.get("name"))
            for match in matches
            if self._is_disease_label(match.get("label"))
            and str(match.get("name", "")) not in self._GENERIC_ENTITY_NAMES
        ))
        query_normalised = _normalise(raw_query)
        ranked_rows: List[Tuple[int, int, sqlite3.Row]] = []
        for base_rank, row in enumerate(rows):
            question = _normalise(row["question"])
            entity_hits = sum(bool(term and term in question) for term in disease_terms)
            aspect_hit = any(_normalise(term) in question for term in aspect_terms)
            direct_hit = bool(query_normalised and (
                query_normalised in question or question in query_normalised
            ))
            score = 200 * int(direct_hit) + 80 * entity_hits + 30 * int(aspect_hit) - base_rank
            ranked_rows.append((score, entity_hits, row))
        if disease_terms and any(entity_hits for _, entity_hits, _ in ranked_rows):
            ranked_rows = [item for item in ranked_rows if item[1] > 0]
        if aspect_terms:
            # Explicit aspect requests must never be padded with overview or a
            # neighboring aspect merely because no matching document exists.
            # An empty branch is a meaningful evidence gap for the caller.
            ranked_rows = [
                item for item in ranked_rows
                if any(_normalise(term) in _normalise(item[2]["question"]) for term in aspect_terms)
            ]
        ranked_rows.sort(key=lambda item: -item[0])
        rows = [item[2] for item in ranked_rows]
        retrieval["sparse_candidates"] = len(sparse_ids)
        retrieval["dense_candidates"] = len(dense_hits)
        retrieval["document_branch_ms"] = round(
            (time.perf_counter() - branch_started) * 1000, 3
        )

        evidence: List[Dict[str, Any]] = []
        seen: set[Tuple[str, str]] = set()
        for row in rows:
            if self._unsafe_document_question(row["question"]):
                continue
            key = (str(row["question"]), str(row["answer"]))
            if key in seen:
                continue
            seen.add(key)
            evidence.append(
                {
                    "type": "document",
                    "question": _short_text(row["question"], 100),
                    "text": _short_text(row["answer"], 220),
                    "source": _short_text(row["source"], 100),
                }
            )
            if len(evidence) >= min(2, max(1, limit)):
                break
        return evidence, retrieval

    def _edge_task(
        self, matches: Sequence[Mapping[str, Any]], intent: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        started = time.perf_counter()
        with self._connect() as connection:
            result = self._edge_associations(connection, matches, intent, limit=3)
        return result, (time.perf_counter() - started) * 1000

    def _fact_task(
        self, raw_query: str, matches: Sequence[Mapping[str, Any]], intent: str
    ) -> Tuple[List[Dict[str, Any]], float]:
        started = time.perf_counter()
        with self._connect() as connection:
            result = self._fact_evidence(
                connection, raw_query, matches, intent, limit=2
            )
        return result, (time.perf_counter() - started) * 1000

    def _department_task(
        self, matches: Sequence[Mapping[str, Any]], raw_query: str
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], float]:
        started = time.perf_counter()
        with self._connect() as connection:
            departments, associations = self._department_candidates(
                connection, matches, raw_query, limit=3
            )
        return departments, associations, (time.perf_counter() - started) * 1000

    @staticmethod
    def _clarifying_questions(
        raw_query: str,
        matches: Sequence[Mapping[str, Any]],
        intent: str,
        medication_allowed: bool,
        negative_symptoms: Sequence[str] = (),
    ) -> List[str]:
        if intent == "medication" and not medication_allowed:
            questions = [
                "是否已有医生明确诊断？请提供确诊疾病名称。",
                "症状持续多久、严重程度如何，是否正在加重？",
                "是否有药物过敏、孕期情况或正在使用其他药物？",
            ]
        else:
            del matches
            combined = _normalise(raw_query)
            for negated in negative_symptoms:
                combined = combined.replace(_normalise(negated), "")
                if negated == "发烧":
                    combined = combined.replace("发热", "")
                elif negated == "发热":
                    combined = combined.replace("发烧", "")
            if any(term in combined for term in ("腹痛", "肚子痛", "胃痛")):
                questions = [
                "疼痛位于上腹、下腹、右下腹还是其他位置？",
                "是突然剧痛、阵发绞痛，还是持续隐痛？持续多久了？",
                "是否伴有发热、反复呕吐、便血、黑便或腹部僵硬？",
                ]
            elif "头痛" in combined:
                questions = [
                "头痛是突然出现还是逐渐出现，持续多久、程度如何？",
                "疼痛位置在哪里，是否伴有发热、呕吐、视物异常或肢体无力？",
                "近期是否有头部外伤、血压明显升高或使用新药？",
                ]
            elif any(term in combined for term in ("发热", "发烧")):
                questions = [
                "最高体温是多少，发热持续多久？",
                "是否伴有咳嗽、皮疹、颈部僵硬、呼吸困难或持续呕吐？",
                "患者年龄及是否有孕期、免疫低下或严重基础病情况？",
                ]
            else:
                questions = [
                    "症状从何时开始，突然出现还是逐渐出现？",
                    "具体部位、严重程度和变化趋势如何？",
                    "还伴有哪些症状，是否有基础病、过敏或正在用药？",
                ]
        negated_terms = set(negative_symptoms)
        if "发烧" in negated_terms:
            negated_terms.add("发热")
        if "发热" in negated_terms:
            negated_terms.add("发烧")
        if "肚子痛" in negated_terms:
            negated_terms.add("腹痛")
        if "腹痛" in negated_terms:
            negated_terms.add("肚子痛")
        # Do not ask the user to reconfirm a symptom they explicitly denied.
        filtered = [
            question for question in questions
            if not any(term and term in question for term in negated_terms)
        ]
        return filtered or ["除已明确否认的情况外，还有哪些伴随症状？"]

    @staticmethod
    def _compact_result(result: Dict[str, Any], max_bytes: int = 5000) -> Dict[str, Any]:
        def size() -> int:
            return len(json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))

        if size() <= max_bytes:
            return result
        for evidence in result.get("evidence", ()):
            if "text" in evidence:
                evidence["text"] = _short_text(evidence["text"], 120)
        while size() > max_bytes and len(result.get("associations", ())) > 2:
            result["associations"].pop()
        while size() > max_bytes and len(result.get("evidence", ())) > 1:
            result["evidence"].pop()
        while size() > max_bytes and len(result.get("normalized_terms", ())) > 2:
            result["normalized_terms"].pop()
        return result

    def consult(self, raw_query: str) -> Dict[str, Any]:
        """Interpret one medical question and return a compact retrieval result."""
        original_query = _short_text(raw_query, 300)
        if not original_query:
            raise ValueError("query 不能为空")

        query, removed_side_task = self._medical_subquery(original_query)
        non_medical_reason = self._non_medical_reason(query)
        if non_medical_reason:
            return {
                "status": "out_of_scope",
                "query": original_query,
                "intent": "non_medical",
                "positive_symptoms": [],
                "negative_symptoms": [],
                "normalized_terms": [],
                "red_flags": [],
                "urgency": "routine",
                "recommended_destination": "",
                "departments": [],
                "medication_allowed": False,
                "medication_notice": "",
                "questions": [],
                "associations": [],
                "evidence": [],
                "retrieval": {"mode": "not_run", "reason": non_medical_reason},
                "message": "该问题不应由医疗知识库回答，请改用对应的通用、计算或导航工具。",
            }

        matches = self._focus_matches(self._exact_matches(query))
        if not matches:
            matches = self._focus_matches(self._fuzzy_matches(query))
        intent = self._detect_intent(query, matches)
        positive_symptoms, negative_symptoms = self._symptom_polarity(query, matches)
        flags = self._red_flags(query, negative_symptoms)
        ambiguous = (
            len(matches) > 1
            and matches[0].get("match") == "fuzzy"
            and float(matches[0].get("confidence", 0))
            - float(matches[1].get("confidence", 0)) < 0.05
        )
        medication_allowed = self._medication_context_allowed(query, matches)

        associations: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        departments: List[Dict[str, Any]] = []
        retrieval: Dict[str, Any] = {
            "mode": "not_run",
            "dense_enabled": self._dense_retriever is not None,
            "dense_used": False,
        }
        if removed_side_task:
            retrieval["medical_query"] = query
            retrieval["removed_side_task"] = removed_side_task
        # A symptom-only request for medication intentionally receives no drug
        # facts, document snippets, or graph edges.  The caller gets questions
        # needed to establish the clinical context instead.
        blocked_medication = intent == "medication" and not medication_allowed
        # Urgent and unsafe medication requests stop before retrieval.  Ordinary
        # symptom consultations now receive document-only hybrid evidence;
        # graph disease fan-out remains disabled to avoid implying a diagnosis.
        skip_retrieval = bool(flags) or blocked_medication
        if not skip_retrieval:
            parallel_started = time.perf_counter()
            structured_allowed = intent != "symptom_consultation"
            edge_future = (
                _RETRIEVAL_EXECUTOR.submit(self._edge_task, matches, intent)
                if structured_allowed and self._INTENT_RELATIONS[intent] else None
            )
            fact_future = (
                _RETRIEVAL_EXECUTOR.submit(self._fact_task, query, matches, intent)
                if structured_allowed and self._FACT_ASPECTS[intent] else None
            )
            department_future = (
                _RETRIEVAL_EXECUTOR.submit(self._department_task, matches, query)
                if intent == "department_recommendation" else None
            )
            dense_future = None
            if intent != "medication" and self._dense_retriever is not None:
                dense_future = _RETRIEVAL_EXECUTOR.submit(
                    self._dense_retriever.search, query, self._dense_top_k
                )

            # FTS runs in the caller thread with its own read-only connection,
            # concurrently with graph, facts and dense retrieval.
            if intent == "medication":
                documents = []
                retrieval["reason"] = "medication_document_block"
                retrieval["parallel"] = True
            else:
                with self._connect() as connection:
                    documents, retrieval = self._document_evidence(
                        connection,
                        query,
                        matches,
                        intent,
                        limit=2,
                        dense_future=dense_future,
                    )
            if edge_future is not None:
                associations, edge_ms = edge_future.result()
            else:
                edge_ms = 0.0
            if fact_future is not None:
                facts, fact_ms = fact_future.result()
            else:
                facts, fact_ms = [], 0.0
            if department_future is not None:
                departments, department_associations, department_ms = department_future.result()
                # The dedicated path is ranked for routing and replaces generic
                # department edges so the same fact is not repeated twice.
                associations = department_associations or associations
                retrieval["department_ms"] = round(department_ms, 3)
            retrieval.update(
                {
                    "edge_ms": round(edge_ms, 3),
                    "fact_ms": round(fact_ms, 3),
                    "parallel_total_ms": round(
                        (time.perf_counter() - parallel_started) * 1000, 3
                    ),
                }
            )
            evidence = (facts[:1] + documents[:2])[:3]
        elif flags:
            retrieval["reason"] = "red_flag"
        elif blocked_medication:
            retrieval["reason"] = "unsafe_medication_request"
        else:
            retrieval["reason"] = "not_run"

        if flags:
            status = "urgent"
            message = "检测到可能需要紧急处理的危险信号，请尽快就医或联系当地急救服务。"
        elif ambiguous:
            status = "ambiguous"
            message = "口语或识别结果可能对应多个医学术语，请先确认具体症状。"
        elif blocked_medication:
            status = "need_more_info"
            message = "仅凭症状不能安全确定病因或用药；请先补充问诊信息。关联疾病不代表诊断。"
        elif intent == "department_recommendation" and departments:
            status = "ok"
            message = "以下科室来自本地图谱的就诊关系，仅用于分诊参考，不代表已经确诊。"
        elif intent == "department_recommendation":
            status = "need_more_info"
            message = "现有资料不足以可靠推荐具体科室，请补充主要症状、持续时间和患者年龄。"
        elif intent == "symptom_consultation":
            status = "need_more_info"
            message = "检索结果仅用于辅助追问，不能据此确定病因；请补充问诊信息。"
        elif matches and not evidence and not associations and not departments:
            status = "evidence_gap"
            message = "本地资料识别到咨询对象，但没有覆盖当前所问方面。"
        elif not matches and not evidence:
            status = "not_found"
            message = "本地医疗资料未找到足够可靠的匹配，请补充规范疾病名或更具体的症状。"
        else:
            status = "ok"
            message = "以下为本地医疗资料的有限检索结果，仅供信息参考，不能替代医生诊断。"

        questions: List[str] = []
        if status == "ambiguous":
            candidates = "、".join(str(match["name"]) for match in matches[:3])
            questions = [f"你描述的是以下哪一种：{candidates}？"]
        elif status == "need_more_info":
            questions = self._clarifying_questions(
                query, matches, intent, medication_allowed, negative_symptoms
            )

        if removed_side_task:
            retrieval["medical_query"] = query
            retrieval["removed_side_task"] = removed_side_task

        result: Dict[str, Any] = {
            "status": status,
            "query": original_query,
            "intent": intent,
            "positive_symptoms": positive_symptoms,
            "negative_symptoms": negative_symptoms,
            "normalized_terms": [
                {
                    "surface": match["surface"],
                    "canonical": match["name"],
                    "type": match["label"],
                    "confidence": round(float(match["confidence"]), 3),
                    "match": match["match"],
                }
                for match in matches[:5]
            ],
            "red_flags": flags,
            "urgency": "emergency" if flags else "routine",
            "recommended_destination": "急诊科" if flags else "",
            "departments": departments,
            "medication_allowed": medication_allowed,
            "medication_notice": (
                "药物关系仅是资料关联，不代表适合当前用户；抗菌药、抗病毒药及处方药须由医生评估后使用。"
                if intent == "medication" and medication_allowed
                else ""
            ),
            "questions": questions[:3],
            "associations": associations[:3],
            "evidence": evidence[:3],
            "retrieval": retrieval,
            "message": message,
        }
        return self._compact_result(result)


__all__ = ["MedicalRetriever"]
