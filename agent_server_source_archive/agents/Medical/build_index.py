"""Build the local SQLite medical search index.

The builder deliberately keeps the provenance and roles of its inputs separate:
the Neo4j export supplies graph facts, Huatuo KG train supplies supplementary
structured facts, Huatuo encyclopedia train supplies retrieval documents, and
patient-side dialogue text supplies only language/evaluation queries.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import datetime as dt
import hashlib
import heapq
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from typing import Any, Iterable, Iterator, Sequence


DEFAULT_ENTITY_JSON = Path("/home/linaro/.zeroclaw/workspace/med_neo4j/entity.json")
DEFAULT_RELATION_JSON = Path("/home/linaro/.zeroclaw/workspace/med_neo4j/relation.json")
DEFAULT_DATA_ROOT = Path("/data/structure/med_database")
DEFAULT_ALIASES = Path(__file__).with_name("aliases.json")
SCHEMA_VERSION = "1"
SOURCE_GRAPH_ENTITIES = "med_neo4j/entity.json"
SOURCE_GRAPH_EDGES = "med_neo4j/relation.json"
BATCH_SIZE = 5_000
DIALOGUE_LIMIT_PER_FILE = 10_000


ASPECT_ALIASES = {
    "推荐药": "推荐药",
    "辅助治疗": "辅助治疗",
    "手术治疗": "手术治疗",
    "临床表现": "症状",
    "症状": "症状",
    "影像学检查": "检查",
    "辅助检查": "检查",
    "内窥镜检查": "检查",
    "组织学检查": "检查",
    "检查": "检查",
    "筛查": "检查",
    "结果": "检查结果",
    "病因": "病因",
    "发病原因": "病因",
    "遗传因素": "遗传因素",
    "发病机制": "发病机制",
    "并发症": "并发症",
    "发病部位": "发病部位",
    "就诊科室": "就诊科室",
    "高危因素": "风险因素",
    "风险评估因素": "风险因素",
    "预防措施": "预防",
    "治疗方式": "治疗",
    "放射治疗": "放射治疗",
    "化疗": "化疗",
    "治愈率": "治愈率",
    "治疗周期": "治疗周期",
    "简介": "简介",
    "多发群体": "易感人群",
    "患病比例": "患病率",
    "发病率": "发病率",
    "发病年龄": "发病年龄",
    "发病性别倾向": "性别倾向",
    "多发地区": "多发地区",
    "多发季节": "多发季节",
    "转移部位": "转移部位",
    "治疗后症状": "治疗后症状",
    "死亡率": "死亡率",
    "传播途径": "传播途径",
    "预后生存率": "预后生存率",
    "宜食": "宜食",
    "忌食": "忌食",
}

_ASPECT_PATTERN = "|".join(
    re.escape(item) for item in sorted(ASPECT_ALIASES, key=len, reverse=True)
)
_KG_QUESTION_RE = re.compile(
    rf"^(?P<subject>.+)的(?P<aspect>{_ASPECT_PATTERN})"
    r"(?:有些什么|有哪一些|有哪些|是什么|是多少|多长|是啥|是哪里|在哪里|"
    r"是什么时候|如何|怎么样|怎么治疗)?[？?]?$"
)
_KG_FOOD_RE = re.compile(r"^(?P<subject>.+?)(?P<aspect>宜食|忌食)什么[？?]?$")
_KG_TRANSFORM_RE = re.compile(r"^(?P<subject>.+?)会转化成什么[？?]?$")


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _flatten_text(value: Any) -> list[str]:
    """Return all non-empty scalar strings from arbitrarily nested lists."""
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        result: list[str] = []
        for item in value:
            result.extend(_flatten_text(item))
        return result
    if value is None:
        return []
    text = str(value).strip()
    return [text] if text else []


def _dedupe(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(item for item in items if item))


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Stream values from a top-level JSON array without loading it in memory."""
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8-sig") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            buffer = buffer[position:]
            position = 0
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        refill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                break
            if eof:
                raise ValueError(f"empty JSON input: {path}")
            refill()
        if buffer[position] != "[":
            raise ValueError(f"expected a top-level JSON array: {path}")
        position += 1

        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace() or buffer[position] == ","
                ):
                    position += 1
                if position < len(buffer) or eof:
                    break
                refill()
            if position < len(buffer) and buffer[position] == "]":
                return
            if eof and position >= len(buffer):
                raise ValueError(f"unterminated JSON array: {path}")

            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"malformed JSON array: {path}") from None
                    refill()
                    continue
                position = end
                yield value
                break


def _batched_insert(
    connection: sqlite3.Connection,
    sql: str,
    rows: Iterable[Sequence[Any]],
    batch_size: int,
) -> int:
    batch: list[Sequence[Any]] = []
    count = 0
    for row in rows:
        batch.append(row)
        if len(batch) >= batch_size:
            connection.executemany(sql, batch)
            connection.commit()
            count += len(batch)
            batch.clear()
    if batch:
        connection.executemany(sql, batch)
        connection.commit()
        count += len(batch)
    return count


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode=DELETE;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;

        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE entities (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            label TEXT NOT NULL,
            properties_json TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE TABLE aliases (
            alias TEXT NOT NULL,
            entity_id INTEGER NOT NULL,
            weight REAL NOT NULL,
            source TEXT NOT NULL,
            UNIQUE(alias, entity_id)
        );
        CREATE TABLE edges (
            src_id INTEGER NOT NULL,
            dst_id INTEGER NOT NULL,
            relation TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE TABLE facts (
            id INTEGER PRIMARY KEY,
            subject TEXT NOT NULL,
            aspect TEXT NOT NULL,
            answer TEXT NOT NULL,
            source TEXT NOT NULL,
            source_line INTEGER NOT NULL,
            quality REAL NOT NULL
        );
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            source TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE document_fts USING fts5(
            search_tokens,
            content=''
        );
        CREATE TABLE dialogue_queries (
            id INTEGER PRIMARY KEY,
            department TEXT NOT NULL,
            title TEXT NOT NULL,
            question TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('train', 'eval'))
        );
        """
    )
    connection.commit()


def _entity_rows(
    path: Path,
) -> Iterator[tuple[tuple[Any, ...], tuple[Any, ...]]]:
    for wrapper in iter_json_array(path):
        node = wrapper.get("n", wrapper) if isinstance(wrapper, dict) else {}
        properties = node.get("properties") or {}
        identity = node.get("identity", node.get("id"))
        if identity is None:
            raise ValueError(f"entity without identity in {path}")
        labels = node.get("labels") or []
        if isinstance(labels, str):
            label = labels
        else:
            label = ",".join(str(item) for item in labels if item)
        name = properties.get("名称", properties.get("name", ""))
        entity_id = int(identity)
        name = str(name).strip()
        entity = (entity_id, name, label, _json_text(properties), SOURCE_GRAPH_ENTITIES)
        canonical_alias = (name, entity_id, 1.0, "canonical_name")
        yield entity, canonical_alias


def import_entities(
    connection: sqlite3.Connection, path: Path, batch_size: int
) -> tuple[int, dict[str, list[tuple[int, str]]]]:
    entity_batch: list[tuple[Any, ...]] = []
    alias_batch: list[tuple[Any, ...]] = []
    name_to_ids: dict[str, list[tuple[int, str]]] = {}
    count = 0
    for entity, canonical_alias in _entity_rows(path):
        entity_batch.append(entity)
        if canonical_alias[0]:
            alias_batch.append(canonical_alias)
            name_to_ids.setdefault(entity[1], []).append((entity[0], entity[2]))
        if len(entity_batch) >= batch_size:
            connection.executemany(
                "INSERT INTO entities(id,name,label,properties_json,source) VALUES(?,?,?,?,?)",
                entity_batch,
            )
            connection.executemany(
                "INSERT OR IGNORE INTO aliases(alias,entity_id,weight,source) VALUES(?,?,?,?)",
                alias_batch,
            )
            connection.commit()
            count += len(entity_batch)
            entity_batch.clear()
            alias_batch.clear()
    if entity_batch:
        connection.executemany(
            "INSERT INTO entities(id,name,label,properties_json,source) VALUES(?,?,?,?,?)",
            entity_batch,
        )
        connection.executemany(
            "INSERT OR IGNORE INTO aliases(alias,entity_id,weight,source) VALUES(?,?,?,?)",
            alias_batch,
        )
        connection.commit()
        count += len(entity_batch)
    return count, name_to_ids


def _preferred_entity(candidates: list[tuple[int, str]]) -> int:
    def rank(candidate: tuple[int, str]) -> tuple[int, int]:
        entity_id, label = candidate
        labels = set(label.split(","))
        priority = 0 if "症状" in labels else 1 if "疾病" in labels else 2
        return priority, entity_id

    return min(candidates, key=rank)[0]


def import_manual_aliases(
    connection: sqlite3.Connection,
    aliases_path: Path,
    name_to_ids: dict[str, list[tuple[int, str]]],
) -> tuple[int, list[str]]:
    with aliases_path.open("r", encoding="utf-8-sig") as handle:
        aliases = json.load(handle)
    if not isinstance(aliases, dict):
        raise ValueError(f"aliases file must be an object: {aliases_path}")
    rows: list[tuple[Any, ...]] = []
    unresolved: list[str] = []
    for canonical, variants in aliases.items():
        candidates = name_to_ids.get(str(canonical).strip())
        if not candidates:
            unresolved.append(str(canonical))
            continue
        entity_id = _preferred_entity(candidates)
        if not isinstance(variants, list):
            raise ValueError(f"aliases for {canonical!r} must be a list")
        for alias in _dedupe(str(item).strip() for item in variants):
            rows.append((alias, entity_id, 0.98, "manual_aliases"))
    before = connection.total_changes
    connection.executemany(
        "INSERT OR IGNORE INTO aliases(alias,entity_id,weight,source) VALUES(?,?,?,?)",
        rows,
    )
    connection.commit()
    return connection.total_changes - before, unresolved


def _edge_rows(path: Path) -> Iterator[tuple[Any, ...]]:
    for wrapper in iter_json_array(path):
        edge = wrapper.get("r", wrapper) if isinstance(wrapper, dict) else {}
        src = edge.get("start", edge.get("src_id"))
        dst = edge.get("end", edge.get("dst_id"))
        relation = edge.get("type", edge.get("relation", ""))
        if src is None or dst is None:
            raise ValueError(f"edge without endpoints in {path}")
        yield int(src), int(dst), str(relation).strip(), SOURCE_GRAPH_EDGES


def parse_kg_question(question: str) -> tuple[str, str, float]:
    question = question.strip()
    match = _KG_QUESTION_RE.match(question) or _KG_FOOD_RE.match(question)
    if match:
        subject = match.group("subject").strip()
        aspect = ASPECT_ALIASES[match.group("aspect")]
        return subject, aspect, 1.0
    match = _KG_TRANSFORM_RE.match(question)
    if match:
        return match.group("subject").strip(), "转化", 1.0
    # Retain unrecognised train records, but mark them so retrieval can rank
    # well-understood template facts above them.
    return question.rstrip("？?").strip(), "其他", 0.4


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                yield line_number, json.loads(line)


def _fact_rows(
    path: Path, counters: dict[str, int] | None = None
) -> Iterator[tuple[Any, ...]]:
    source = "huatuo_knowledge_graph_qa/train"
    for line_number, record in _iter_jsonl(path):
        if counters is not None:
            counters["seen"] = counters.get("seen", 0) + 1
        questions = _dedupe(_flatten_text(record.get("questions")))
        answers = _dedupe(_flatten_text(record.get("answers")))
        if not questions or not answers:
            if counters is not None:
                counters["skipped"] = counters.get("skipped", 0) + 1
            continue
        subject, aspect, quality = parse_kg_question(questions[0])
        # Unknown question forms do not have a reliable subject/aspect boundary.
        # Keeping the whole question as a subject harms exact retrieval and makes
        # the index needlessly large, so only recognised templates are facts.
        if quality < 1.0:
            if counters is not None:
                counters["skipped"] = counters.get("skipped", 0) + 1
            continue
        answer = "；".join(answers).strip()[:4000]
        if subject and answer:
            yield subject, aspect, answer, source, line_number, quality


def question_search_tokens(question: str) -> str:
    """Create space-separated character unigram/bigram tokens for FTS5."""
    runs: list[list[str]] = []
    current: list[str] = []
    for char in question.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    tokens: list[str] = []
    for run in runs:
        tokens.extend(run)
        tokens.extend("".join(run[index : index + 2]) for index in range(len(run) - 1))
    return " ".join(tokens)


def import_documents(
    connection: sqlite3.Connection, path: Path, batch_size: int
) -> int:
    source = "huatuo_encyclopedia_qa/train"
    document_batch: list[tuple[str, str, str]] = []
    token_batch: list[tuple[int, str]] = []
    count = 0
    next_id = 1
    for _, record in _iter_jsonl(path):
        questions = _dedupe(_flatten_text(record.get("questions")))
        answers = _dedupe(_flatten_text(record.get("answers")))
        if not questions or not answers:
            continue
        question = "；".join(questions)
        answer = "\n".join(answers)[:4000]
        document_batch.append((question, answer, source))
        token_batch.append((next_id, question_search_tokens(question)))
        next_id += 1
        if len(document_batch) >= batch_size:
            connection.executemany(
                "INSERT INTO documents(question,answer,source) VALUES(?,?,?)",
                document_batch,
            )
            connection.executemany(
                "INSERT INTO document_fts(rowid,search_tokens) VALUES(?,?)", token_batch
            )
            connection.commit()
            count += len(document_batch)
            document_batch.clear()
            token_batch.clear()
    if document_batch:
        connection.executemany(
            "INSERT INTO documents(question,answer,source) VALUES(?,?,?)", document_batch
        )
        connection.executemany(
            "INSERT INTO document_fts(rowid,search_tokens) VALUES(?,?)", token_batch
        )
        connection.commit()
        count += len(document_batch)
    return count


def _detect_csv_encoding(path: Path) -> str:
    with path.open("rb") as handle:
        sample = handle.read(1 << 20)
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
            decoder.decode(sample, final=False)
            return encoding
        except UnicodeDecodeError:
            continue
    raise UnicodeError(f"cannot decode dialogue CSV as utf-8-sig or gb18030: {path}")


def _dialogue_candidates(path: Path, limit: int) -> list[tuple[str, str, str, str]]:
    """Select the deterministic lowest hashes without retaining answer text."""
    encoding = _detect_csv_encoding(path)
    heap: list[tuple[int, int, tuple[str, str, str, str]]] = []
    fallback_department = path.parent.name.rsplit("_", 1)[-1]
    csv.field_size_limit(min(sys.maxsize, 2_147_483_647))
    with path.open("r", encoding=encoding, newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [item.lstrip("\ufeff").strip().lower() for item in next(reader)]
        except StopIteration:
            return []
        indexes = {name: index for index, name in enumerate(header)}
        required = {"department", "title", "ask"}
        if not required.issubset(indexes):
            missing = ", ".join(sorted(required - indexes.keys()))
            raise ValueError(f"missing CSV columns {missing} in {path}")
        needed_max = max(indexes[name] for name in required)
        for serial, row in enumerate(reader, 1):
            if len(row) <= needed_max:
                continue
            department = row[indexes["department"]].strip() or fallback_department
            title = row[indexes["title"]].strip()
            question = row[indexes["ask"]].strip()
            if not question:
                continue
            digest = hashlib.sha256(
                (department + "\0" + title + "\0" + question).encode("utf-8")
            ).digest()
            hash_value = int.from_bytes(digest[:8], "big")
            split = "eval" if int.from_bytes(digest[8:16], "big") % 10 == 0 else "train"
            item = (department, title, question, split)
            candidate = (-hash_value, -serial, item)
            if len(heap) < limit:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)
    return [entry[2] for entry in sorted(heap, key=lambda entry: (-entry[0], -entry[1]))]


def import_dialogues(
    connection: sqlite3.Connection, data_dir: Path, batch_size: int
) -> tuple[int, int, int]:
    csv_paths = sorted(data_dir.glob("*/*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"no dialogue CSV files under {data_dir}")
    total = train_count = eval_count = 0
    for path in csv_paths:
        rows = _dialogue_candidates(path, DIALOGUE_LIMIT_PER_FILE)
        for offset in range(0, len(rows), batch_size):
            connection.executemany(
                "INSERT INTO dialogue_queries(department,title,question,split) "
                "VALUES(?,?,?,?)",
                rows[offset : offset + batch_size],
            )
            connection.commit()
        total += len(rows)
        train_count += sum(row[3] == "train" for row in rows)
        eval_count += sum(row[3] == "eval" for row in rows)
    return total, train_count, eval_count


def _create_indexes(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE INDEX idx_entities_name ON entities(name);
        CREATE INDEX idx_entities_label ON entities(label);
        CREATE INDEX idx_aliases_alias ON aliases(alias);
        CREATE INDEX idx_aliases_entity_id ON aliases(entity_id);
        CREATE INDEX idx_edges_src_id ON edges(src_id);
        CREATE INDEX idx_edges_dst_id ON edges(dst_id);
        CREATE INDEX idx_edges_relation ON edges(relation);
        CREATE INDEX idx_facts_subject ON facts(subject);
        CREATE INDEX idx_facts_aspect ON facts(aspect);
        CREATE INDEX idx_dialogue_split ON dialogue_queries(split);
        """
    )
    connection.commit()


def _write_meta(connection: sqlite3.Connection, values: dict[str, Any]) -> None:
    rows = [(key, str(value)) for key, value in sorted(values.items())]
    connection.executemany("INSERT INTO meta(key,value) VALUES(?,?)", rows)
    connection.commit()


def build_index(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent, delete=False
    )
    temporary_path = Path(temporary_handle.name)
    temporary_handle.close()

    stats: dict[str, Any] = {}
    try:
        connection = sqlite3.connect(temporary_path)
        try:
            _create_schema(connection)
            entity_path = Path(args.entity_json)
            relation_path = Path(args.relation_json)
            data_root = Path(args.data_root)
            entity_count, name_to_ids = import_entities(
                connection, entity_path, args.batch_size
            )
            stats["entities_count"] = entity_count
            manual_count, unresolved = import_manual_aliases(
                connection, Path(args.aliases), name_to_ids
            )
            stats["manual_aliases_count"] = manual_count
            stats["manual_aliases_unresolved"] = ",".join(unresolved)
            stats["aliases_count"] = connection.execute(
                "SELECT count(*) FROM aliases"
            ).fetchone()[0]
            stats["edges_count"] = _batched_insert(
                connection,
                "INSERT INTO edges(src_id,dst_id,relation,source) VALUES(?,?,?,?)",
                _edge_rows(relation_path),
                args.batch_size,
            )

            if args.skip_facts:
                stats["facts_count"] = 0
                stats["facts_skipped_count"] = 0
            else:
                facts_path = data_root / "huatuo_knowledge_graph_qa" / "train_datasets.jsonl"
                fact_counters: dict[str, int] = {}
                stats["facts_count"] = _batched_insert(
                    connection,
                    "INSERT INTO facts(subject,aspect,answer,source,source_line,quality) "
                    "VALUES(?,?,?,?,?,?)",
                    _fact_rows(facts_path, fact_counters),
                    args.batch_size,
                )
                stats["facts_skipped_count"] = fact_counters.get("skipped", 0)

            if args.skip_documents:
                stats["documents_count"] = 0
            else:
                documents_path = data_root / "huatuo_encyclopedia_qa" / "train_datasets.jsonl"
                stats["documents_count"] = import_documents(
                    connection, documents_path, args.batch_size
                )

            if args.skip_dialogues:
                dialogue_total = dialogue_train = dialogue_eval = 0
            else:
                dialogue_dir = data_root / "Chinese-medical-dialogue-data" / "Data"
                dialogue_total, dialogue_train, dialogue_eval = import_dialogues(
                    connection, dialogue_dir, args.batch_size
                )
            stats.update(
                dialogue_queries_count=dialogue_total,
                dialogue_train_count=dialogue_train,
                dialogue_eval_count=dialogue_eval,
            )

            _create_indexes(connection)
            built_at = dt.datetime.now(dt.timezone.utc).isoformat()
            meta = {
                "schema_version": SCHEMA_VERSION,
                "built_at_utc": built_at,
                "entity_source": str(entity_path),
                "relation_source": str(relation_path),
                "facts_policy": "Huatuo knowledge-graph QA train only; validation/test excluded; "
                "known templates normalized; unparsed train records skipped; answers truncated "
                "to 4000 characters",
                "documents_policy": "Huatuo encyclopedia QA train only; validation/test excluded; "
                "answers truncated to 4000 characters; only question tokens indexed",
                "document_fts_policy": "lowercase Chinese/English alphanumeric character "
                "unigrams and adjacent bigrams, space separated",
                "dialogues_policy": "patient-side department/title/ask only; answer excluded; "
                "deterministic SHA-256 bottom-hash sample capped at 10000 per specialty file",
                "dialogue_split_policy": "SHA-256 deterministic 90% train / 10% eval",
                "aliases_policy": "all canonical entity names plus reviewed aliases.json; "
                "manual aliases prefer symptom-labelled entities",
                "skip_facts": int(args.skip_facts),
                "skip_documents": int(args.skip_documents),
                "skip_dialogues": int(args.skip_dialogues),
                **stats,
            }
            _write_meta(connection, meta)
            connection.execute("PRAGMA optimize")
            connection.commit()
        finally:
            connection.close()
        os.replace(temporary_path, output)
    except BaseException:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise
    return stats


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="destination SQLite database")
    parser.add_argument("--entity-json", default=str(DEFAULT_ENTITY_JSON))
    parser.add_argument("--relation-json", default=str(DEFAULT_RELATION_JSON))
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--aliases", default=str(DEFAULT_ALIASES))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--skip-facts", action="store_true")
    parser.add_argument("--skip-documents", action="store_true")
    parser.add_argument("--skip-dialogues", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    stats = build_index(args)
    print(json.dumps(stats, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
