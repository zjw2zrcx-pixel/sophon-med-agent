"""Medical knowledge graph query tool backed by med_query binary."""
from __future__ import annotations

import asyncio
import logging
from typing import Dict

from ..base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

MED_QUERY_BIN = "/home/linaro/.zeroclaw/workspace/med_neo4j/med_query"


class MedQueryTool(Tool):
    name = "med_query"
    description = (
        "查询本地医疗知识图谱，获取疾病、症状、药物等医学信息。"
        "用 fuzzy 将口语词校正为精确实体名；用 entity 查看实体可查询的属性；"
        "用 graph 查看一个实体的全部关联；用 prop 查询一个实体的具体属性；"
        "仅当 arg1、arg2 都是具体实体名时使用 rel 查询两实体关系。"
    )
    param_schema = {
        "command": "entity | prop | rel | graph | fuzzy | exists",
        "arg1": "entity/graph: 精确实体名；fuzzy: 关键词；prop/rel: 第一个精确实体名",
        "arg2": "prop: entity 返回的属性名；rel: 另一个具体实体名。不要把“症状/疾病”等类别名传给 rel",
        "filter": "graph 可选筛选词，如 常用药品、症状、治疗方法；询问吃什么药时使用 常用药品",
        "limit": "模糊搜索返回数量，默认10 (仅fuzzy命令使用，可选)",
    }
    modes = ["Voice"]
    harness_metadata = {
        "effect": "READ", "idempotent": True,
        "produces": ["medical.query_result"], "invalidates": [],
        "retry": {
            "max_attempts": 2,
            "allowed_errors": ["NOT_FOUND", "TIMEOUT", "TEMPORARY_UNAVAILABLE"],
        },
    }

    async def call(self, params: Dict[str, str], context: ToolContext) -> ToolResult:
        cmd = params.get("command", "").strip().lower()
        arg1 = params.get("arg1", "").strip()
        arg2 = params.get("arg2", "").strip()

        valid = {"entity", "prop", "rel", "graph", "fuzzy", "exists"}
        if cmd not in valid:
            return ToolResult(
                success=False, error=f"无效子命令 '{cmd}'",
                data=f"可用命令: {', '.join(sorted(valid))}"
            )
        if not arg1:
            return ToolResult(
                success=False,
                error=f"{cmd} 命令缺少 arg1",
                data="请提供要查询的实体、关键词或属性名",
            )
        if cmd in {"prop", "rel"} and not arg2:
            return ToolResult(
                success=False,
                error=f"{cmd} 命令缺少 arg2",
                data="prop/rel 命令必须提供第二个参数",
            )

        args = [MED_QUERY_BIN, cmd]
        if arg1:
            args.append(arg1)
        if arg2:
            args.append(arg2)
        if cmd == "fuzzy" and params.get("limit"):
            try:
                limit = int(params["limit"])
                if not 1 <= limit <= 100:
                    raise ValueError
            except (TypeError, ValueError):
                return ToolResult(
                    success=False,
                    error="limit 必须是 1 到 100 的整数",
                    data="",
                )
            args.extend(["--limit", str(limit)])

        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=30.0
            )
            output = stdout.decode("utf-8", errors="replace").strip()
            if stderr:
                err_text = stderr.decode("utf-8", errors="replace").strip()
                if err_text:
                    output += f"\n[stderr] {err_text}"
            if proc.returncode != 0:
                return ToolResult(
                    success=False,
                    error=f"med_query 退出码 {proc.returncode}",
                    data=output,
                )
            if cmd == "graph" and params.get("filter", "").strip():
                filter_text = params["filter"].strip()
                matched = [
                    line for line in output.splitlines()[1:]
                    if filter_text in line
                ]
                if not matched:
                    output = f"未找到包含筛选词 '{filter_text}' 的关联"
                else:
                    output = (
                        f"实体 '{arg1}' 中包含 '{filter_text}' 的关联"
                        f"（共 {len(matched)} 条）:\n"
                        + "\n".join(matched[:20])
                    )
            if not output:
                return ToolResult(
                    success=False,
                    data="(无结果)",
                    error="查询未返回结果",
                    error_type="NOT_FOUND",
                    empty=True,
                    retryable=True,
                    recovery_hint="请缩短关键词、替换同义词或改变查询句式。",
                )
            return ToolResult(success=True, data=output)
        except asyncio.TimeoutError:
            if proc and proc.returncode is None:
                proc.kill()
                await proc.communicate()
            return ToolResult(
                success=False, error="查询超时(30s)", error_type="TIMEOUT",
                retryable=True, data="",
            )
        except FileNotFoundError:
            return ToolResult(
                success=False, error=f"med_query 二进制未找到: {MED_QUERY_BIN}",
                error_type="UNAVAILABLE", data="",
            )
        except Exception as e:
            return ToolResult(success=False, error=str(e), data="")
