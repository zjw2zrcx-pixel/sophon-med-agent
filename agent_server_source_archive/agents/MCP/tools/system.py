from __future__ import annotations

import asyncio
import platform
import psutil
import logging
from datetime import datetime
from typing import Dict

from ..base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)


class GetSystemStatsTool(Tool):
    name = "get_system_stats"
    description = "获取系统运行状态，包括CPU、内存、磁盘等信息"
    param_schema = {}
    modes = ["Voice", "Benchmark"]
    harness_metadata = {
        "effect": "READ", "idempotent": True,
        "produces": ["system.stats"], "invalidates": [],
        "retry": {"max_attempts": 2},
    }

    async def call(self, params: Dict[str, str], context: ToolContext) -> ToolResult:
        try:
            cpu_percent = await asyncio.to_thread(psutil.cpu_percent, 0.5)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            stats = (
                f"操作系统: {platform.system()} {platform.release()}\n"
                f"CPU使用率: {cpu_percent}%\n"
                f"内存: {memory.percent}% (已用 {memory.used // (1024**3)}GB / 总计 {memory.total // (1024**3)}GB)\n"
                f"磁盘: {disk.percent}% (已用 {disk.used // (1024**3)}GB / 总计 {disk.total // (1024**3)}GB)"
            )
            return ToolResult(success=True, data=stats, facts={"system.stats": stats})
        except Exception as e:
            logger.error(f"获取系统状态失败: {e}")
            return ToolResult(success=False, error=str(e), data=f"获取系统状态失败: {e}")


class GetTimeTool(Tool):
    name = "get_time"
    description = "获取当前系统时间和日期"
    param_schema = {
        "format": "时间格式(可选, 默认为 'full'，可选 'date' 仅日期, 'time' 仅时间)",
    }
    modes = ["Voice", "Benchmark"]
    harness_metadata = {
        "effect": "READ", "idempotent": True,
        "produces": ["system.time"], "invalidates": [],
        "retry": {"max_attempts": 2},
    }

    async def call(self, params: Dict[str, str], context: ToolContext) -> ToolResult:
        fmt = params.get("format", "full")
        now = datetime.now()
        if fmt == "date":
            weekdays = "一二三四五六日"
            result = now.strftime("%Y-%m-%d") + f" 星期{weekdays[now.weekday()]}"
        elif fmt == "time":
            result = now.strftime("%H:%M:%S")
        else:
            result = now.strftime("%Y-%m-%d %H:%M:%S")
        text = f"当前时间: {result}"
        return ToolResult(success=True, data=text, facts={"system.time": result})
