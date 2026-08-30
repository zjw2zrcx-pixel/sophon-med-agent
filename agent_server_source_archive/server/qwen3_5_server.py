import sys
import os
import uuid
import asyncio
import json
import tempfile
import base64
import time
import argparse
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

import atexit

from logging_utils import setup_colored_logging
from prefix_cache import prepare_prompt
from benchmark_api import (
    completion_limit, finish_reason as infer_finish_reason, no_cache_diagnostics,
    output_token_hash, prompt_slot_messages, request_max_tokens,
)

logger = setup_colored_logging("qwen3-5")

# --- log level for verbose per-request detail; set to logging.DEBUG to suppress ---
_MSG_DUMP_LEVEL = logging.DEBUG

pipeline = None
server_state = "initializing"
sessions = {}
executor = ThreadPoolExecutor(max_workers=1)
last_request_timing = {}

class SimpleArgs:
    devid = 0
    video_ratio = 0.25
    max_new_tokens = 512
    model_path = ""
    config_path = ""


def make_args(cfg: dict):
    args = SimpleArgs()
    args.devid = cfg.get("devid", 0)
    args.video_ratio = cfg.get("video_ratio", 0.25)
    args.max_new_tokens = cfg.get("max_new_tokens", 512)
    args.max_new_tokens = cfg.get("max_new_tokens", 512)
    args.model_path = cfg["model_path"]
    args.config_path = cfg["config_path"]
    return args


async def load_model():
    global pipeline, server_state
    module_path = os.environ.get("MODULE_PATH", "../Qwen3_5/python_demo")
    cfg = app.state.cfg

    try:
        server_state = "loading"
        logger.info(f"Adding module path: {module_path}")
        if module_path not in sys.path:
            sys.path.insert(0, module_path)

        import chat
        from pipeline import Qwen3_5

        args = make_args(cfg)
        logger.info(f"Loading model from {args.model_path} ...")
        loop = asyncio.get_event_loop()
        pipeline = await loop.run_in_executor(None, _load_model_sync, args, Qwen3_5)
        if getattr(pipeline, "support_history", False):
            raise RuntimeError(
                "history-capable model must use qwen3_5_history_server.py"
            )
        server_state = "ready"
        logger.info("Model loaded successfully.")
    except Exception as e:
        server_state = "error"
        logger.error(f"Failed to load model: {e}")
        if pipeline is not None:
            try:
                pipeline.model.deinit()
            except Exception:
                pass
            pipeline = None


def _load_model_sync(args, cls):
    return cls(args)


def _cleanup_model():
    global pipeline
    if pipeline is not None:
        try:
            logger.info("Calling model.deinit() (atexit) ...")
            pipeline.model.deinit()
        except Exception:
            pass
        pipeline = None


# ---------------------------------------------------------------------------
# Session GC — prevent unbounded dict growth
# ---------------------------------------------------------------------------

SESSION_TTL_SECONDS = 900  # 15 min idle


def _touch_session(session_id: str, clear_history: bool) -> bool:
    """Register/touch a session; returns True if KV cache should be cleared."""
    if not session_id:
        return True
    now = time.time()
    info = sessions.get(session_id)
    if info is None:
        sessions[session_id] = {"processed_len": 0, "last_access": now}
        return True
    info["last_access"] = now
    return clear_history


async def _session_cleanup_loop():
    """Evict stale sessions every 5 minutes."""
    while True:
        await asyncio.sleep(300)
        now = time.time()
        stale = [
            sid for sid, info in sessions.items()
            if now - info.get("last_access", 0) > SESSION_TTL_SECONDS
        ]
        for sid in stale:
            del sessions[sid]
        if stale:
            logger.info(f"Session GC: evicted {len(stale)}, {len(sessions)} remain")


@asynccontextmanager
async def lifespan(fastapi_app: FastAPI):
    asyncio.get_event_loop().create_task(load_model())
    asyncio.get_event_loop().create_task(_session_cleanup_loop())
    yield
    _cleanup_model()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": server_state}


@app.get("/status")
async def status():
    return {
        "status": server_state,
        "model": "qwen3.5",
        "type": "chat",
    }


@app.post("/shutdown")
async def do_shutdown():
    _cleanup_model()
    logger.info("Shutting down...")
    asyncio.get_event_loop().call_later(1.0, lambda: os._exit(0))
    return JSONResponse({"status": "shutting_down"})


_TOOL_CALL_TAG_START = "\n```tool_call\n"
_TOOL_CALL_TAG_END = "\n```\n"


def _build_tool_prompt(tools):
    """将 OpenAI tools 格式转换为文本描述，注入 system prompt。"""
    if not tools:
        return ""
    lines = [
        "",
        "You may call one or more tools. After receiving tool results, you may call more tools or provide a final answer.",
        "If you decide not to call any tool, respond normally with text.",
        "",
    ]
    for tool in tools:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "")
        lines.append(f"## {name}")
        if desc:
            lines.append(desc)
        params = func.get("parameters", {})
        if params:
            lines.append("Parameters: " + json.dumps(params, ensure_ascii=False))
        lines.append("")
    lines.append("To call a tool, output a single line in this exact format:")
    lines.append("```tool_call")
    lines.append('{"name": "<tool_name>", "arguments": {"param1": "value1"}}')
    lines.append("```")
    lines.append("Only one tool call per response. After receiving the tool result, you may call another tool or provide a final answer.")
    return "\n".join(lines)


def _parse_tool_calls_from_text(text):
    """从模型输出中解析 tool_call 块，返回 (clean_text, tool_calls_list)。

    tool_call 块格式:
    ```tool_call
    {"name": "...", "arguments": {...}}
    ```
    """
    tool_calls = []
    remaining_parts = []
    pos = 0
    tag_start = _TOOL_CALL_TAG_START
    tag_end = _TOOL_CALL_TAG_END
    while True:
        start = text.find(tag_start, pos)
        if start == -1:
            remaining_parts.append(text[pos:])
            break
        remaining_parts.append(text[pos:start])
        end = text.find(tag_end, start + len(tag_start))
        if end == -1:
            remaining_parts.append(text[start:])
            break
        inner = text[start + len(tag_start):end].strip()
        pos = end + len(tag_end)
        try:
            obj = json.loads(inner)
            name = obj.get("name", "")
            arguments = obj.get("arguments", {})
            if isinstance(arguments, dict):
                arguments = json.dumps(arguments, ensure_ascii=False)
            elif not isinstance(arguments, str):
                arguments = json.dumps(arguments, ensure_ascii=False)
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
                },
            })
        except (json.JSONDecodeError, AttributeError) as e:
            logger.warning(f"Failed to parse tool call JSON: {inner!r}: {e}")
            remaining_parts.append(text[start:end + len(tag_end)])
    clean = "".join(remaining_parts).strip()
    return clean, tool_calls


def _build_msgs_from_messages(messages, pipeline_obj, tools=None):
    """
    将 OpenAI 格式的 messages 列表转换为 Qwen3.5 pipeline 支持的格式。

    关键约束:
    - Qwen3.5 的 apply_chat_template 遍历所有消息的 content，
      对每个 item 访问 content["type"] 来提取视觉内容
    - 如果 content 是字符串，迭代会得到单字符，导致 string indices must be integers
    - 因此：所有消息的 content 必须是 list-of-dicts 格式，
      纯文本消息也用 [{"type": "text", "text": "..."}]

    策略:
    1. 如果提供了 tools，将工具描述注入到 system 消息
    2. 遍历所有消息，提取图像到临时文件，记录哪条消息有图像
    3. 处理 OpenAI tool 角色和 assistant 的 tool_calls 字段
    4. 所有消息的 content 统一为 list-of-dicts 格式
    5. 含图像的消息使用 [image, text] 列表
    6. 合并连续同角色消息以满足 ChatML 交替要求
    """
    msgs = []
    temp_image_paths = []

    # 如果有 tools，构建工具描述准备注入 system 消息
    tool_prompt = _build_tool_prompt(tools) if tools else ""

    processed = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        effective_role = "user"
        content_prefix = ""

        if role == "system":
            effective_role = "system"
        elif role == "assistant":
            effective_role = "assistant"
        elif role == "tool":
            # OpenAI 标准工具结果角色，转为 user
            effective_role = "user"
            tool_call_id = msg.get("tool_call_id", "")
            if tool_call_id:
                content_prefix = f"[Tool Result (id={tool_call_id})]\n"
            else:
                content_prefix = "[Tool Result]\n"
        elif role == "tool_result":
            effective_role = "user"
            source = msg.get("name", "")
            if source:
                content_prefix = f"[工具结果: {source}]\n"
        elif role == "skill_result":
            effective_role = "user"
            source = msg.get("name", "")
            if source:
                content_prefix = f"[技能结果: {source}]\n"
        else:
            effective_role = "user"

        # 处理 assistant 消息中的 tool_calls 字段
        if role == "assistant" and msg.get("tool_calls"):
            tool_calls = msg.get("tool_calls", [])
            tool_call_texts = []
            for tc in tool_calls:
                func = tc.get("function", {})
                name = func.get("name", "")
                arguments = func.get("arguments", "{}")
                try:
                    args_obj = json.loads(arguments) if isinstance(arguments, str) else arguments
                    tool_call_texts.append(
                        json.dumps({"name": name, "arguments": args_obj}, ensure_ascii=False)
                    )
                except (json.JSONDecodeError, TypeError):
                    tool_call_texts.append(
                        json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
                    )
            combined = _TOOL_CALL_TAG_START + ("\n".join(tool_call_texts)) + _TOOL_CALL_TAG_END
            if isinstance(content, str) and content.strip():
                text_parts = [content, combined]
            elif isinstance(content, list):
                text_parts = [_extract_text_from_content(content), combined]
            else:
                text_parts = [combined]
            content = "\n\n".join(text_parts)

        text_parts_local = []
        image_path = None

        if isinstance(content, str):
            text_parts_local.append(content)
        elif isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    text_parts_local.append(part.get("text", ""))
                elif part.get("type") == "image_url":
                    image_url = part.get("image_url", {}).get("url", "")
                    if image_url.startswith("data:"):
                        header, b64data = image_url.split(",", 1)
                        ext = "jpg"
                        if "image/png" in header:
                            ext = "png"
                        elif "image/webp" in header:
                            ext = "webp"
                        tmp = tempfile.NamedTemporaryFile(suffix=f".{ext}", delete=False)
                        tmp.write(base64.b64decode(b64data))
                        tmp.close()
                        image_path = tmp.name
                        temp_image_paths.append(tmp.name)
                    else:
                        image_path = image_url

        text_content = content_prefix + "\n".join(text_parts_local) if text_parts_local else content_prefix
        has_image = image_path is not None

        # 将工具描述注入到第一条 system 消息中
        if effective_role == "system" and tool_prompt and not content_prefix.endswith(tool_prompt):
            if tool_prompt not in text_content:
                text_content = text_content + tool_prompt if text_content else tool_prompt.strip()
            tool_prompt = ""  # 只注入一次

        processed.append((effective_role, text_content, has_image, image_path))

    # 如果有工具描述但未注入（没有 system 消息），在开头插入一条 system 消息
    if tool_prompt:
        has_system = any(r == "system" for r, _, _, _ in processed)
        if not has_system:
            processed.insert(0, ("system", tool_prompt.strip(), False, None))
        # 如果已有 system 消息但上面注入过程中 tool_prompt 未被消费（不应发生），也插入
        elif tool_prompt:
            # tool_prompt 仍非空说明它未被任何 system 消息消费，追加到最后一条 system
            for i, (r, t, hi, ip) in enumerate(processed):
                if r == "system":
                    tool_text = tool_prompt.strip()
                    processed[i] = (r, t + tool_text if t else tool_text, hi, ip)
                    break

    for i, (role, text_content, has_image, image_path) in enumerate(processed):
        if has_image and image_path:
            msg_content = [
                {
                    "type": "image",
                    "image": image_path,
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": pipeline_obj.model.MAX_PIXELS,
                },
                {"type": "text", "text": text_content if text_content else ""},
            ]
            msgs.append({"role": role, "content": msg_content})
            # image_path is already in temp_image_paths if it was a temp file
        else:
            msg_content = [{"type": "text", "text": text_content}]
            msgs.append({"role": role, "content": msg_content})

    msgs = _merge_consecutive_roles(msgs)

    return msgs, temp_image_paths


def _extract_text_from_content(content):
    """从多模态 content 列表中提取文本"""
    if isinstance(content, str):
        return content
    texts = []
    for part in content:
        if part.get("type") == "text":
            texts.append(part.get("text", ""))
    return "\n".join(texts)


def _merge_consecutive_roles(msgs):
    """
    合并连续相同角色的消息。
    ChatML 要求 user/assistant 交替，连续同角色需要合并。

    重要: 合并后的消息 content 必须保持 list-of-dicts 格式，
    因为 apply_chat_template 遍历 content 时对每个 item 访问 item["type"]，
    字符串格式会导致 string indices must be integers 错误。
    合并时提取所有文本部分拼接，图像仅保留第一条消息的。
    """
    if not msgs:
        return msgs

    merged = []
    buffer = msgs[0].copy()

    def _ensure_list(content):
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        return content

    buffer["content"] = _ensure_list(buffer.get("content", ""))

    for msg in msgs[1:]:
        buf_role = buffer.get("role", "user")
        msg_role = msg.get("role", "user")

        def normalize_role(r):
            return "user" if r in ("user", "tool_result", "skill_result") else r

        msg_content_list = _ensure_list(msg.get("content", ""))

        if normalize_role(buf_role) == normalize_role(msg_role):
            buf_text = _extract_text_from_content(buffer["content"])
            msg_text = _extract_text_from_content(msg_content_list)
            buffer["content"] = [{"type": "text", "text": buf_text + "\n" + msg_text}]
        else:
            merged.append(buffer)
            buffer = msg.copy()
            buffer["content"] = _ensure_list(buffer.get("content", ""))

    merged.append(buffer)

    if merged and merged[0].get("role") not in ("system", "user"):
        merged.insert(0, {"role": "system", "content": [{"type": "text", "text": ""}]})

    return merged


def _list_content_to_str(content_list):
    """将列表格式的 content 转换为纯字符串（提取文本部分，忽略图像）"""
    if not isinstance(content_list, list):
        return str(content_list)
    texts = []
    for part in content_list:
        if isinstance(part, dict):
            if part.get("type") == "text":
                texts.append(part.get("text", ""))
        else:
            texts.append(str(part))
    return "\n".join(texts)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    if server_state != "ready" or pipeline is None:
        return JSONResponse(
            {"error": {"message": f"Server not ready, current state: {server_state}", "type": "server_error"}},
            status_code=503,
        )

    body = await request.json()
    model_name = body.get("model", "")
    messages = body.get("messages", [])
    prompt_slots = body.get("prompt_slots")
    stream = body.get("stream", False)
    temperature = body.get("temperature", None)
    tools = body.get("tools", None)
    benchmark = body.get("benchmark", False) is True
    session_id = request.headers.get("x-session-id", "")
    clear_history = request.headers.get("x-clear-history", "").lower() == "true"

    try:
        max_tokens = request_max_tokens(
            body, app.state.cfg.get("max_new_tokens", 512))
        if prompt_slots is not None:
            if not benchmark:
                raise ValueError("prompt_slots is available only when benchmark=true")
            messages = prompt_slot_messages(prompt_slots)
    except ValueError as exc:
        return JSONResponse(
            {"error": {"message": str(exc), "type": "invalid_request_error"}},
            status_code=400)

    clear_flag = _touch_session(session_id, clear_history)

    # 构建消息列表 (支持多角色、工具注入)
    msgs, temp_image_paths = _build_msgs_from_messages(messages, pipeline, tools=tools)

    # Log raw input to LLM
    import json as _json
    try:
        raw_input = _json.dumps(
            [{"role": m.get("role", "?"), "content": str(m.get("content", ""))}
             for m in msgs], ensure_ascii=False)
    except Exception:
        raw_input = "<serialization failed>"
    # Log each message on its own line for readability
    logger.info(f"LLM_RAW_INPUT ({len(msgs)} msgs):")
    for _i, _m in enumerate(msgs):
        _role = _m.get("role", "?")
        _content = str(_m.get("content", ""))
        logger.info(f"  LLM_IN_MSG[{_i}] role={_role} len={len(_content)}")
        for _line in _content.split('\n'):
            if _line.strip():
                logger.info(f"  LLM_IN_MSG[{_i}] | {_line}")

    # 确定媒体类型
    media_type = "text"
    if temp_image_paths:
        media_type = "image"

    if not msgs:
        return JSONResponse({"error": {"message": "No valid messages provided", "type": "invalid_request_error"}}, status_code=400)

    # 日志: 记录消息结构
    if logger.isEnabledFor(_MSG_DUMP_LEVEL):
        msg_summary = []
        for m in msgs:
            role = m.get("role", "?")
            content = m.get("content", "")
            if isinstance(content, list):
                text_part = next((p.get("text", "")[:50] for p in content if p.get("type") == "text"), "")
                img_count = sum(1 for p in content if p.get("type") == "image")
                msg_summary.append(f"{role}[{len(content)} parts, {img_count} img]: {text_part}...")
            else:
                msg_summary.append(f"{role}: {str(content)[:80]}")
        logger.log(
            _MSG_DUMP_LEVEL,
            f"Request fields: {sorted(body.keys())} | "
            f"model={model_name} stream={stream} msgs={len(messages)} temp={temperature} "
            f"tools={len(tools) if tools else 0}"
        )
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            if isinstance(content, list):
                total_len = sum(len(json.dumps(p, ensure_ascii=False)) for p in content)
            else:
                total_len = len(str(content))
            logger.log(_MSG_DUMP_LEVEL, f"  raw_msg[{i}] role={role} content_len={total_len}")
        logger.log(_MSG_DUMP_LEVEL, f"Messages ({len(msgs)}): {' | '.join(msg_summary)}")
        total_chars = 0
        for i, m in enumerate(msgs):
            content = m.get("content", "")
            if isinstance(content, list):
                text = " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
            else:
                text = str(content)
            chars = len(text)
            total_chars += chars
            role = m.get("role", "?")
            preview = text[:80].replace("\n", "\\n")
            logger.log(_MSG_DUMP_LEVEL, f"  msg[{i}] role={role} chars={chars} preview={preview}...")
        logger.log(_MSG_DUMP_LEVEL, f"TOTAL estimated input: {total_chars} chars (actual tokens will be logged after processing)")

    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())

    if stream:
        return StreamingResponse(
            _stream_generate(msgs, media_type, clear_flag, chat_id, created,
                             model_name, temp_image_paths, tools, max_tokens),
            media_type="text/event-stream",
        )
    else:
        loop = asyncio.get_event_loop()
        try:
            full_text, usage = await loop.run_in_executor(
                executor, _collect_generate, msgs, media_type, clear_flag,
                max_tokens
            )
        except Exception as e:
            logger.error(f"Non-stream generate failed: {e}")
            return JSONResponse(
                {"error": {"message": str(e), "type": "server_error"}},
                status_code=500,
            )
        finally:
            _cleanup_temp_files(temp_image_paths)

        # 解析工具调用
        clean_text, tool_calls = _parse_tool_calls_from_text(full_text)

        if tool_calls:
            tool_calls_id = tool_calls[0].get("id", f"call_{uuid.uuid4().hex[:24]}")
            message_obj = {
                "role": "assistant",
                "content": clean_text if clean_text else None,
                "tool_calls": tool_calls,
            }
            finish_reason = "tool_calls" if tool_calls else "stop"
            choice = {
                "index": 0,
                "message": message_obj,
                "finish_reason": finish_reason,
            }
        else:
            message_obj = {"role": "assistant", "content": full_text}
            choice = {
                "index": 0,
                "message": message_obj,
                "finish_reason": infer_finish_reason(
                    pipeline, usage["completion_tokens"], max_tokens),
            }

        response = {
            "id": chat_id,
            "object": "chat.completion",
            "created": created,
            "model": model_name,
            "choices": [choice],
            "usage": usage,
        }
        if benchmark:
            elapsed_ms = float(last_request_timing.get("request_ms", 0.0))
            ttft_ms = last_request_timing.get("ttft_internal_ms")
            prepared = prepare_prompt(pipeline.tokenizer, prompt_slots)
            diagnostics = no_cache_diagnostics(prepared, pipeline, elapsed_ms)
            diagnostics.update({
                "ttft_internal_ms": ttft_ms,
                "max_tokens": max_tokens,
                "stop_reason": choice["finish_reason"],
                "seqlen": int(getattr(pipeline.model, "SEQLEN", 0)),
                "remaining_seqlen": max(0, int(getattr(
                    pipeline.model, "SEQLEN", 0)) - int(getattr(
                        pipeline.model, "history_length", 0))),
                "output_token_ids_sha256": output_token_hash(pipeline),
            })
            response["benchmark"] = diagnostics
        return JSONResponse(response)


def _current_usage():
    prompt_tokens = max(0, int(getattr(pipeline, "last_input_token_count", 0)))
    completion_tokens = max(
        0, int(getattr(pipeline, "last_output_token_count", 0)))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _collect_generate(msgs, media_type, clear_flag, max_tokens=512):
    global last_request_timing
    full_text = ""
    started = time.perf_counter()
    ttft_ms = None
    try:
        with completion_limit(pipeline, max_tokens):
            for text_chunk in pipeline.generate(
                    msgs, media_type, clear_history_flag=clear_flag):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - started) * 1000
                full_text += text_chunk
        logger.info(f"LLM_RAW_OUTPUT ({len(full_text)} chars): [{full_text}]")
        usage = _current_usage()
        logger.info(
            "Chat done: %d chars, prompt_tokens=%d completion_tokens=%d",
            len(full_text), usage["prompt_tokens"], usage["completion_tokens"])
    except Exception as e:
        logger.error(f"Generate error: {e}")
        raise
    last_request_timing = {
        "request_ms": (time.perf_counter() - started) * 1000,
        "ttft_internal_ms": ttft_ms,
    }
    return full_text, usage


def _cleanup_temp_files(paths):
    """Delete temporary files, ignoring any errors."""
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.unlink(p)
            except OSError:
                pass


async def _stream_generate(msgs, media_type, clear_flag, chat_id, created,
                           model_name, temp_image_paths, tools=None,
                           max_tokens=512):
    queue = asyncio.Queue()

    def run_generate():
        try:
            chars = 0
            raw_chunks = []
            with completion_limit(pipeline, max_tokens):
                gen = pipeline.generate(
                    msgs, media_type, clear_history_flag=clear_flag)
                for text_chunk in gen:
                    if text_chunk is not None and text_chunk != "":
                        chars += len(text_chunk)
                        raw_chunks.append(text_chunk)
                        asyncio.run_coroutine_threadsafe(
                            queue.put(("chunk", text_chunk)), loop)
            logger.info(f"LLM_RAW_OUTPUT ({chars} chars): [{''.join(raw_chunks)}]")
            usage = _current_usage()
            logger.info(
                "Stream done: %d chars, prompt_tokens=%d completion_tokens=%d",
                chars, usage["prompt_tokens"], usage["completion_tokens"])
            asyncio.run_coroutine_threadsafe(queue.put(("done", usage)), loop)
        except Exception as e:
            logger.error(f"Stream error: {e}")
            asyncio.run_coroutine_threadsafe(queue.put(("error", str(e))), loop)

    loop = asyncio.get_event_loop()
    executor.submit(run_generate)

    try:
        accumulated_text = ""
        usage = None
        while True:
            msg_type, data = await queue.get()
            if msg_type == "done":
                usage = data
                break
            elif msg_type == "error":
                error_chunk = {"error": {"message": data, "type": "server_error"}}
                yield f"data: {json.dumps(error_chunk, ensure_ascii=False)}\n\n"
                break
            elif msg_type == "chunk":
                accumulated_text += data
                chunk = {
                    "id": chat_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": data},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # 流式完成后尝试解析工具调用
        has_tool_calls = False
        if accumulated_text:
            _, tool_calls = _parse_tool_calls_from_text(accumulated_text)
            if tool_calls and tools:
                has_tool_calls = True
                for tc in tool_calls:
                    tc_chunk = {
                        "id": chat_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [{
                                    "index": 0,
                                    "id": tc["id"],
                                    "type": "function",
                                    "function": {
                                        "name": tc["function"]["name"],
                                        "arguments": tc["function"]["arguments"],
                                    },
                                }],
                            },
                            "finish_reason": None,
                        }],
                    }
                    yield f"data: {json.dumps(tc_chunk, ensure_ascii=False)}\n\n"

        final_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "tool_calls" if has_tool_calls else infer_finish_reason(
                    pipeline, (usage or {}).get("completion_tokens", 0), max_tokens),
            }],
            "usage": usage or {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    finally:
        _cleanup_temp_files(temp_image_paths)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--model-path", type=str, default="../Qwen3_5/qwen3.5.bmodel")
    parser.add_argument("--config-path", type=str, default="../Qwen3_5/config")
    parser.add_argument("--module-path", type=str, default="../Qwen3_5/python_demo")
    parser.add_argument("--devid", type=int, default=0)
    parser.add_argument("--video-ratio", type=float, default=0.25)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    app.state.cfg = {
        "model_path": args.model_path,
        "config_path": args.config_path,
        "devid": args.devid,
        "video_ratio": args.video_ratio,
        "max_new_tokens": args.max_new_tokens,
    }
    os.environ["MODULE_PATH"] = args.module_path

    atexit.register(_cleanup_model)

    uvicorn.run(app, host=args.host, port=args.port, log_config={
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"default": {"format": "%(asctime)s [%(levelname)s] %(message)s", "datefmt": "%Y-%m-%d %H:%M:%S"}},
        "handlers": {"default": {"class": "logging.StreamHandler", "formatter": "default", "stream": "ext://sys.stderr"}},
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.access": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    })
