#!/usr/bin/env python3
"""
OpenAI 兼容服务器全链路测试脚本

使用方法:
  python test_all.py                        # 使用默认配置运行所有测试
  python test_all.py --router-url http://192.168.1.100:8000  # 自定义路由地址
  python test_all.py --skip chat            # 跳过 chat 测试
  python test_all.py --skip asr             # 跳过 asr 测试
  python test_all.py --only router          # 仅测试路由层
  python test_all.py --skip image           # 跳过图像测试
  python test_all.py --skip multirole       # 跳过多角色测试
"""

import argparse
import base64
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve()
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_ROUTER_URL = "http://localhost:8000"

TEST_IMAGE = PROJECT_DIR / "bird.jpg"
TEST_AUDIO = PROJECT_DIR / "test.mp3"

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"
SKIP = "\033[93mSKIP\033[0m"
INFO = "\033[94mINFO\033[0m"


def ts():
    return f"\033[90m{datetime.now().strftime('%H:%M:%S.%f')[:-3]}\033[0m"


def print_header(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name, ok, detail=""):
    status = PASS if ok else FAIL
    msg = f"  {ts()} [{status}] {name}"
    if detail:
        msg += f"  ({detail})"
    print(msg)


# ============================================================
# 1-4: 基础健康检查
# ============================================================

def test_router_health(base_url):
    print_header("1. Router Health Check")
    try:
        resp = httpx.get(f"{base_url}/health", timeout=5)
        ok = resp.status_code == 200 and resp.json().get("status") == "ok"
        print_result("Router /health", ok, f"status={resp.status_code}")
    except Exception as e:
        ok = False
        print_result("Router /health", False, str(e))
    return ok


def test_models_list(base_url):
    print_header("2. Models List")
    try:
        resp = httpx.get(f"{base_url}/v1/models", timeout=10)
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            models = data.get("data", [])
            print(f"  {INFO} Found {len(models)} model(s):")
            for m in models:
                print(f"       - {m['id']:15s} type={m.get('type','?'):5s} status={m.get('status','?')}")
        print_result("GET /v1/models", ok)
        return ok
    except Exception as e:
        print_result("GET /v1/models", False, str(e))
        return False


def test_chat_backend_health(base_url, chat_port=8001):
    print_header("3. Chat Server Health (Direct)")
    try:
        resp = httpx.get(f"http://127.0.0.1:{chat_port}/health", timeout=5)
        ok = resp.status_code == 200
        print_result("Chat server /health", ok, f"status={resp.status_code}")
    except Exception as e:
        ok = False
        print_result("Chat server /health", False, str(e))
    return ok


def test_asr_backend_health(base_url, asr_port=8002):
    print_header("4. ASR Server Health (Direct)")
    try:
        resp = httpx.get(f"http://127.0.0.1:{asr_port}/health", timeout=5)
        ok = resp.status_code == 200
        print_result("ASR server /health", ok, f"status={resp.status_code}")
    except Exception as e:
        ok = False
        print_result("ASR server /health", False, str(e))
    return ok


# ============================================================
# 5-6: 基础文本对话
# ============================================================

def test_chat_text(base_url, model="qwen3.5"):
    print_header("5. Chat Completion - Text Only")
    try:
        print(f"  {ts()} Sending request...")
        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "请用一句话介绍你自己。"}],
                "stream": False,
            },
            timeout=120,
        )
        elapsed = time.time() - start
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            has_response = "id" in data and len(content) > 0
            print(f"  {INFO} Response: {content[:100]}{'...' if len(content) > 100 else ''}")
            print(f"  {INFO} Time: {elapsed:.2f}s")
            print_result("Chat text (non-stream)", has_response, f"id={data.get('id','?')}")
            return has_response
        else:
            print_result("Chat text (non-stream)", False, f"status={resp.status_code} body={resp.text[:200]}")
            return False
    except Exception as e:
        print_result("Chat text (non-stream)", False, str(e))
        return False


def test_chat_stream(base_url, model="qwen3.5"):
    print_header("6. Chat Completion - Streaming")
    try:
        print(f"  {ts()} Sending stream request...")
        start = time.time()
        collected = ""
        chunk_count = 0
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "请用一句话描述春天。"}],
                "stream": True,
            },
            timeout=120,
        ) as resp:
            ok = resp.status_code == 200
            if not ok:
                print_result("Chat text (stream)", False, f"status={resp.status_code}")
                return False

            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "error" in chunk:
                            print(f"  {FAIL} Stream error: {chunk['error']}")
                            return False
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected += content
                            chunk_count += 1
                    except json.JSONDecodeError:
                        pass

        elapsed = time.time() - start
        has_response = len(collected) > 0 and chunk_count > 0
        print(f"  {INFO} Streamed {chunk_count} chunks, {len(collected)} chars")
        print(f"  {INFO} Response: {collected[:100]}{'...' if len(collected) > 100 else ''}")
        print(f"  {INFO} Time: {elapsed:.2f}s")
        print_result("Chat text (stream)", has_response)
        return has_response
    except Exception as e:
        print_result("Chat text (stream)", False, str(e))
        return False


# ============================================================
# 7: 图像测试
# ============================================================

def test_chat_image(base_url, model="qwen3.5", image_path=None):
    print_header("7. Chat Completion - Image + Text")

    if image_path is None:
        image_path = TEST_IMAGE

    if not image_path.exists():
        print(f"  {SKIP} Test image not found: {image_path}")
        return None

    try:
        print(f"  {ts()} Sending image request...")
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()

        ext = image_path.suffix.lstrip(".")
        mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}
        mime = mime_map.get(ext, "image/jpeg")

        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "请用一句话描述这张图片的内容。"},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                    ],
                }],
                "stream": False,
            },
            timeout=300,
        )
        elapsed = time.time() - start

        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            has_response = len(content) > 0
            print(f"  {INFO} Response: {content[:150]}{'...' if len(content) > 150 else ''}")
            print(f"  {INFO} Time: {elapsed:.2f}s")
            print_result("Chat image (non-stream)", has_response, f"image={image_path.name}")
            return has_response
        else:
            print_result("Chat image (non-stream)", False, f"status={resp.status_code} body={resp.text[:200]}")
            return False
    except Exception as e:
        print_result("Chat image (non-stream)", False, str(e))
        return False


# ============================================================
# 8: 会话续接 (旧版兼容 - 只有单条 user)
# ============================================================

def test_chat_session(base_url, model="qwen3.5"):
    print_header("8. Chat Session (x-session-id)")
    session_id = "test-session-001"
    try:
        print(f"  {ts()} Session round 1...")
        resp1 = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "请记住数字42，用一句话回复。"}],
                "stream": False,
            },
            headers={"x-session-id": session_id},
            timeout=120,
        )
        ok1 = resp1.status_code == 200
        if ok1:
            content1 = resp1.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  {INFO} Session round 1 response: {content1[:80]}...")
        else:
            print(f"  {FAIL} Session round 1 failed: status={resp1.status_code}")
            return False

        resp2 = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "我刚才让你记住什么数字？请用一句话回答。"}],
                "stream": False,
            },
            headers={"x-session-id": session_id},
            timeout=120,
        )
        ok2 = resp2.status_code == 200
        if ok2:
            content2 = resp2.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            print(f"  {INFO} Session round 2 response: {content2[:80]}...")
        else:
            print(f"  {FAIL} Session round 2 failed: status={resp2.status_code}")
            return False

        print_result("Chat session", ok1 and ok2, f"session_id={session_id}")
        return ok1 and ok2
    except Exception as e:
        print_result("Chat session", False, str(e))
        return False


# ============================================================
# 9: 多角色消息 - system prompt
# ============================================================

def test_chat_system_prompt(base_url, model="qwen3.5"):
    print_header("9. Multi-Role - System Prompt")
    try:
        print(f"  {ts()} Sending system + user messages...")
        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个数学助手。只回答数字，不要说任何其他话。"},
                    {"role": "user", "content": "1+1等于几？请用一句话回答。"},
                ],
                "stream": False,
            },
            headers={"x-session-id": f"test-system-{int(time.time())}", "x-clear-history": "true"},
            timeout=120,
        )
        elapsed = time.time() - start
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            has_response = len(content) > 0
            print(f"  {INFO} Response: {content[:100]}")
            print(f"  {INFO} Time: {elapsed:.2f}s")
            print_result("System + User messages", has_response)
        else:
            print_result("System + User messages", False, f"status={resp.status_code} body={resp.text[:200]}")
            return False
    except Exception as e:
        print_result("System + User messages", False, str(e))
        return False


# ============================================================
# 10: 多角色消息 - 多轮对话 (system + user + assistant + user)
# ============================================================

def test_chat_multi_turn(base_url, model="qwen3.5"):
    print_header("10. Multi-Role - Multi-Turn Conversation")
    try:
        print(f"  {ts()} Sending multi-turn messages...")
        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个知识问答助手。请用一句话回答问题。"},
                    {"role": "user", "content": "中国的首都是哪里？"},
                    {"role": "assistant", "content": "中国的首都是北京。"},
                    {"role": "user", "content": "它有多少人口？请用一句话回答。"},
                ],
                "stream": False,
            },
            headers={"x-session-id": f"test-multiturn-{int(time.time())}", "x-clear-history": "true"},
            timeout=120,
        )
        elapsed = time.time() - start
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            has_response = len(content) > 0
            print(f"  {INFO} Response: {content[:150]}{'...' if len(content) > 150 else ''}")
            print(f"  {INFO} Time: {elapsed:.2f}s")
            print_result("Multi-turn conversation", has_response)
        else:
            body = resp.text[:300] if resp.text else "no body"
            print_result("Multi-turn conversation", False, f"status={resp.status_code} body={body}")
            return False
    except Exception as e:
        print_result("Multi-turn conversation", False, str(e))
        return False


# ============================================================
# 11: 多角色消息 - tool_result 消息
# ============================================================

def test_chat_tool_result(base_url, model="qwen3.5"):
    print_header("11. Multi-Role - Tool Result")
    try:
        print(f"  {ts()} Sending conversation with tool_result...")
        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是机器人助手。根据工具结果继续对话，请用一句话回复。"},
                    {"role": "user", "content": "帮我找到厨房"},
                    {"role": "assistant", "content": '{tool_call:"navigate_to" param{target:"厨房"}}'},
                    {"role": "user", "content": "[工具结果: navigate_to]\n正在导航到厨房..."},
                    {"role": "user", "content": "好的，已经到达厨房了"},
                ],
                "stream": False,
            },
            headers={"x-session-id": f"test-tool-{int(time.time())}", "x-clear-history": "true"},
            timeout=120,
        )
        elapsed = time.time() - start
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            has_response = len(content) > 0
            print(f"  {INFO} Response: {content[:150]}{'...' if len(content) > 150 else ''}")
            print(f"  {INFO} Time: {elapsed:.2f}s")
            print_result("Tool result conversation", has_response)
        else:
            body = resp.text[:300] if resp.text else "no body"
            print_result("Tool result conversation", False, f"status={resp.status_code} body={body}")
            return False
    except Exception as e:
        print_result("Tool result conversation", False, str(e))
        return False


# ============================================================
# 12: 多角色消息 - 命令格式指令遵循 (核心测试)
# ============================================================

def test_chat_command_format(base_url, model="qwen3.5"):
    print_header("12. Multi-Role - Command Format Compliance")
    try:
        print(f"  {ts()} Testing command format output...")
        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是机器人助手。用户要求导航时，用工具调用格式回复。\n格式: {tool_call:\"工具名\" param{参数名:\"参数值\"}}\n\n可用工具:\n- navigate_to: 导航到位置，参数 target\n- describe_scene: 描述场景，无参数\n\n示例:\n用户: 去厨房\n助手: {tool_call:\"navigate_to\" param{target:\"厨房\"}}"},
                    {"role": "user", "content": "去客厅"},
                ],
                "stream": False,
            },
            headers={"x-session-id": f"test-cmd-{int(time.time())}", "x-clear-history": "true"},
            timeout=120,
        )
        elapsed = time.time() - start
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            has_tool_call = "{tool_call" in content or "{tool" in content.lower()
            print(f"  {INFO} Response: {content[:200]}")
            print(f"  {INFO} Contains tool_call: {has_tool_call}")
            print(f"  {INFO} Time: {elapsed:.2f}s")
            print_result("Command format compliance", ok, f"has_tool_call={has_tool_call}")
        else:
            body = resp.text[:300] if resp.text else "no body"
            print_result("Command format compliance", False, f"status={resp.status_code} body={body}")
            return False
    except Exception as e:
        print_result("Command format compliance", False, str(e))
        return False


# ============================================================
# 13: OpenAI tools 字段 - 非流式
# ============================================================

def test_chat_tools_non_stream(base_url, model="qwen3.5"):
    print_header("13. OpenAI tools - Non-Stream")
    try:
        print(f"  {ts()} Sending request with tools field...")
        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个助手。请用一句话回答。"},
                    {"role": "user", "content": "北京今天天气怎么样？"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "获取指定城市的天气信息",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "城市名称"},
                                },
                                "required": ["city"],
                            },
                        },
                    },
                ],
                "stream": False,
            },
            headers={"x-session-id": f"test-tools-ns-{int(time.time())}", "x-clear-history": "true"},
            timeout=120,
        )
        elapsed = time.time() - start
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", [])
            finish_reason = choice.get("finish_reason", "")

            has_tool_calls = len(tool_calls) > 0
            has_content = len(content) > 0
            if has_tool_calls:
                tc = tool_calls[0]
                func = tc.get("function", {})
                print(f"  {INFO} Tool call: name={func.get('name', '?')} args={func.get('arguments', '?')[:100]}")
            if has_content:
                print(f"  {INFO} Content: {content[:150]}")
            print(f"  {INFO} finish_reason={finish_reason} tool_calls={len(tool_calls)} content_len={len(content)}")
            print_result("Tools non-stream (request accepted)", ok, f"finish_reason={finish_reason} tool_calls={len(tool_calls)}")
            return ok
        else:
            body = resp.text[:300] if resp.text else "no body"
            print_result("Tools non-stream", False, f"status={resp.status_code} body={body}")
            return False
    except Exception as e:
        print_result("Tools non-stream", False, str(e))
        return False


# ============================================================
# 14: OpenAI tools 字段 - 流式
# ============================================================

def test_chat_tools_stream(base_url, model="qwen3.5"):
    print_header("14. OpenAI tools - Stream")
    try:
        print(f"  {ts()} Sending stream request with tools field...")
        start = time.time()
        collected = ""
        chunk_count = 0
        tool_calls_found = []
        finish_reason = None
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个助手。请用一句话回答。"},
                    {"role": "user", "content": "上海明天会下雨吗？"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "获取指定城市的天气信息",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "城市名称"},
                                },
                                "required": ["city"],
                            },
                        },
                    },
                ],
                "stream": True,
            },
            headers={"x-session-id": f"test-tools-s-{int(time.time())}", "x-clear-history": "true"},
            timeout=120,
        ) as resp:
            ok = resp.status_code == 200
            if not ok:
                print_result("Tools stream", False, f"status={resp.status_code}")
                return False

            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "error" in chunk:
                            print(f"  {FAIL} Stream error: {chunk['error']}")
                            return False
                        choice = chunk.get("choices", [{}])[0]
                        delta = choice.get("delta", {})
                        content = delta.get("content", "")
                        if content:
                            collected += content
                            chunk_count += 1
                        tc_deltas = delta.get("tool_calls", [])
                        for tc_delta in tc_deltas:
                            func = tc_delta.get("function", {})
                            if func.get("name"):
                                tool_calls_found.append(func["name"])
                        fr = choice.get("finish_reason")
                        if fr:
                            finish_reason = fr
                    except json.JSONDecodeError:
                        pass

        elapsed = time.time() - start
        has_content = len(collected) > 0
        print(f"  {INFO} Streamed {chunk_count} chunks, {len(collected)} chars, finish_reason={finish_reason}")
        if tool_calls_found:
            print(f"  {INFO} Tool calls found: {tool_calls_found}")
        if has_content:
            print(f"  {INFO} Content: {collected[:100]}")
        print(f"  {INFO} Time: {elapsed:.2f}s")
        print_result("Tools stream (request accepted)", True, f"finish_reason={finish_reason} tool_calls={len(tool_calls_found)}")
        return True
    except Exception as e:
        print_result("Tools stream", False, str(e))
        return False


# ============================================================
# 15: OpenAI tools 字段 - 带历史 tool_calls + tool 结果
# ============================================================

def test_chat_tools_with_history(base_url, model="qwen3.5"):
    print_header("15. OpenAI tools - With Tool Call History")
    try:
        print(f"  {ts()} Sending request with tool_calls + tool result history...")
        start = time.time()
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个天气助手。请用一句话回答。"},
                    {"role": "user", "content": "北京天气怎么样？"},
                    {"role": "assistant", "content": None, "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "北京"}',
                            },
                        },
                    ]},
                    {"role": "tool", "tool_call_id": "call_abc123", "content": "北京今天晴，气温25°C"},
                    {"role": "user", "content": "谢谢，那需要带伞吗？请用一句话回答。"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "获取指定城市的天气信息",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "string", "description": "城市名称"},
                                },
                                "required": ["city"],
                            },
                        },
                    },
                ],
                "stream": False,
            },
            headers={"x-session-id": f"test-tools-hist-{int(time.time())}", "x-clear-history": "true"},
            timeout=120,
        )
        elapsed = time.time() - start
        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            choice = data.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content", "") or ""
            tool_calls = message.get("tool_calls", [])
            finish_reason = choice.get("finish_reason", "")

            if tool_calls:
                tc = tool_calls[0]
                func = tc.get("function", {})
                print(f"  {INFO} Tool call: name={func.get('name', '?')} args={func.get('arguments', '?')[:100]}")
            if content:
                print(f"  {INFO} Content: {content[:150]}")
            print(f"  {INFO} finish_reason={finish_reason} tool_calls={len(tool_calls)} content_len={len(content)}")
            print_result("Tools with history (request accepted)", ok, f"finish_reason={finish_reason} tool_calls={len(tool_calls)}")
            return ok
        else:
            body = resp.text[:300] if resp.text else "no body"
            print_result("Tools with history", False, f"status={resp.status_code} body={body}")
            return False
    except Exception as e:
        print_result("Tools with history", False, str(e))
        return False


# ============================================================
# 16-17: ASR 测试
# ============================================================

def test_asr(base_url, model="qwen3-asr", audio_path=None):
    print_header("16. Audio Transcription")

    if audio_path is None:
        audio_path = TEST_AUDIO

    if not audio_path.exists():
        print(f"  {SKIP} Test audio not found: {audio_path}")
        return None

    try:
        print(f"  {ts()} Sending audio request...")
        start = time.time()
        with open(audio_path, "rb") as f:
            resp = httpx.post(
                f"{base_url}/v1/audio/transcriptions",
                files={"file": (audio_path.name, f, "audio/mpeg")},
                data={"model": model},
                timeout=300,
            )
        elapsed = time.time() - start

        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            text = data.get("text", "")
            has_text = len(text) > 0
            print(f"  {INFO} Transcription: {text[:150]}{'...' if len(text) > 150 else ''}")
            print(f"  {INFO} Time: {elapsed:.2f}s")
            print_result("ASR transcription", has_text, f"audio={audio_path.name}")
            return has_text
        else:
            print_result("ASR transcription", False, f"status={resp.status_code} body={resp.text[:200]}")
            return False
    except Exception as e:
        print_result("ASR transcription", False, str(e))
        return False


def test_asr_with_language(base_url, model="qwen3-asr", audio_path=None, language="zh"):
    print_header("17. Audio Transcription - With Language")

    if audio_path is None:
        audio_path = TEST_AUDIO

    if not audio_path.exists():
        print(f"  {SKIP} Test audio not found: {audio_path}")
        return None

    try:
        print(f"  {ts()} Sending audio request (language={language})...")
        start = time.time()
        with open(audio_path, "rb") as f:
            resp = httpx.post(
                f"{base_url}/v1/audio/transcriptions",
                files={"file": (audio_path.name, f, "audio/mpeg")},
                data={"model": model, "language": language},
                timeout=300,
            )
        elapsed = time.time() - start

        ok = resp.status_code == 200
        if ok:
            data = resp.json()
            text = data.get("text", "")
            has_text = len(text) > 0
            print(f"  {INFO} Transcription (lang={language}): {text[:150]}{'...' if len(text) > 150 else ''}")
            print(f"  {INFO} Time: {elapsed:.2f}s")
            print_result("ASR transcription (with language)", has_text, f"language={language}")
            return has_text
        else:
            print_result("ASR transcription (with language)", False, f"status={resp.status_code}")
            return False
    except Exception as e:
        print_result("ASR transcription (with language)", False, str(e))
        return False


# ============================================================
# 15: 错误处理
# ============================================================

def test_error_handling(base_url):
    print_header("18. Error Handling")

    results = []

    # Test nonexistent model
    try:
        resp = httpx.post(
            f"{base_url}/v1/chat/completions",
            json={"model": "nonexistent-model", "messages": [{"role": "user", "content": "hi"}]},
            timeout=10,
        )
        ok = resp.status_code >= 400
        results.append(("Nonexistent model", ok, f"status={resp.status_code}"))
        print_result("Nonexistent model returns error", ok, f"status={resp.status_code}")
    except Exception as e:
        results.append(("Nonexistent model", False, str(e)))
        print_result("Nonexistent model returns error", False, str(e))

    # Test nonexistent audio model
    try:
        with open(TEST_AUDIO, "rb") as f:
            resp = httpx.post(
                f"{base_url}/v1/audio/transcriptions",
                files={"file": ("test.mp3", f, "audio/mpeg")},
                data={"model": "nonexistent-asr"},
                timeout=10,
            )
        ok = resp.status_code >= 400
        results.append(("Nonexistent ASR model", ok, f"status={resp.status_code}"))
        print_result("Nonexistent ASR model returns error", ok, f"status={resp.status_code}")
    except Exception as e:
        results.append(("Nonexistent ASR model", False, str(e)))
        print_result("Nonexistent ASR model returns error", False, str(e))

    return all(r[1] for r in results)


# ============================================================
# 16-17: OpenAI 客户端兼容性
# ============================================================

def test_openai_client_compatibility(base_url, model="qwen3.5"):
    print_header("19. OpenAI Python Client Compatibility")
    try:
        from openai import OpenAI
        client = OpenAI(api_key="test-key", base_url=f"{base_url}/v1")

        print(f"  {ts()} Sending request via openai client...")
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "1+1等于几？只回答数字"}],
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        has_response = len(content) > 0
        print(f"  {INFO} Response: {content[:80]}")
        print(f"  {INFO} Time: {elapsed:.2f}s")
        print_result("openai.ChatCompletion.create", has_response)
        return has_response
    except ImportError:
        print(f"  {SKIP} openai package not installed, skip this test")
        return None
    except Exception as e:
        print_result("openai.ChatCompletion.create", False, str(e))
        return False


def test_openai_stream_compatibility(base_url, model="qwen3.5"):
    print_header("20. OpenAI Python Client - Stream Compatibility")
    try:
        from openai import OpenAI
        client = OpenAI(api_key="test-key", base_url=f"{base_url}/v1")

        print(f"  {ts()} Sending stream request via openai client...")
        start = time.time()
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "说一个字"}],
            stream=True,
        )
        collected = ""
        chunk_count = 0
        for chunk in stream:
            if chunk.choices[0].delta.content:
                collected += chunk.choices[0].delta.content
                chunk_count += 1
        elapsed = time.time() - start

        has_response = len(collected) > 0
        print(f"  {INFO} Streamed {chunk_count} chunks, {len(collected)} chars")
        print(f"  {INFO} Time: {elapsed:.2f}s")
        print_result("openai.ChatCompletion.create (stream)", has_response)
        return has_response
    except ImportError:
        print(f"  {SKIP} openai package not installed, skip this test")
        return None
    except Exception as e:
        print_result("openai.ChatCompletion.create (stream)", False, str(e))
        return False


def test_openai_audio_compatibility(base_url, model="qwen3-asr", audio_path=None):
    print_header("21. OpenAI Python Client - Audio Compatibility")
    if audio_path is None:
        audio_path = TEST_AUDIO
    if not audio_path.exists():
        print(f"  {SKIP} Test audio not found: {audio_path}")
        return None

    try:
        from openai import OpenAI
        client = OpenAI(api_key="test-key", base_url=f"{base_url}/v1")

        print(f"  {ts()} Sending audio request via openai client...")
        start = time.time()
        with open(audio_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model=model,
                file=f,
            )
        elapsed = time.time() - start

        has_text = len(transcript.text) > 0
        print(f"  {INFO} Transcription: {transcript.text[:150]}{'...' if len(transcript.text) > 150 else ''}")
        print(f"  {INFO} Time: {elapsed:.2f}s")
        print_result("openai.Audio.transcriptions.create", has_text)
        return has_text
    except ImportError:
        print(f"  {SKIP} openai package not installed, skip this test")
        return None
    except Exception as e:
        print_result("openai.Audio.transcriptions.create", False, str(e))
        return False


# ============================================================
# 19: OpenAI 客户端 - 多角色消息
# ============================================================

def test_openai_multirole(base_url, model="qwen3.5"):
    print_header("22. OpenAI Client - Multi-Role Messages")
    try:
        from openai import OpenAI
        client = OpenAI(api_key="test-key", base_url=f"{base_url}/v1")

        print(f"  {ts()} Sending system + user messages via openai client...")
        start = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个数学助手。只回答数字。"},
                {"role": "user", "content": "2+3等于几？"},
            ],
        )
        elapsed = time.time() - start
        content = response.choices[0].message.content
        has_response = len(content) > 0
        print(f"  {INFO} Response: {content[:80]}")
        print(f"  {INFO} Time: {elapsed:.2f}s")
        print_result("openai multi-role (system+user)", has_response)
        return has_response
    except ImportError:
        print(f"  {SKIP} openai package not installed, skip this test")
        return None
    except Exception as e:
        print_result("openai multi-role", False, str(e))
        return False


# ============================================================
# 23: 性能测试 - 首字延迟 & 输出速度
# ============================================================

def test_ttft(base_url, model="qwen3.5", prompt="请详细介绍中国古代四大发明，每个发明写一段话。"):
    print_header("23. Performance - TTFT (Time To First Token)")
    try:
        print(f"  {ts()} Sending stream request for TTFT measurement...")
        start = time.time()
        first_token_time = None
        chunk_count = 0
        collected = ""
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=120,
        ) as resp:
            ok = resp.status_code == 200
            if not ok:
                print_result("TTFT", False, f"status={resp.status_code}")
                return False

            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "error" in chunk:
                            print(f"  {FAIL} Stream error: {chunk['error']}")
                            return False
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content and first_token_time is None:
                            first_token_time = time.time()
                        if content:
                            collected += content
                            chunk_count += 1
                    except json.JSONDecodeError:
                        pass

        total_time = time.time() - start
        if first_token_time is None:
            print_result("TTFT", False, "no content token received")
            return False

        ttft = first_token_time - start
        print(f"  {INFO} TTFT: {ttft:.3f}s")
        print(f"  {INFO} Total time: {total_time:.2f}s")
        print(f"  {INFO} Output: {len(collected)} chars, {chunk_count} chunks")
        print_result("TTFT", ttft < 30, f"ttft={ttft:.3f}s")
        return ttft
    except Exception as e:
        print_result("TTFT", False, str(e))
        return False


def test_throughput(base_url, model="qwen3.5", prompt="请写一篇500字的短文，主题是：人工智能的未来。"):
    print_header("24. Performance - Output Throughput (chars/s)")
    try:
        print(f"  {ts()} Sending stream request for throughput measurement...")
        start = time.time()
        first_token_time = None
        chunk_count = 0
        collected = ""
        last_content_time = None
        with httpx.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
            timeout=180,
        ) as resp:
            ok = resp.status_code == 200
            if not ok:
                print_result("Throughput", False, f"status={resp.status_code}")
                return False

            for line in resp.iter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        if "error" in chunk:
                            print(f"  {FAIL} Stream error: {chunk['error']}")
                            return False
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content", "")
                        if content and first_token_time is None:
                            first_token_time = time.time()
                        if content:
                            collected += content
                            chunk_count += 1
                            last_content_time = time.time()
                    except json.JSONDecodeError:
                        pass

        total_time = time.time() - start
        if first_token_time is None or len(collected) == 0:
            print_result("Throughput", False, "no output received")
            return False

        generation_time = (last_content_time or time.time()) - first_token_time
        chars_per_sec = len(collected) / generation_time if generation_time > 0 else 0
        ttft = first_token_time - start

        print(f"  {INFO} TTFT: {ttft:.3f}s")
        print(f"  {INFO} Generation time (first->last token): {generation_time:.2f}s")
        print(f"  {INFO} Total time: {total_time:.2f}s")
        print(f"  {INFO} Output: {len(collected)} chars, {chunk_count} chunks")
        print(f"  {INFO} Throughput: {chars_per_sec:.1f} chars/s")
        print_result("Throughput", chars_per_sec > 0, f"{chars_per_sec:.1f} chars/s")
        return {"ttft": ttft, "generation_time": generation_time, "chars": len(collected), "chars_per_sec": chars_per_sec}
    except Exception as e:
        print_result("Throughput", False, str(e))
        return False


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Test OpenAI-compatible model servers")
    parser.add_argument("--router-url", default=DEFAULT_ROUTER_URL, help="Router URL")
    parser.add_argument("--skip", nargs="*", default=[],
                         choices=["chat", "asr", "session", "image", "openai", "multirole", "tools", "perf"],
                         help="Skip specified test categories")
    parser.add_argument("--only", choices=["router", "chat", "asr", "multirole", "tools", "perf", "all"], default="all",
                        help="Run only specified test category")
    parser.add_argument("--image", type=str, default=None, help="Path to test image file")
    parser.add_argument("--audio", type=str, default=None, help="Path to test audio file")
    parser.add_argument("--chat-model", default="qwen3.5", help="Chat model name")
    parser.add_argument("--asr-model", default="qwen3-asr", help="ASR model name")
    args = parser.parse_args()

    base_url = args.router_url.rstrip("/")

    results = {}

    # --- Router tests ---
    if args.only in ("router", "all"):
        results["router_health"] = test_router_health(base_url)
        results["models_list"] = test_models_list(base_url)

        if args.only == "all":
            results["chat_backend_health"] = test_chat_backend_health(base_url)
            results["asr_backend_health"] = test_asr_backend_health(base_url)

    # --- Chat tests ---
    if args.only in ("chat", "all") and "chat" not in args.skip:
        results["chat_text"] = test_chat_text(base_url, model=args.chat_model)
        results["chat_stream"] = test_chat_stream(base_url, model=args.chat_model)

        if "image" not in args.skip:
            img = Path(args.image) if args.image else TEST_IMAGE
            results["chat_image"] = test_chat_image(base_url, model=args.chat_model, image_path=img)

        if "session" not in args.skip:
            results["chat_session"] = test_chat_session(base_url, model=args.chat_model)

    # --- Multi-role tests ---
    if args.only in ("multirole", "all") and "multirole" not in args.skip:
        results["chat_system_prompt"] = test_chat_system_prompt(base_url, model=args.chat_model)
        results["chat_multi_turn"] = test_chat_multi_turn(base_url, model=args.chat_model)
        results["chat_tool_result"] = test_chat_tool_result(base_url, model=args.chat_model)
        results["chat_command_format"] = test_chat_command_format(base_url, model=args.chat_model)

    # --- Tool call tests ---
    if args.only in ("tools", "all") and "tools" not in args.skip:
        results["tools_non_stream"] = test_chat_tools_non_stream(base_url, model=args.chat_model)
        results["tools_stream"] = test_chat_tools_stream(base_url, model=args.chat_model)
        results["tools_with_history"] = test_chat_tools_with_history(base_url, model=args.chat_model)

    # --- ASR tests ---
    if args.only in ("asr", "all") and "asr" not in args.skip:
        audio = Path(args.audio) if args.audio else TEST_AUDIO
        results["asr"] = test_asr(base_url, model=args.asr_model, audio_path=audio)
        results["asr_language"] = test_asr_with_language(base_url, model=args.asr_model, audio_path=audio)

    # --- Error handling ---
    if args.only in ("all", "router"):
        results["error_handling"] = test_error_handling(base_url)

    # --- OpenAI client compatibility ---
    if args.only in ("chat", "all") and "openai" not in args.skip:
        results["openai_chat"] = test_openai_client_compatibility(base_url, model=args.chat_model)
        results["openai_stream"] = test_openai_stream_compatibility(base_url, model=args.chat_model)

    if args.only in ("asr", "all") and "openai" not in args.skip:
        audio = Path(args.audio) if args.audio else TEST_AUDIO
        results["openai_audio"] = test_openai_audio_compatibility(base_url, model=args.asr_model, audio_path=audio)

    if args.only in ("multirole", "all") and "openai" not in args.skip:
        results["openai_multirole"] = test_openai_multirole(base_url, model=args.chat_model)

    if args.only in ("perf", "all") and "perf" not in args.skip:
        results["ttft"] = bool(test_ttft(base_url, model=args.chat_model))
        results["throughput"] = bool(test_throughput(base_url, model=args.chat_model))

    # --- Summary ---
    print_header("Test Summary")
    passed = 0
    failed = 0
    skipped = 0
    for name, ok in results.items():
        if ok is None:
            skipped += 1
            status_str = SKIP
        elif ok:
            passed += 1
            status_str = PASS
        else:
            failed += 1
            status_str = FAIL
        print(f"  [{status_str}] {name}")

    print(f"\n  Total: {passed} passed, {failed} failed, {skipped} skipped")
    if failed > 0:
        print(f"\n  {FAIL} Some tests failed!")
        sys.exit(1)
    else:
        print(f"\n  {PASS} All tests passed!")


if __name__ == "__main__":
    main()