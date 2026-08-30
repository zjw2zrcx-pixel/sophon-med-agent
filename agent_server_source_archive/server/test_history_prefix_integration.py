"""Manual integration check for a running qwen3.5 history server."""

import copy
import json
import sys

import httpx


BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8007"


def request(session_id, slots):
    response = httpx.post(
        BASE_URL + "/v1/chat/completions",
        headers={
            "x-session-id": session_id,
            "x-clear-history": "false",
        },
        json={
            "model": "qwen3.5-4b-history",
            "prompt_slots": slots,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main():
    base = {
        "version": "suha.v1",
        "system": "你是一个准确、简洁的中文助手。",
        "user": "17加25等于多少？只回答结果。",
        "history": "[]",
        "attempt": "[]",
    }
    cases = []
    cases.append(("cold", request("prefix-one", base)))
    cases.append(("A", request("prefix-one", base)))

    changed = copy.deepcopy(base)
    changed["attempt"] = '[{"error":"先前格式错误，请只回答数字"}]'
    cases.append(("H", request("prefix-one", changed)))
    changed["history"] = '[{"result":"已确认需要精确计算"}]'
    cases.append(("U", request("prefix-one", changed)))
    changed["user"] = "中国的首都是哪里？只回答城市名。"
    cases.append(("S", request("prefix-one", changed)))

    second = copy.deepcopy(base)
    second["user"] = "9乘以8等于多少？只回答结果。"
    cases.append(("session-two", request("prefix-two", second)))
    third = copy.deepcopy(base)
    third["user"] = "水的化学式是什么？只回答化学式。"
    cases.append(("session-three", request("prefix-three", third)))

    long_prompt = copy.deepcopy(base)
    long_prompt["system"] = (
        "你是本服务部署的Qwen3.5-4B模型。当前BModel编译上下文上限是"
        "8192 token；本HTTP端点暂不接收图片，也没有内置联网或代码执行"
        "能力。不要声称未知的训练截止日期。"
    )
    long_prompt["user"] = (
        "你是什么模型？请用一段较长的中文说明你的能力、限制、"
        "上下文处理方式和回答原则，并严格遵守system中给出的部署事实。"
    )
    cases.append(("long", request("prefix-three", long_prompt)))

    status = httpx.get(BASE_URL + "/status", timeout=10).json()
    print(json.dumps({"cases": cases, "status": status}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
