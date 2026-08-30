"""Wire-level helpers for ``remote-audio.v1`` JSON messages."""
from __future__ import annotations

import base64
import binascii
import json
import uuid
from typing import Any, Mapping

PROTOCOL_VERSION = "remote-audio.v1"
OPERATIONS = frozenset({"speak", "record", "speak_and_record"})


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, operation_id: str = ""):
        super().__init__(message)
        self.code = code
        self.operation_id = operation_id


def new_operation_id() -> str:
    return str(uuid.uuid4())


def message(message_type: str, *, operation_id: str = "", **payload: Any) -> dict:
    value = {"protocol": PROTOCOL_VERSION, "type": message_type}
    if operation_id:
        value["operation_id"] = operation_id
    value.update(payload)
    return value


def parse_json(raw: str | bytes | Mapping[str, Any]) -> dict:
    if isinstance(raw, Mapping):
        value = dict(raw)
    else:
        try:
            value = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ProtocolError("INVALID_JSON", f"消息不是有效 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ProtocolError("INVALID_MESSAGE", "消息顶层必须是 JSON object")
    return value


def validate_version(value: Mapping[str, Any]) -> None:
    version = str(value.get("protocol", ""))
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            "UNSUPPORTED_PROTOCOL",
            f"需要 {PROTOCOL_VERSION}，收到 {version or '未声明版本'}",
            str(value.get("operation_id", "")),
        )


def require_operation_id(value: Mapping[str, Any]) -> str:
    operation_id = str(value.get("operation_id", "")).strip()
    if not operation_id:
        raise ProtocolError("MISSING_OPERATION_ID", "操作消息缺少 operation_id")
    try:
        uuid.UUID(operation_id)
    except ValueError as exc:
        raise ProtocolError("INVALID_OPERATION_ID", "operation_id 必须是 UUID", operation_id) from exc
    return operation_id


def decode_audio(value: Any, operation_id: str = "") -> bytes:
    if not isinstance(value, str) or not value:
        raise ProtocolError("INVALID_AUDIO", "音频 data 必须是非空 Base64 字符串", operation_id)
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ProtocolError("INVALID_AUDIO", "音频 data 不是有效 Base64", operation_id) from exc
