"""Remote microphone/speaker operations used by the voice Agent."""

from .operations import (
    AudioFormat,
    RecordingResult,
    RemoteAudioError,
    RemoteAudioOperations,
)
from .protocol import PROTOCOL_VERSION, ProtocolError, new_operation_id

__all__ = [
    "AudioFormat",
    "PROTOCOL_VERSION",
    "ProtocolError",
    "RecordingResult",
    "RemoteAudioError",
    "RemoteAudioOperations",
    "new_operation_id",
]
