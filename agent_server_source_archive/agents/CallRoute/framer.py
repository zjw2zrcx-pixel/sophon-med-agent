from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List


class FrameType(Enum):
    TEXT = "text"
    POTENTIAL_COMMAND = "potential_command"


@dataclass
class Frame:
    type: FrameType
    content: str
    confidence: float = 1.0


_COMMAND_KEYWORDS_LOWER = {
    "tool_call": True,
    "skill_call": True,
    "judge": True,
}

_COMMAND_KEYWORDS_NORM = {
    "tool_call", "toolcall", "toolcal", "tol_call", "toll_call", "tool",
    "skill_call", "skillcall", "skillcal", "skil_call", "skilcal",
    "judge", "judg", "judgement",
}


def _edit_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _check_keyword(kw_candidate: str) -> bool:
    kw = kw_candidate.lower().replace(' ', '').replace('-', '')
    if kw in _COMMAND_KEYWORDS_NORM:
        return True
    for canon in _COMMAND_KEYWORDS_LOWER:
        if _edit_distance(kw, canon.replace('_', '')) <= 2:
            return True
    return False


class StreamFramer:
    """
    从流式文本中识别和分割"可能的命令块"与"纯文本块"。
    
    关键设计: 逐 token 流式输出时，花括号可能单独作为一个 chunk 到来。
    因此采用延迟决策策略:
    - 遇到 '{' 时，不立即决定是命令还是文本
    - 缓冲后续字符，累积到足够内容后判断
    - 缓冲区超时（长度/字数限制）后强制决策
    - 对于已经确定为文本的 '{'，立即输出
    """

    _BRACE_DETECT_RE = re.compile(r'\{[^{}]*[:=][^{}]*\}', re.IGNORECASE)
    _COMMAND_START_PATTERNS = [
        re.compile(r'\{\s*tool\s*_?\s*call', re.IGNORECASE),
        re.compile(r'\{\s*skill\s*_?\s*call', re.IGNORECASE),
        re.compile(r'\{\s*judge', re.IGNORECASE),
        re.compile(r'\{\s*\w+\s*[:=]"\w', re.IGNORECASE),
    ]

    # 决策阈值: 缓冲多少字符后开始尝试判断
    DECISION_MIN_LENGTH = 5
    # 最大缓冲长度: 超过此长度仍未匹配命令则回退为文本
    MAX_BRACE_BUFFER = 300

    def __init__(
        self,
        max_buffer_length: int = 500,
        command_timeout_chars: int = 300,
    ):
        self.text_buffer = ""
        self.in_command = False
        self.brace_depth = 0
        self.command_buffer = ""
        self.char_count_in_command = 0
        self.max_buffer_length = max_buffer_length
        self.command_timeout_chars = command_timeout_chars
        # 延迟决策缓冲区: 遇到 '{' 后暂时缓冲
        self.brace_candidate = ""
        self.in_brace_candidate = False

    def feed(self, chunk: str) -> List[Frame]:
        frames: List[Frame] = []
        for char in chunk:
            new_frames = self._feed_char(char)
            frames.extend(new_frames)
        return frames

    def _feed_char(self, char: str) -> List[Frame]:
        frames: List[Frame] = []

        if self.in_command:
            return self._handle_command_char(char)
        
        if self.in_brace_candidate:
            return self._handle_brace_candidate_char(char)
        
        if char == '{':
            self.in_brace_candidate = True
            self.brace_candidate = "{"
            return frames
        
        self.text_buffer += char
        if self.text_buffer:
            frames.append(Frame(FrameType.TEXT, self.text_buffer, 1.0))
            self.text_buffer = ""
        return frames

    def _handle_brace_candidate_char(self, char: str) -> List[Frame]:
        frames: List[Frame] = []
        self.brace_candidate += char

        if char == '{':
            # 嵌套花括号 - 可能是 param{
            # 继续缓冲
            pass
        elif char == '}':
            # 可能的命令结束
            # 检查是否是完整命令
            if self._is_complete_command(self.brace_candidate):
                conf = self._estimate_confidence(self.brace_candidate)
                if conf > 0.3:
                    frames.append(Frame(FrameType.POTENTIAL_COMMAND, self.brace_candidate, conf))
                else:
                    frames.append(Frame(FrameType.TEXT, self.brace_candidate, 1.0))
                self.in_brace_candidate = False
                self.brace_candidate = ""
                return frames
            # 否则继续缓冲 (可能只是嵌套的 })
            pass
        
        # 检查是否已经有足够内容判断
        if len(self.brace_candidate) >= self.DECISION_MIN_LENGTH:
            if self._looks_like_command_start(self.brace_candidate):
                # 确认是命令开始，切换到命令模式
                if self.text_buffer:
                    frames.append(Frame(FrameType.TEXT, self.text_buffer, 1.0))
                    self.text_buffer = ""
                
                self.in_command = True
                self.in_brace_candidate = False
                # 计算已有的花括号深度
                self.brace_depth = self.brace_candidate.count('{') - self.brace_candidate.count('}')
                self.command_buffer = self.brace_candidate
                self.char_count_in_command = len(self.brace_candidate)
                self.brace_candidate = ""
                return frames
            elif self._is_definitely_not_command(self.brace_candidate):
                # 确定不是命令，回退为文本
                text = self.text_buffer + self.brace_candidate
                self.text_buffer = ""
                self.in_brace_candidate = False
                self.brace_candidate = ""
                # 逐字符处理文本（可能包含新的 '{'）
                for c in text:
                    new_frames = self._feed_char(c)
                    frames.extend(new_frames)
                return frames
        
        # 缓冲区超时
        if len(self.brace_candidate) > self.MAX_BRACE_BUFFER:
            # 强制回退为文本
            text = self.text_buffer + self.brace_candidate
            frames.append(Frame(FrameType.TEXT, text, 1.0))
            self.text_buffer = ""
            self.in_brace_candidate = False
            self.brace_candidate = ""
            return frames
        
        return frames

    def _handle_command_char(self, char: str) -> List[Frame]:
        frames: List[Frame] = []
        self.command_buffer += char
        self.char_count_in_command += 1

        if char == '{':
            self.brace_depth += 1
        elif char == '}':
            self.brace_depth -= 1
            if self.brace_depth <= 0:
                # 检查后续是否有多余的 }
                # (在 flush 或下层处理)
                conf = self._estimate_confidence(self.command_buffer)
                frames.append(
                    Frame(FrameType.POTENTIAL_COMMAND, self.command_buffer, conf)
                )
                self.in_command = False
                self.command_buffer = ""
                self.brace_depth = 0
                self.char_count_in_command = 0
                return frames

        if self.char_count_in_command > self.command_timeout_chars:
            conf = self._estimate_confidence(self.command_buffer)
            if conf > 0.3:
                frames.append(
                    Frame(FrameType.POTENTIAL_COMMAND, self.command_buffer, conf)
                )
            else:
                self.text_buffer += self.command_buffer
                if self.text_buffer:
                    frames.append(Frame(FrameType.TEXT, self.text_buffer, 1.0))
                    self.text_buffer = ""
            self.in_command = False
            self.command_buffer = ""
            self.brace_depth = 0
            self.char_count_in_command = 0

        return frames

    def flush(self) -> List[Frame]:
        frames: List[Frame] = []
        
        if self.in_command:
            conf = self._estimate_confidence(self.command_buffer)
            if conf > 0.3:
                frames.append(
                    Frame(FrameType.POTENTIAL_COMMAND, self.command_buffer, conf)
                )
            else:
                self.text_buffer += self.command_buffer
        
        if self.in_brace_candidate:
            if self._looks_like_command_start(self.brace_candidate):
                conf = self._estimate_confidence(self.brace_candidate)
                if conf > 0.3:
                    frames.append(
                        Frame(FrameType.POTENTIAL_COMMAND, self.brace_candidate, conf)
                    )
                else:
                    self.text_buffer += self.brace_candidate
            else:
                self.text_buffer += self.brace_candidate
        
        if self.text_buffer:
            frames.append(Frame(FrameType.TEXT, self.text_buffer, 1.0))
        
        return frames

    def reset(self):
        self.text_buffer = ""
        self.in_command = False
        self.brace_depth = 0
        self.command_buffer = ""
        self.char_count_in_command = 0
        self.brace_candidate = ""
        self.in_brace_candidate = False

    def _looks_like_command_start(self, text: str) -> bool:
        text_stripped = text.lstrip('{').lstrip()
        for pattern in self._COMMAND_START_PATTERNS:
            if pattern.search(text):
                return True
        
        # 提取 { 后面的关键词部分
        m = re.match(r'\{\s*(\w[\w_\-]*)', text, re.IGNORECASE)
        if m:
            kw = m.group(1)
            if _check_keyword(kw):
                return True
        
        # 检查是否包含 : 或 = (可能是参数部分)
        if '{' in text and (':' in text or '=' in text):
            return True
        
        return False

    def _is_definitely_not_command(self, text: str) -> bool:
        # 如果缓冲区中只有 { 加少量字符，且不包含任何关键词特征
        # 在很短的缓冲区时不做此判断
        if len(text) < 3:
            return False
        
        # 检查是否包含数字或常见自然语言模式
        text_inner = text.lstrip('{').rstrip('}')
        
        # 如果内部以常见中文标点或自然语言开头，不是命令
        if re.match(r'\{\s*[，。、？！；：""''（）【】]', text):
            return True
        
        # 如果没有 : = " 等命令特征，且已经够长，判定不是命令
        if len(text) >= 8 and ':' not in text and '=' not in text and '"' not in text:
            return True
        
        return False

    def _is_complete_command(self, text: str) -> bool:
        # 检查花括号是否平衡
        depth = 0
        for c in text:
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
        return depth == 0 and '{' in text

    def _estimate_confidence(self, text: str) -> float:
        text_lower = text.lower()
        for kw in _COMMAND_KEYWORDS_LOWER:
            if kw in text_lower or kw.replace('_', '') in text_lower:
                return 0.9
            kw_norm = kw.replace('_', '')
            for i in range(max(0, len(text_lower) - len(kw_norm))):
                if _edit_distance(text_lower[i:i + len(kw_norm)], kw_norm) <= 2:
                    return 0.7
        if '{' in text and ':' in text:
            return 0.4
        return 0.1
