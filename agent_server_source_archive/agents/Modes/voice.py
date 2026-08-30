"""Voice mode — always-listening hotword wake-up, like Siri."""
from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import Optional

from .base import ModeBase, LoopResult
from .prompt_config import VOICE_SYSTEM_PROMPT
from ..CallRoute.parser.parser import Command
from ..medical_policy import grounded_medical_speech

logger = logging.getLogger(__name__)


class VoiceMode(ModeBase):
    """Siri-like wake-word voice interaction mode.

    Listens continuously for the hotword "朋友", acknowledges with a beep,
    captures the following utterance and handles it with one unified prompt.
    """

    name = "Voice"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._voice_server = None
        self._server_task: Optional[asyncio.Task] = None
        self._running = False
        self._host = "0.0.0.0"
        self._port = 8766
        self._uvicorn_server = None
        self._pending_utterance: Optional[str] = None

    def get_system_prompt(self) -> str:
        return VOICE_SYSTEM_PROMPT
        return (
            "## 最高优先级：先行动，后播报\n"
            "用户用命令口吻要求机器人移动时，普通文字不能完成任务，必须调用 navigate。\n"
            "判断导航意图时优先看动作表达，不要只等用户说出“导航”二字。"
            "出现“带我去/到、引导我去/到、引导导我去/到、领我去/到、送我去/到、"
            "陪我去/到、带路去/到、指引我去/到、导航到/至、前往、回到、返回”等"
            "物理移动表达时，必须先检查 navigate，并从后文提取目的地。\n"
            "如果目的地属于 navigate 工具公布的可用地点，第一条回复必须只调用 navigate；"
            "如果目的地缺失或不在可用地点中，才可以询问用户，严禁假装已经行动。\n"
            "“带我了解、引导我理解、告诉我”等知识请求不是物理移动，不要调用 navigate。\n"
            "目的地明确时无需口头确认，不得只回复“好的”“我现在带你去”等承诺文字。\n"
            "口语目的地映射：取药、拿药、领药、配药、去拿药 → 药房；"
            "急诊、急诊室、看急诊 → 急诊科；返回、回去（明确指返回位置时）→ 医院大门。\n"
            "例如，用户说“小麦，带我去取药。”时，你的完整第一条回复只能是：\n"
            '{tool_call:"navigate" param{action:"start", target:"药房"}}\n'
            "用户说“请引导我去急诊室”时，只输出：\n"
            '{tool_call:"navigate" param{action:"start", target:"急诊科"}}\n'
            "用户说“领我回到医院大门”时，只输出：\n"
            '{tool_call:"navigate" param{action:"start", target:"医院大门"}}\n'
            "不要在工具调用前后添加任何自然语言。navigate 返回结果后才能调用 speak 播报。\n\n"
            "## 身份\n"
            "你叫小麦，一个部署在机器人上的语音助手。用户将你称为“小麦”，"
            "“小麦”是你的称谓和唤醒名。\n"
            "用户在对话中提到“小麦”时，默认优先理解为在呼叫或指代你，"
            "不要擅自解释为植物；只有上下文明显在讨论农作物、粮食或种植时，"
            "才将“小麦”理解为植物。\n\n"
            "核心职责：\n"
            "1) 医疗问答 — 回答医学、健康、护理相关问题\n"
            "2) 小车操控 — 通过工具控制机器人移动与操作\n"
            "3) 通用问答 — 回答日常问题\n\n"
            "## 核心原则\n"
            "**诚实**：你没有任何实时信息（时间、日期等）。\n"
            "需要实时数据时必须调用对应工具，**严禁编造**。\n"
            "不确定的事情直接说\"我无法确定\"，不要猜测或虚构。\n\n"
            "**工具优先**：只能使用系统在下方列出的可用工具。\n"
            "用户问\"现在几点\"→ 必须先调 get_time，再调 speak。\n"
            "用户问\"系统状态\"→ 必须先调 get_system_stats，再调 speak。\n"
            "**不要跳过工具直接编造答案。**\n\n"
            "**语音场景**：输入来自 ASR 语音识别，可能同音字错误。\n"
            "结合上下文纠错（如\"请借\"→\"请介绍\"），不确定则确认。\n"
            "回复将被 TTS 朗读，请控制在 2-3 句话以内。\n\n"
            "## 任务处理\n"
            "**医疗问答**：先调用一次 medical_consult，并把用户完整原话放入 query。\n"
            "不要自己把口语改写成实体，也不要调用 entity、graph、fuzzy 等底层命令。\n"
            "只能依据工具 red_flags、questions、departments、associations 和 evidence 回答；"
            "associations 不是诊断，科室只能取 departments.name。"
            "status=need_more_info 且没有 evidence/departments 时只复述澄清问题；"
            "status=urgent 时采用 recommended_destination。\n"
            "medical_consult 返回 medication_allowed=false 时禁止推荐具体药物。\n"
            "即使 medication_allowed=true，药物关系也仅供参考；"
            "不得直接指导用户使用抗菌药、抗病毒药或处方药。\n"
            "**小车操控**：识别到目标明确的移动指令后立即调用工具执行。\n"
            "用户使用带我、引导我、领我、送我、陪我、带路、指引我、去、到、前往、导航、"
            "回到等移动动作前往可用地点，或要求取药/拿药/领药/配药时，"
            "第一条回复必须只调用 navigate，"
            "禁止先说“好的”“我现在去”，也禁止先调用 speak。\n"
            "目标映射：取药/拿药/领药/配药使用 target=\"药房\"；"
            "急诊/急诊室使用 target=\"急诊科\"。"
            "到达工具返回成功前，绝对不能声称已经出发、正在导航或已经到达。\n"
            "调用 navigate 成功后，才可调用 speak 告知执行结果；失败则如实播报失败。\n"
            "**时间/状态查询**：必须调工具获取实时数据。\n\n"
            "## 行为准则\n"
            "1. 普通回答最终用 speak 播报；需要导航或查询实时信息时，先执行对应工具，"
            "工具返回后再调用 speak\n"
            "2. 调用 speak 后立即停止，不输出其他内容\n"
            "3. 医疗建议后提醒\"请咨询专业医生\"\n"
            "4. 小车目标明确时立即执行；只有目标不明确或不受支持时才询问\n"
            "5. 无法处理时诚实告知原因\n\n"
            "## 输入槽位语义\n"
            "system 是固定不变的身份、规则和工具说明。user 是本轮固定的用户任务。\n"
            "history 只包含本轮已经真实执行成功的工具或技能，属于可信事实；"
            "不得重复其中已经成功的动作。\n"
            "attempt 只包含最近一次成功提交之后的失败调用或格式错误；"
            "不得假装这些动作成功，也不得原样重复失败调用。\n\n"
            "## 工具调用格式\n"
            '{tool_call:"工具名" param{参数名:"参数值"}}\n'
            "示例:\n"
            '{tool_call:"get_time" param{}}\n'
            '{tool_call:"speak" param{text:"下午3点"}}\n'
            '{tool_call:"get_system_stats" param{}}\n'
            '{tool_call:"medical_consult" param{query:"我肚子痛两天了怎么办"}}\n'
            '{tool_call:"navigate" param{action:"start", target:"药房"}}\n'
            '{skill_call:"技能名"}\n'
        )

    def _requires_navigation(self, user_input: str) -> bool:
        mcp = getattr(self, "mcp", None)
        navigate = mcp.tools.get("navigate") if mcp is not None else None
        return bool(navigate and navigate.match_request(user_input) is not None)

    def _requires_medical_consult(self, user_input: str) -> bool:
        # Mixed medical + navigation requests require both tools; navigation
        # no longer suppresses medical consultation.
        text = user_input.splitlines()[-1].strip()
        # Metalinguistic questions can contain words such as "看病" without
        # asking for medical advice.  Keep them on the general-answer path.
        if re.search(r"(?:词|词语|说法).{0,8}(?:区别|含义|意思)|[‘'\"]\S+[’'\"].{0,12}(?:区别|含义|意思)", text):
            return False
        mcp = getattr(self, "mcp", None)
        navigate = mcp.tools.get("navigate") if mcp is not None else None
        navigation_request = navigate.match_request(text) if navigate else None
        medical_question = re.search(
            r"怎么办|怎么治|如何治|什么原因|为什么|严重吗|要紧吗|"
            r"吃什么|用什么|能不能|可以吗|是什么意思|症状|病因|治疗|护理|预后|注意事项",
            text,
        )
        # A symptom may only explain a destination: “孩子发烧了，带他去儿科，
        # 怎么走”.  This is a navigation command, not a request for diagnosis.
        if navigation_request and re.search(r"怎么走|怎么去", text) and not medical_question:
            return False
        medical = re.search(
            r"疾病|病因|病症|症状|生病|感冒|发烧|发热|咳嗽|流鼻涕|鼻塞|咽痛|头晕|头痛|头疼|肚子|腹痛|胃痛|"
            r"腹泻|便秘|恶心|呕吐|胃酸|反酸|烧心|用药|吃什么药|治疗|诊断|血压|血糖|"
            r"过敏|月经|怀孕|伤口|胸闷|胸.{0,2}闷|气短|喘|麻木|胳膊.{0,2}麻|"
            r"抽搐|痉挛|眼角.{0,2}(?:跳|抽)|出血|皮疹|疼|痛|不舒服|"
            r"健康|并发症|辅助检查|检查方法|预防|护理|康复|预后|遗传|传染|"
            r"风险因素|危险因素|高危因素|"
            r"抗生素|处方(?:药)?|药物|用药|成分|注意事项|禁忌|副作用|"
            r"胶囊|药片|颗粒|口服液|中成药|药材|原形态|"
            r"避孕套|促甲状腺激素|勃起障碍|肉芽|坐月子|"
            r"一枝黄花|五月茶|生长习性|"
            r"双眼皮|埋线|全切|丸(?:成分|处方|怎么吃|用法)|"
            r"挂什么科|看什么科|该去哪看|"
            r"(?:炎|病|症|疮|癌|瘤|肿|菌|病毒|手术|检查|闭经|成瘾)",
            text,
        )
        return bool(medical)

    @staticmethod
    def _requires_system_stats(user_input: str) -> bool:
        return bool(re.search(
            r"系统状态|资源使用|资源占用|内存.{0,8}(?:够|使用|占用)|"
            r"CPU.{0,8}(?:使用|占用)|磁盘.{0,8}(?:使用|占用)",
            user_input,
            re.IGNORECASE,
        ))

    @staticmethod
    def _requires_current_time(user_input: str) -> bool:
        return bool(re.search(
            r"(?:现在|当前|今天).{0,8}(?:几点|时间|日期|几月几号|星期几|周几)|"
            r"(?:几点|几月几号|星期几|周几)(?:了|\?|？)?$",
            user_input,
        ))

    @staticmethod
    def _is_confident_general(user_input: str) -> bool:
        return bool(re.search(
            r"什么叫|什么意思|词语|说法|"
            r"(?:有什么)?区别|等于多少|该找多少|怎么算",
            user_input,
        ))

    @staticmethod
    def _is_simple_greeting(user_input: str) -> bool:
        """Recognize greetings that require no retrieval or planning.

        A hotword turn often contains only ``你好``.  Letting that reach the
        general planner previously allowed it to invent a get_time step merely
        to "confirm service state", adding two costly LLM calls with no user
        benefit.  Keep this deliberately narrow so a greeting followed by a
        real request still goes through normal routing.
        """
        lines = [line.strip() for line in str(user_input).splitlines() if line.strip()]
        text = lines[-1] if lines else ""
        text = re.sub(r"[，,。！？!?~～\s]+", "", text)
        return bool(re.fullmatch(
            r"(?:小麦)?(?:你好|您好|嗨|哈喽|hello|hi|早上好|上午好|中午好|下午好|晚上好|在吗)",
            text,
            flags=re.IGNORECASE,
        ))

    def _build_compact_workflow_plan(
        self, user_input: str, *, atomic_navigation: bool
    ) -> dict:
        """Create a deterministic plan for high-confidence production intents."""
        medical = self._requires_medical_consult(user_input)
        navigation = self._requires_navigation(user_input)
        system_stats = self._requires_system_stats(user_input)
        current_time = self._requires_current_time(user_input)

        def step(step_id, goal, tool, depends_on=(), condition=None):
            return {
                "step_id": step_id,
                "goal": goal,
                "preferred_tool": tool,
                "allowed_tools": [tool],
                "depends_on": list(depends_on),
                "condition": condition,
            }

        steps = []
        if medical:
            steps.append(step("s1", "查询医疗证据", "medical_consult"))
            previous = "s1"
            if navigation:
                steps.append(step("s2", "启动用户请求的导航", "navigate", (previous,)))
                previous = "s2"
            steps.append(step(
                "s_followup", "在医疗工具明确要求时向用户追问", "query", (previous,),
                {"fact": "dialogue.followup_required", "operator": "eq", "value": True},
            ))
            steps.append(step(
                "s_final", "依据已验证事实播报答复", "speak", ("s_followup",)
            ))
            goal = "完成医疗咨询" + ("并启动导航" if navigation else "")
        elif navigation:
            if atomic_navigation:
                steps = [step("s1", "启动导航并由导航工具播报结果", "navigate")]
            else:
                steps = [
                    step("s1", "启动用户请求的导航", "navigate"),
                    step("s_final", "播报导航启动结果", "speak", ("s1",)),
                ]
            goal = "启动导航并播报结果"
        elif system_stats or current_time:
            tool = "get_system_stats" if system_stats else "get_time"
            goal_text = "获取系统资源状态" if system_stats else "获取当前日期时间"
            steps = [
                step("s1", goal_text, tool),
                step("s_final", "依据实时工具结果播报答复", "speak", ("s1",)),
            ]
            goal = goal_text + "并播报结果"
        else:
            steps = [step("s_final", "播报对用户的答复", "speak")]
            goal = "回应用户问候" if self._is_simple_greeting(user_input) else "回答用户问题"
        return {
            "goal": goal,
            "goal_description": goal,
            "success_conditions": [],
            "steps": steps,
            "done_when": "完成预定工具动作后以 speak 或 query 结束当前外部轮次",
        }

    def get_compact_workflow_plan(self, user_input: str):
        """Use compact routing only when production intent is high-confidence.

        Unrecognized medical entities and ambiguous movement requests retain
        the model-planning path instead of being frozen into a generic speak
        step by an incomplete lexical router.
        """
        if not (
            self._requires_medical_consult(user_input)
            or self._requires_navigation(user_input)
            or self._requires_system_stats(user_input)
            or self._requires_current_time(user_input)
            or self._is_confident_general(user_input)
            or self._is_simple_greeting(user_input)
        ):
            return None
        return self._build_compact_workflow_plan(
            user_input, atomic_navigation=False
        )

    def _get_scheduled_compact_command(
        self, user_input: str, *, atomic_navigation: bool
    ) -> Optional[Command]:
        state = self.session.execution_state if self.session is not None else None
        step = state.active_step if state is not None else None
        if step is None:
            return None
        tool = step.preferred_tool
        arguments = None
        if tool == "medical_consult":
            arguments = {"query": user_input}
        elif tool == "navigate":
            navigate = self.mcp.tools.get("navigate")
            request = navigate.match_request(user_input) if navigate else None
            if request is not None:
                action, target = request
                arguments = {"action": action}
                if target:
                    arguments["target"] = target
                    if atomic_navigation and not self._requires_medical_consult(user_input):
                        arguments["announcement"] = f"我将导航到{target}，请跟紧我。"
        elif tool == "query":
            questions = state.facts.get("dialogue.followup_questions")
            values = questions.value if questions is not None and questions.valid else []
            question = str(values[0]).strip() if values else ""
            if question:
                arguments = {"question": question}
        elif tool == "get_system_stats":
            arguments = {}
        elif tool == "get_time":
            date_only = bool(re.search(r"日期|几月几号|星期几|周几", user_input))
            arguments = {"format": "date" if date_only else "full"}
        elif tool == "speak":
            facts = state.facts
            stats = facts.get("system.stats")
            current_time = facts.get("system.time")
            navigation = facts.get("navigation.target")
            medical = facts.get("medical.consultation")
            if stats is not None and stats.valid:
                arguments = {"text": str(stats.value)}
            elif current_time is not None and current_time.valid:
                arguments = {"text": f"当前时间：{current_time.value}"}
            elif (
                navigation is not None
                and navigation.valid
                and (medical is None or not medical.valid)
            ):
                arguments = {
                    "text": f"已开始导航至{navigation.value}，请跟紧我。"
                }
            elif medical is not None and medical.valid:
                text = grounded_medical_speech(
                    medical.value,
                    navigation_target=(
                        str(navigation.value)
                        if navigation is not None and navigation.valid else ""
                    ),
                )
                if text:
                    arguments = {"text": text}
            elif self._is_simple_greeting(user_input):
                arguments = {
                    "text": "你好，我是医院机器人助手小麦。请问有什么可以帮您？"
                }
        if arguments is None:
            return None
        return Command(
            type="tool_call", name=tool, params=arguments,
            raw="[agent_workflow_scheduler]",
            confidence=1.0,
        )

    def get_scheduled_workflow_command(self, user_input: str):
        return self._get_scheduled_compact_command(
            user_input, atomic_navigation=False
        )

    def get_turn_policy_prompt(
        self,
        user_input: str,
        successful_tools: set,
        attempted_tools: set,
    ) -> str:
        if self._requires_navigation(user_input) and self._requires_medical_consult(user_input):
            navigate = self.mcp.tools.get("navigate")
            request = navigate.match_request(user_input) if navigate else None
            target = request[1] if request and request[0] == "start" else ""
            return (
                "## 本轮医疗+导航混合任务约束（最高优先级）\n"
                "必须先调用 medical_consult，并将用户完整原话作为 query；只有获得医疗证据后，"
                "才能执行用户明确要求的物理导航。不得用模型自身医学知识替代查询，不得根据"
                "症状擅自诊断。"
                + (f"用户明确请求的合法导航目标是“{target}”。" if target else "")
                + "navigate 成功只表示开始导航，不表示已经到达。两个动作都完成后调用 "
                  "speak 播报最终答复。"
            )
        return super().get_turn_policy_prompt(user_input, successful_tools, attempted_tools)

    def get_plain_response_rejection(
        self,
        user_input: str,
        response_text: str,
        successful_tools: set,
        attempted_tools: set | None = None,
    ) -> str:
        navigation_rejection = super().get_plain_response_rejection(
            user_input,
            response_text,
            successful_tools,
            attempted_tools,
        )
        if navigation_rejection:
            return navigation_rejection
        if (
            self._requires_medical_consult(user_input)
            and "medical_consult" not in successful_tools
        ):
            return (
                "这是医疗问题，但 medical_consult 尚未成功执行。"
                "请先将用户完整原话作为 query 调用该工具。"
            )
        return ""

    def get_command_rejection(
        self,
        user_input: str,
        command,
        successful_tools: set,
        attempted_tools: set | None = None,
    ) -> str:
        if str(getattr(command, "raw", "")).startswith(
            "[agent_workflow_blocked_terminal]"
        ):
            return ""
        navigation_rejection = super().get_command_rejection(
            user_input,
            command,
            successful_tools,
            attempted_tools,
        )
        if navigation_rejection:
            return navigation_rejection
        if (
            self._requires_navigation(user_input)
            and self._requires_medical_consult(user_input)
            and command.type == "tool_call"
            and command.name == "navigate"
            and "medical_consult" not in successful_tools
        ):
            return "混合任务必须先执行 medical_consult 获取医疗证据，再执行导航。"
        # medical_consult is a read-only evidence lookup. The lexical router is
        # intentionally one-way: it may require retrieval for an obvious
        # medical request, but it must not veto a model-planned lookup merely
        # because an uncommon database term was absent from the regex.
        if (
            self._requires_medical_consult(user_input)
            and command.type == "tool_call"
            and command.name == "speak"
            and "medical_consult" not in successful_tools
        ):
            return "医疗检索尚未执行，禁止先调用 speak；请先调用 medical_consult。"
        return ""

    async def start(self, host: str = "0.0.0.0", port: int = 8766):
        """Start the voice WebSocket server and begin listening."""
        from ..Voice.hotword import HotwordDetector
        from ..Voice.voice_server import VoiceServer, AgentMessage

        self._host = host
        self._port = port

        detector = HotwordDetector(hotwords=["朋友"])
        self._voice_server = VoiceServer(detector)
        self._voice_server.set_on_utterance(self._on_utterance_captured)

        self._running = True
        self._server_task = asyncio.create_task(
            self._run_server(host, port)
        )
        logger.info(f"Voice mode started on ws://{host}:{port}")

    async def _run_server(self, host: str, port: int):
        """Run the FastAPI WebSocket server."""
        from fastapi import FastAPI, WebSocket
        import uvicorn

        voice_server = self._voice_server

        app = FastAPI()

        # Load frontend HTML for same-origin serving (avoids browser file:// WebSocket blocks)
        _frontend_html = "<h1>Frontend not found</h1>"
        try:
            import os as _os
            fp = _os.path.join(_os.path.dirname(__file__), "..", "..", "voice_frontend.html")
            fp = _os.path.abspath(fp)
            if _os.path.exists(fp):
                with open(fp, "r", encoding="utf-8") as _f:
                    _frontend_html = _f.read()
                logger.info(f"Frontend loaded from {fp}")
        except Exception:
            pass

        @app.get("/health")
        async def health():
            return {"status": "ok", "voice_state": voice_server.state.value}

        @app.get("/")
        async def serve_frontend():
            from fastapi.responses import HTMLResponse
            return HTMLResponse(content=_frontend_html)


        @app.websocket("/asr")
        async def asr_proxy(browser_ws: WebSocket):
            """Proxy browser ASR WebSocket to local ASR server (port 8765).
            This avoids cross-port WebSocket issues in browsers."""
            await browser_ws.accept()
            try:
                import websockets as _ws
                async with _ws.connect("ws://127.0.0.1:8765/") as _asr_ws:
                    async def _fwd_to_asr():
                        while True:
                            msg = await browser_ws.receive()
                            if "text" in msg:
                                await _asr_ws.send(msg["text"])
                            elif "bytes" in msg:
                                await _asr_ws.send(msg["bytes"])
                    async def _fwd_to_browser():
                        while True:
                            msg = await _asr_ws.recv()
                            if isinstance(msg, str):
                                await browser_ws.send_text(msg)
                            else:
                                await browser_ws.send_bytes(msg)
                    await asyncio.gather(_fwd_to_asr(), _fwd_to_browser())
            except Exception as _e:
                logger.debug(f"ASR proxy disconnected: {_e}")

        @app.websocket("/ws")
        async def ws_endpoint(ws: WebSocket):
            await ws.accept()
            voice_server.add_client(ws)
            try:
                while self._running:
                    try:
                        data = await asyncio.wait_for(
                            ws.receive_text(), timeout=1.0
                        )
                        await voice_server.handle_message(ws, data)
                    except asyncio.TimeoutError:
                        utterance = await voice_server.check_timeout()
                        if utterance:
                            self._pending_utterance = utterance
                        continue
            except Exception as e:
                logger.debug(f"Voice WS disconnected: {e}")
            finally:
                voice_server.remove_client(ws)

        config = uvicorn.Config(
            app, host=host, port=port, log_level="warning", ws="wsproto"
        )
        self._uvicorn_server = uvicorn.Server(config)
        await self._uvicorn_server.serve()

    async def stop(self):
        """Stop the voice server."""
        self._running = False
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
        if self._server_task:
            self._server_task.cancel()
            try:
                await self._server_task
            except asyncio.CancelledError:
                pass
        logger.info("Voice mode stopped")

    async def _on_utterance_captured(self, utterance: str):
        """Called by VoiceServer when a complete utterance is captured."""
        self._pending_utterance = utterance

    async def poll_utterance(self) -> Optional[str]:
        """Non-blocking check for captured utterances."""
        if self._pending_utterance:
            text = self._pending_utterance
            self._pending_utterance = None
            return text
        return None

    async def loop(
        self,
        user_input: str,
        image: Optional[str] = None,
        context: str = "",
        sensor_data: str = "",
        tool_context_extra: Optional[dict] = None,
    ) -> LoopResult:
        """Process voice-captured utterance through the agent pipeline."""
        return await super().loop(
            user_input=user_input,
            image=image,
            context=context,
            sensor_data=sensor_data,
            tool_context_extra=tool_context_extra,
        )

    async def broadcast_response(self, text: str):
        """Send a speak+response message to the browser."""
        if self._voice_server:
            from ..Voice.voice_server import AgentMessage
            await self._voice_server.broadcast(
                AgentMessage(type="speak", text=text)
            )
            await self._voice_server.broadcast(
                AgentMessage(type="response", text=text)
            )

    async def broadcast_ack(self):
        """Send acknowledgment (beep) to the browser."""
        if self._voice_server:
            from ..Voice.voice_server import AgentMessage
            await self._voice_server.broadcast(AgentMessage(type="ack"))
