from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .API.api import API
from .API.session import Session
from .CallRoute.router import CallRouter
from .MCP.manager import MCPManager
from .MCP.tools import ALL_TOOLS
from .Camera import create_camera, CameraBackend
from .Skill.loader import SkillLoader
from .Skill.manager import SkillManager
from .Modes import VoiceMode, BenchmarkMode
from .MCP.tools.navigate import NavigateTool
from .MCP.tools.medconsult import MedicalConsultTool
from .Modes.base import LoopResult

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    server_url: str = "http://127.0.0.1:8000"
    skill_dir: str = ""
    model_name: str = "qwen3.5-4b-history"
    max_context_tokens: int = 4096
    auto_compact_threshold: float = 0.8
    history_visible_entries: int = 8
    asr_ws_url: str = "ws://127.0.0.1:8765"
    voice_port: int = 8766
    camera_backend: str = "auto"
    camera_device: str = "/dev/video0"
    camera_image_path: str = ""
    default_mode: str = "Voice"
    benchmark_enabled: bool = False
    benchmark_max_tokens: int = 128
    emit_agent_events: bool = True
    trajectory_enabled: bool = True
    trajectory_dir: str = "/data/structure/trajectories"
    navigation_location_profile: str = "auto"
    medical_dense_enabled: Optional[bool] = None
    online_max_concurrency: int = 3
    context_window_tokens: int = 8192
    tokenizer_path: str = str(
        Path(__file__).resolve().parents[1] / "qwen3.5history" / "config"
    )


class Agent:
    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or AgentConfig()
        self.api = API(
            server_url=self.config.server_url,
            default_model=self.config.model_name,
            benchmark=self.config.benchmark_enabled,
            max_tokens=self.config.benchmark_max_tokens,
            emit_events=self.config.emit_agent_events,
            trajectory_enabled=self.config.trajectory_enabled,
            trajectory_dir=self.config.trajectory_dir,
            online_max_concurrency=self.config.online_max_concurrency,
            context_window_tokens=self.config.context_window_tokens,
            tokenizer_path=self.config.tokenizer_path,
        )
        self.mcp = MCPManager()
        self.skill_loader = SkillLoader(
            self.config.skill_dir
            or str(Path(__file__).parent / "Skill" / "examples")
        )
        self.skill_manager = SkillManager(self.skill_loader)
        self.call_router = CallRouter(mcp=self.mcp, skill_manager=self.skill_manager)
        self.voice_mode: Optional[VoiceMode] = None
        self.benchmark_mode: Optional[BenchmarkMode] = None
        self._ros_bridge = None
        self._voice_task: Optional[asyncio.Task] = None
        self._voice_running = False
        self._session_lock = asyncio.Lock()
        self.camera: Optional[CameraBackend] = None

    def initialize(self):
        self._register_tools()
        self._load_skills()
        self._init_voice_mode()
        self._init_benchmark_mode()
        self.call_router.update_registered_names()
        logger.info("Agent 初始化完成")

    def _register_tools(self):
        for tool_cls in ALL_TOOLS:
            if tool_cls is NavigateTool:
                profile = self.config.navigation_location_profile
                if profile == "auto":
                    profile = "hospital" if self.config.default_mode == "Benchmark" else "basic"
                tool_instance = tool_cls(
                    execution_mode="mock" if self.config.default_mode == "Benchmark" else "real",
                    location_profile=profile,
                )
            elif tool_cls is MedicalConsultTool:
                dense_enabled = self.config.medical_dense_enabled
                if dense_enabled is None and self.config.default_mode == "Benchmark":
                    dense_enabled = False
                tool_instance = tool_cls(
                    dense_enabled=dense_enabled
                )
            else:
                tool_instance = tool_cls()
            self.mcp.register(tool_instance)
            if isinstance(tool_instance, MedicalConsultTool):
                tool_instance.start_prewarm()
        logger.info(f"注册 {len(self.mcp.tools)} 个工具")

    def _load_skills(self):
        skills = self.skill_loader.load_all()
        registered_tools = set(self.mcp.tools)
        for name, skill in list(skills.items()):
            missing = set(skill.allowed_tools) - registered_tools
            if missing:
                logger.warning(
                    f"跳过不可执行技能 {name}: 未注册工具 {sorted(missing)}"
                )
                del skills[name]
        logger.info(f"加载 {len(skills)} 个技能: {list(skills.keys())}")

    def _init_voice_mode(self):
        common_args = {
            "api": self.api,
            "mcp": self.mcp,
            "skill_manager": self.skill_manager,
            "call_router": self.call_router,
        }
        self.voice_mode = VoiceMode(**common_args)
        self.voice_mode.config.history_visible_entries = (
            self.config.history_visible_entries
        )
        logger.info("初始化 Voice Agent")
        # Initialize camera
        self.camera = create_camera(
            backend=self.config.camera_backend,
            device=self.config.camera_device,
            path=self.config.camera_image_path,
        )
        if self.camera.is_available():
            logger.info("Camera backend ready")
        else:
            logger.info("No camera available — image input disabled")

    def _init_benchmark_mode(self):
        common_args = {
            "api": self.api,
            "mcp": self.mcp,
            "skill_manager": self.skill_manager,
            "call_router": self.call_router,
        }
        self.benchmark_mode = BenchmarkMode(**common_args)
        self.benchmark_mode.config.history_visible_entries = self.config.history_visible_entries

    async def handle_input(
        self,
        user_input: str,
        image: Optional[str] = None,
    ) -> LoopResult:
        """
        处理用户输入的主入口。请求交给 Voice Agent；高置信工具意图由
        确定性工作流分解和调度，未覆盖或含糊任务回退模型 plan/act。
        """
        if not user_input.strip():
            logger.debug("空输入，忽略")
            return LoopResult()

        if self.voice_mode is None:
            raise RuntimeError("Agent 尚未初始化")
        return await self.voice_mode.loop(
            user_input=user_input,
            image=image,
            context="",
            sensor_data=self._get_sensor_data(),
        )

    async def handle_input_in_session(
        self,
        user_input: str,
        image: Optional[str] = None,
        session: Optional[Session] = None,
        tool_context_extra: Optional[dict] = None,
    ) -> LoopResult:
        """
        使用指定 Session 处理 Voice 请求，以实现 Headless 会话隔离。
        返回后恢复 Voice Agent 原有的 Session。
        """
        if not user_input.strip():
            logger.debug("空输入，忽略")
            return LoopResult()

        if self.voice_mode is None:
            raise RuntimeError("Agent 尚未初始化")

        # VoiceMode instance is shared by all headless sessions. Serialize access
        # so concurrent clients cannot overwrite its temporary session reference.
        async with self._session_lock:
            saved_session = self.voice_mode.session
            if session is not None:
                self.voice_mode.session = session

            try:
                return await self.voice_mode.loop(
                    user_input=user_input,
                    image=image,
                    context="",
                    sensor_data=self._get_sensor_data(),
                    tool_context_extra=tool_context_extra,
                )
            finally:
                if session is not None:
                    self.voice_mode.session = saved_session

    async def handle_benchmark_input(
        self,
        user_input: str,
        session: Optional[Session] = None,
    ) -> LoopResult:
        if not user_input.strip():
            return LoopResult()
        if self.benchmark_mode is None:
            raise RuntimeError("Benchmark Agent 尚未初始化")
        async with self._session_lock:
            saved_session = self.benchmark_mode.session
            if session is not None:
                self.benchmark_mode.session = session
            try:
                return await self.benchmark_mode.loop(user_input=user_input)
            finally:
                if session is not None:
                    self.benchmark_mode.session = saved_session

    def capture_image(self) -> Optional[str]:
        """Capture a frame from the camera, return base64 JPEG or None."""
        if self.camera is None:
            return None
        return self.camera.capture()

    def _get_sensor_data(self) -> str:
        return ""

    async def start_voice_mode(self, host: str = "0.0.0.0", port: int = 8766):
        """Start the voice interaction mode.

        This launches a WebSocket server that accepts browser connections,
        listens for hotword-triggered utterances, and sends them through
        the unified Voice pipeline.
        """
        voice_mode = self.voice_mode
        if voice_mode is None:
            raise RuntimeError("Agent 尚未初始化")

        await voice_mode.start(host=host, port=port)
        self._voice_running = True
        self._voice_task = asyncio.create_task(
            self._voice_loop(voice_mode)
        )
        logger.info(f"Voice mode active on ws://{host}:{port}")

    async def _voice_loop(self, voice_mode):
        """Main voice interaction loop: poll for utterances and respond."""
        while self._voice_running:
            try:
                utterance = await voice_mode.poll_utterance()
                if not utterance:
                    await asyncio.sleep(0.1)
                    continue

                logger.info(f"Voice utterance: {utterance[:80]}...")

                # Wrap ASR transcription to inform LLM about potential inaccuracies
                wrapped_input = (
                    "[语音识别结果，可能含有不准确信息，请结合上下文尝试修正与辨别]\n"
                    + utterance
                )

                # Capture camera frame for multimodal input
                image = await asyncio.to_thread(self.capture_image)

                result = await voice_mode.loop(
                    user_input=wrapped_input,
                    image=image,
                    context="",
                    sensor_data=self._get_sensor_data(),
                )

                # Send response back via voice
                if result.text:
                    await voice_mode.broadcast_response(result.text)
                elif not result.commands:
                    # No text, no commands - LLM may have returned empty
                    await voice_mode.broadcast_response("请再说一遍")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Voice loop error: {e}")
                await asyncio.sleep(0.5)

    async def stop_voice_mode(self):
        """Stop the voice interaction mode."""
        self._voice_running = False
        self.camera: Optional[CameraBackend] = None
        voice_mode = self.voice_mode
        if voice_mode:
            await voice_mode.stop()
        if self._voice_task:
            self._voice_task.cancel()
            try:
                await self._voice_task
            except asyncio.CancelledError:
                pass
        logger.info("Voice mode stopped")

    async def shutdown(self):
        await self.stop_voice_mode()
        await self.api.close()
        logger.info("Agent 关闭")
