from .voice import QueryTool, SpeakTool
from ..tools.system import GetSystemStatsTool, GetTimeTool
from .medquery import MedQueryTool
from .medconsult import MedicalConsultTool
from .navigate import NavigateTool
from .plan import PlanTool
from .act import ActTool

# ``MedQueryTool`` remains importable as a rollback/diagnostic adapter, but is
# intentionally hidden from the model.  Small models should only see the
# high-level, one-argument medical consultation tool.
ALL_TOOLS = [
    PlanTool,
    ActTool,
    SpeakTool,
    QueryTool,
    GetSystemStatsTool,
    GetTimeTool,
    MedicalConsultTool,
    NavigateTool,
]

__all__ = [
    "SpeakTool",
    "QueryTool",
    "GetSystemStatsTool",
    "GetTimeTool",
    "MedicalConsultTool",
    "MedQueryTool",
    "NavigateTool",
    "PlanTool",
    "ActTool",
    "ALL_TOOLS",
]
