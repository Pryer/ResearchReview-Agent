"""兼容旧的 ``app.agent.prompts`` 导入路径，且不预加载 Prompt。"""

from app.prompt_catalog import __all__, __dir__, __getattr__

