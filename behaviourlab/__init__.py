"""BehaviourLabNext：Windows 本地经典视觉动物行为分析应用。

导入本包只暴露版本信息，不创建数据库、窗口、线程或工作进程。
"""

from .config import APP_NAME, APP_VERSION

__all__ = ["APP_NAME", "APP_VERSION"]
__version__ = APP_VERSION
