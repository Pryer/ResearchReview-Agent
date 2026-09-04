"""兼容启动入口：统一启动主 Streamlit 前端。

使用方式：
    python run_chat_frontend.py
"""

import os
import sys
from pathlib import Path

# 设置项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)

# 启动 Streamlit
frontend_path = PROJECT_ROOT / "app" / "frontend" / "chat_app.py"

print("=" * 60)
print("🚀 启动 ResearchReview-Agent 主前端（兼容入口）")
print("=" * 60)
print()
print(f"前端文件: {frontend_path}")
print("访问地址: http://localhost:8501")
print()
print("按 Ctrl+C 停止服务")
print("=" * 60)
print()

os.system(f"streamlit run {frontend_path}")
