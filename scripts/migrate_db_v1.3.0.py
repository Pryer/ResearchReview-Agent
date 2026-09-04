"""数据库迁移脚本：v1.2.0 -> v1.3.0

为 papers 表添加 citation_count_by_source 字段。

运行方式：
    python scripts/migrate_db_v1.3.0.py

注意：
- 仅适用于 SQLite 数据库
- 会自动备份原数据库文件
- 如果字段已存在，则跳过迁移
"""

import shutil
import sys
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.core.config import get_settings
from app.core.logger import get_logger
from app.database.db import engine

logger = get_logger(__name__)


def check_column_exists(table_name: str, column_name: str) -> bool:
    """检查表中是否已存在指定列。"""
    from sqlalchemy import text
    
    with engine.connect() as conn:
        result = conn.execute(text(f"PRAGMA table_info({table_name})"))
        columns = [row[1] for row in result]
        return column_name in columns


def backup_database(db_path: Path) -> Path:
    """备份数据库文件。"""
    if not db_path.exists():
        raise FileNotFoundError(f"数据库文件不存在: {db_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
    
    shutil.copy2(db_path, backup_path)
    logger.info(f"数据库已备份到: {backup_path}")
    return backup_path


def migrate():
    """执行数据库迁移。"""
    settings = get_settings()
    
    # 检查是否为 SQLite
    if not settings.database_url.startswith("sqlite"):
        logger.error("此迁移脚本仅支持 SQLite 数据库")
        sys.exit(1)
    
    # 解析数据库文件路径
    db_path_str = settings.database_url.replace("sqlite:///", "")
    db_path = Path(db_path_str)
    
    logger.info(f"数据库路径: {db_path}")
    
    # 备份数据库
    if db_path.exists():
        try:
            backup_path = backup_database(db_path)
            logger.info(f"✅ 备份成功: {backup_path}")
        except Exception as e:
            logger.error(f"备份失败: {e}")
            sys.exit(1)
    else:
        logger.warning("数据库文件不存在，将创建新数据库")
    
    # 检查是否需要迁移
    if check_column_exists("papers", "citation_count_by_source"):
        logger.info("✅ citation_count_by_source 字段已存在，无需迁移")
        return
    
    # 执行迁移
    from sqlalchemy import text
    
    try:
        with engine.begin() as conn:
            logger.info("开始添加 citation_count_by_source 字段...")
            conn.execute(text(
                "ALTER TABLE papers ADD COLUMN citation_count_by_source TEXT"
            ))
            logger.info("✅ 字段添加成功")
        
        logger.info("✅ 数据库迁移完成")
        logger.info("提示：旧数据的 citation_count_by_source 将为 NULL，")
        logger.info("      在下次 fetch_metadata 时会自动补全")
        
    except Exception as e:
        logger.error(f"❌ 迁移失败: {e}")
        logger.error(f"请使用备份文件恢复: {backup_path if 'backup_path' in locals() else '未知'}")
        sys.exit(1)


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("ResearchReview-Agent 数据库迁移工具 v1.3.0")
    logger.info("=" * 60)
    
    try:
        migrate()
    except KeyboardInterrupt:
        logger.warning("\n迁移被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"未预期的错误: {e}")
        sys.exit(1)
