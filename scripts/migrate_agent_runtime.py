"""幂等创建新增执行/记忆表；不改变既有论文、会话及 artifact 数据。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.database.db import engine
from app.database.models import (
    ResearchRuntimeModel, ResearchTaskModel, ResearchAttemptModel,
    ResearchCheckpointModel, ResearchEventModel, ResearchMemoryModel,
)


def migrate(bind):
    for model in (ResearchRuntimeModel, ResearchTaskModel, ResearchAttemptModel,
                  ResearchCheckpointModel, ResearchEventModel, ResearchMemoryModel):
        model.__table__.create(bind=bind, checkfirst=True)


if __name__ == "__main__":
    migrate(engine)
    print("Agent runtime additive migration complete")
