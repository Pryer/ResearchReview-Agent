"""状态栏可回读摘要的身份目录，不授予业务执行权限。"""

def selectable_context_ids(context):
    """只返回当前可见摘要的ID；论文身份和资料引用不在此命名空间。"""
    return {
        "evidence_ids": [item.evidence_id for item in context.key_evidence],
        "decision_ids": [item.decision_id for item in context.decisions],
        "question_ids": [item.question_id for item in context.open_questions],
        "memory_ids": [item["memory_id"] for item in context.state.research_memory],
    }
