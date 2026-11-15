"""保持证据绑定的主张表述润色协议。"""

CLAIM_REFINEMENT_PROMPT = """你是学术编辑。请优化以下研究路线的论证主张表述，使其更精确、更学术化。

路线：{route_name}
研究问题：{research_question}

当前主张：
{claims_json}

要求：
1. 每个输入 claim_id 必须原样返回且恰好一次，不得遗漏、重复或增加 ID。
2. 仅润色 claim_text，不改变 claim_type、support_level 或证据绑定；single 主张不得升级为趋势判断。
3. 不添加事实，不改变否定、条件、适用范围、单位及结论方向。
4. 保留每条主张中的全部数字及其顺序，包括符号、小数、指数和百分比。
5. 返回与本次输入数量相同的列表，输出顺序不限，代码按 claim_id 对应。

严格返回 JSON，不输出解释：
{{"refined_claims": [{{"claim_id": "输入中的原始ID", "claim_text": "润色后的原主张"}}]}}
"""
