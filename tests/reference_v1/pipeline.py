"""历史协议参考入口，仅供manifest列出的退役协议测试。"""
def run_pipeline(state, *, should_cancel=None, progress_callback=None, local_verification=None):
    from app.agent import graph
    from tests.reference_v1.main_loop import MainAgentLoop
    llm = graph._get_llm()
    handlers, finish = graph.build_research_actions(state, should_cancel=should_cancel,
        progress_callback=progress_callback, local_verification=local_verification, llm=llm)
    status = MainAgentLoop(llm).run(state, handlers=handlers, finish_validator=finish, should_cancel=should_cancel)
    return graph._complete_research_pipeline(state, status, progress_callback=progress_callback)
