from langgraph.graph import END, START, StateGraph

from .agent import default_llm_factory, default_retriever_factory, make_nodes
from .state import KeigoState


def build_graph(llm_factory=default_llm_factory, retriever_factory=default_retriever_factory):
    analyzer, retrieve, specialist, reviewer, route = make_nodes(llm_factory, retriever_factory)
    b = StateGraph(KeigoState)
    b.add_node("analyzer_node", analyzer)
    b.add_node("retrieve_node", retrieve)
    b.add_node("specialist_node", specialist)
    b.add_node("reviewer_node", reviewer)
    b.add_edge(START, "analyzer_node")
    b.add_edge("analyzer_node", "retrieve_node")
    b.add_edge("retrieve_node", "specialist_node")
    b.add_edge("specialist_node", "reviewer_node")
    # Bounded loop: at most cfg.max_revisions revisions after the initial draft.
    b.add_conditional_edges("reviewer_node", route, {"revise": "specialist_node", "end": END})
    return b.compile()


graph = build_graph()  # no network/API keys needed until invoke
