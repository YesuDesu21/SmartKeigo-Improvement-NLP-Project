from langgraph.graph import END, START, StateGraph

from .agent import default_llm_factory, default_retriever_factory, make_nodes
from .state import AgentState, KeigoState


def build_graph(llm_factory=default_llm_factory, retriever_factory=default_retriever_factory):
    analyzer, retrieve, specialist, reviewer, fallback, route = make_nodes(llm_factory, retriever_factory)
    b = StateGraph(AgentState)
    b.add_node("analyzer_node", analyzer)
    b.add_node("retrieve_node", retrieve)
    b.add_node("specialist_node", specialist)
    b.add_node("reviewer_node", reviewer)
    b.add_node("conservative_fallback", fallback)

    # Parallel fan-out from START to analyzer and retriever concurrently
    b.add_edge(START, "analyzer_node")
    b.add_edge(START, "retrieve_node")

    # Fan-in synchronization into Keigo Specialist
    b.add_edge(["analyzer_node", "retrieve_node"], "specialist_node")
    b.add_edge("specialist_node", "reviewer_node")

    # Bounded critique loop: at most cfg.max_revisions revisions, then fallback to conservative Teineigo
    b.add_conditional_edges(
        "reviewer_node",
        route,
        {
            "end": END,
            "revise": "specialist_node",
            "fallback": "conservative_fallback",
        },
    )
    b.add_edge("conservative_fallback", END)
    return b.compile()


graph = build_graph()  # no network/API keys needed until invoke
