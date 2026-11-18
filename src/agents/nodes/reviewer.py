import sys
from pathlib import Path

app_path = str(Path(__file__).resolve().parent.parent.parent.parent / "my-agent-app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from src.agent.agent import default_llm_factory, default_retriever_factory, make_nodes

# Expose reviewer and fallback nodes
_nodes = make_nodes(default_llm_factory, default_retriever_factory)
reviewer_node = _nodes[3]
conservative_fallback_node = _nodes[4]

__all__ = ["reviewer_node", "conservative_fallback_node"]
