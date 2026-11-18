import sys
from pathlib import Path

app_path = str(Path(__file__).resolve().parent.parent.parent / "my-agent-app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from src.agent.graph import build_graph, graph
from src.agent.agent import make_nodes

__all__ = ["build_graph", "graph", "make_nodes"]
