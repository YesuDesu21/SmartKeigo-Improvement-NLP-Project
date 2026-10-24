"""SmartKeigo / Keigo Flow LangGraph agent."""
from .graph import build_graph, graph

keigo_app = graph  # backwards-compatible alias for the old app.py import
__all__ = ["build_graph", "graph", "keigo_app"]
