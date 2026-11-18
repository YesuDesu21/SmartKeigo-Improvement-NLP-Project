import sys
from pathlib import Path

# Add my-agent-app to path if not present
app_path = str(Path(__file__).resolve().parent.parent.parent / "my-agent-app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from src.agent.retrieval import Retriever, format_few_shot_exemplars

__all__ = ["Retriever", "format_few_shot_exemplars"]
