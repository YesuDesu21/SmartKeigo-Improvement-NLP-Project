import sys
from pathlib import Path

# Add my-agent-app to path if not present
app_path = str(Path(__file__).resolve().parent.parent.parent / "my-agent-app")
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from src.agent.guardrails.honorific_rules import (
    NIJUU_KEIGO_PATTERNS,
    audit_nijuu_keigo,
    convert_to_conservative_teineigo,
)

__all__ = ["NIJUU_KEIGO_PATTERNS", "audit_nijuu_keigo", "convert_to_conservative_teineigo"]

