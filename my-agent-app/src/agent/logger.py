"""Empirical Telemetry Logger for Section 4 (Results) and Section 5 (Ablations) of paper.
Logs per-sample execution traces over evaluation splits (val.jsonl, test.jsonl).
"""
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .state import AgentState


class EmpiricalLogger:
    """Logs detailed execution telemetry for each sample evaluated by the multi-agent system."""

    def __init__(self, log_path: Optional[str] = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: List[Dict[str, Any]] = []

    def log_run(
        self,
        state: AgentState,
        sample_id: Optional[str] = None,
        elapsed_seconds: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Extracts telemetry from AgentState and appends to empirical logs."""
        casual_input = state.get("casual_input") or state.get("raw_input") or ""
        final_output = state.get("final_output") or (state.get("drafts") or [casual_input])[-1]

        # Token and timing calculations
        usage = state.get("usage", [])
        total_calls = sum(u.get("calls", 0) for u in usage)
        total_input_tokens = sum(u.get("input_tokens", 0) for u in usage)
        total_output_tokens = sum(u.get("output_tokens", 0) for u in usage)
        duration = (
            elapsed_seconds
            if elapsed_seconds is not None
            else round(sum(u.get("seconds", 0.0) for u in usage), 3)
        )

        # Critique reasons
        rejection_reasons = []
        for c in state.get("critique_history", []):
            if "rejected" in c.lower() or "error" in c.lower() or "critique" in c.lower():
                rejection_reasons.append(c)

        entry = {
            "sample_id": sample_id or f"sample_{len(self.entries) + 1:04d}",
            "casual_input": casual_input,
            "final_output": final_output,
            "target_domain": state.get("target_domain") or state.get("scenario"),
            "requested_register": state.get("requested_register"),
            "iterations_to_converge": state.get("iteration_count", state.get("revision_count", 0)),
            "fallback_triggered": bool(state.get("fallback_triggered", False)),
            "is_approved": bool(state.get("is_approved", False)),
            "status": state.get("status", "unknown"),
            "rejection_reasons": rejection_reasons,
            "latency_seconds": duration,
            "token_usage": {
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_input_tokens + total_output_tokens,
            },
            "calls_count": total_calls,
            "timestamp": time.time(),
        }

        self.entries.append(entry)

        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        return entry

    def get_summary_stats(self) -> Dict[str, Any]:
        """Calculates aggregate statistics for the empirical results section."""
        total = len(self.entries)
        if total == 0:
            return {"total_samples": 0}

        approved_count = sum(1 for e in self.entries if e["is_approved"])
        fallback_count = sum(1 for e in self.entries if e["fallback_triggered"])
        avg_iterations = sum(e["iterations_to_converge"] for e in self.entries) / total
        avg_latency = sum(e["latency_seconds"] for e in self.entries) / total
        avg_tokens = sum(e["token_usage"]["total_tokens"] for e in self.entries) / total

        # Distribution of convergence iterations
        iter_dist = {}
        for e in self.entries:
            it = e["iterations_to_converge"]
            iter_dist[it] = iter_dist.get(it, 0) + 1

        return {
            "total_samples": total,
            "approval_rate": round(approved_count / total, 4),
            "fallback_rate": round(fallback_count / total, 4),
            "avg_iterations_to_converge": round(avg_iterations, 3),
            "iteration_distribution": iter_dist,
            "avg_latency_seconds": round(avg_latency, 3),
            "avg_total_tokens": round(avg_tokens, 1),
        }
