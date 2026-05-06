"""
Safety Manager
Coordinates input/output guardrails and logs all safety events.
"""

from typing import Dict, Any, List, Optional
import logging
import json
from datetime import datetime
from pathlib import Path

from src.guardrails.input_guardrail import InputGuardrail
from src.guardrails.output_guardrail import OutputGuardrail


class SafetyManager:
    """
    Coordinates InputGuardrail and OutputGuardrail, logs safety events,
    and exposes a clean API for the orchestrator and UI.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        safety_config = config.get("safety", {})

        self.enabled = safety_config.get("enabled", True)
        self.log_events = safety_config.get("log_events", True)
        self.logger = logging.getLogger("safety")

        # Safety event log (in-memory, also written to file)
        self.safety_events: List[Dict[str, Any]] = []

        # Initialize sub-guardrails
        self.input_guardrail = InputGuardrail(config)
        self.output_guardrail = OutputGuardrail(config)

        # Violation response strategy from config
        self.on_violation = safety_config.get("on_violation", {
            "action": "refuse",
            "message": "I cannot process this request due to safety policies.",
        })

        # Safety log file path
        log_config = config.get("logging", {})
        self.safety_log_file = log_config.get("safety_log", "logs/safety_events.log")

        # Ensure log directory exists
        Path(self.safety_log_file).parent.mkdir(parents=True, exist_ok=True)

        self.logger.info("SafetyManager initialized (enabled=%s)", self.enabled)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        """
        Run input guardrail on user query.

        Returns:
            Dict with keys: safe, query, violations, action, category
        """
        if not self.enabled:
            return {"safe": True, "query": query, "violations": [], "action": "allow"}

        result = self.input_guardrail.validate(query)

        is_safe = result["valid"]
        violations = result["violations"]
        action = result.get("action", "allow")
        category = result.get("category")

        # Log any violations
        if violations and self.log_events:
            self._log_safety_event(
                event_type="input",
                content=query,
                violations=violations,
                is_safe=is_safe,
                action=action,
            )

        return {
            "safe": is_safe,
            "query": result.get("sanitized_input", query),
            "violations": violations,
            "action": action,
            "category": category,
            # Human-readable refusal message for the UI
            "refusal_message": self._build_refusal_message(category) if not is_safe else None,
        }

    def check_output_safety(
        self,
        response: str,
        sources: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Run output guardrail on generated response.

        Returns:
            Dict with keys: safe, response, violations, action
        """
        if not self.enabled:
            return {"safe": True, "response": response, "violations": [], "action": "allow"}

        result = self.output_guardrail.validate(response, sources)

        is_safe = result["valid"]
        violations = result["violations"]
        action = result.get("action", "allow")
        final_response = result.get("sanitized_output", response)

        # Log any violations
        if violations and self.log_events:
            self._log_safety_event(
                event_type="output",
                content=response,
                violations=violations,
                is_safe=is_safe,
                action=action,
            )

        return {
            "safe": is_safe,
            "response": final_response,
            "violations": violations,
            "action": action,
        }

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log_safety_event(
        self,
        event_type: str,
        content: str,
        violations: List[Dict[str, Any]],
        is_safe: bool,
        action: str,
    ):
        """Record a safety event in memory and to file."""
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,          # "input" | "output"
            "safe": is_safe,
            "action": action,            # "allow" | "warn" | "block" | "sanitize" | "refuse"
            "violations": violations,
            "content_preview": content[:120] + "..." if len(content) > 120 else content,
        }

        self.safety_events.append(event)
        self.logger.warning(
            "Safety event | type=%s | safe=%s | action=%s | violations=%d",
            event_type, is_safe, action, len(violations),
        )

        # Persist to log file
        if self.log_events:
            try:
                with open(self.safety_log_file, "a") as f:
                    f.write(json.dumps(event) + "\n")
            except Exception as e:
                self.logger.error("Failed to write safety log: %s", e)

    def _build_refusal_message(self, category: Optional[str]) -> str:
        """Return a user-friendly refusal message based on the policy category."""
        messages = {
            "harmful_content": (
                "🚫 This request was blocked because it appears to ask for harmful or dangerous content. "
                "Please ask something related to HCI research."
            ),
            "prompt_injection": (
                "🚫 This request was blocked because it appears to attempt to override system instructions. "
                "Please ask a genuine research question."
            ),
            "off_topic_queries": (
                "⚠️ This query doesn't seem related to HCI research. "
                "This assistant specialises in Human-Computer Interaction topics."
            ),
            "empty_input": "Please enter a research query.",
        }
        return messages.get(
            category or "",
            self.on_violation.get(
                "message",
                "🚫 This request was blocked due to safety policies."
            ),
        )

    # ------------------------------------------------------------------
    # Statistics / reporting
    # ------------------------------------------------------------------

    def get_safety_events(self) -> List[Dict[str, Any]]:
        """Return all logged safety events."""
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        """Aggregate statistics for display in the UI."""
        total = len(self.safety_events)
        input_events  = sum(1 for e in self.safety_events if e["type"] == "input")
        output_events = sum(1 for e in self.safety_events if e["type"] == "output")
        blocked       = sum(1 for e in self.safety_events if e["action"] in ("block", "refuse"))
        sanitized     = sum(1 for e in self.safety_events if e["action"] == "sanitize")

        return {
            "total_checks":    total,
            "input_checks":    input_events,
            "output_checks":   output_events,
            "blocked":         blocked,
            "sanitized":       sanitized,
            "violation_rate":  blocked / total if total > 0 else 0.0,
        }

    def clear_events(self):
        """Clear in-memory safety event log."""
        self.safety_events = []
