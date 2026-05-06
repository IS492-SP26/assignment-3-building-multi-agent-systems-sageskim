"""
Output Guardrail
Checks system outputs for safety violations.

Policy categories enforced:
1. harmful_content - Dangerous instructions or violent content in responses
2. pii            - Personally Identifiable Information (email, phone, SSN)
3. misinformation - Unsupported absolute claims without citations
"""

from typing import Dict, Any, List
import re
import logging


class OutputGuardrail:
    """
    Guardrail for checking output safety.

    Enforces three policy categories:
    - PII detection and redaction
    - Harmful content in responses
    - Misinformation / unsupported absolute claims
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("guardrails.output")

        safety_config = config.get("safety", {})
        self.on_violation = safety_config.get("on_violation", {})

        # PII regex patterns
        self.pii_patterns = {
            "email":   r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            "phone":   r'\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
            "ssn":     r'\b\d{3}-\d{2}-\d{4}\b',
            "credit_card": r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
        }

        # Harmful output patterns (high severity — refuse)
        self.harmful_output_patterns = [
            r"step[- ]by[- ]step.{0,40}(kill|attack|harm|bomb|poison|weapon)",
            r"(instructions?|guide|tutorial).{0,30}(make|build|create).{0,20}(weapon|bomb|explosive)",
            r"here('s| is) how to.{0,30}(hack|exploit|bypass|crack)",
        ]

        # Misinformation indicators — absolute unsourced claims (medium severity)
        self.misinformation_patterns = [
            r"\b(everyone knows|it is a fact that|undeniably|proven beyond doubt)\b",
            r"\b(100% (effective|safe|accurate|proven))\b",
            r"\b(there is no (evidence|research|study) that)\b",
        ]

        self.logger.info("OutputGuardrail initialized with 3 policy categories")

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate output response.

        Args:
            response: Generated response to validate
            sources:  Optional list of sources used (for consistency checks)

        Returns:
            Dict with: valid, violations, sanitized_output, action
        """
        violations = []

        violations.extend(self._check_pii(response))
        violations.extend(self._check_harmful_content(response))
        violations.extend(self._check_misinformation(response))

        has_high = any(v["severity"] == "high" for v in violations)
        has_medium = any(v["severity"] == "medium" for v in violations)

        if has_high:
            action = self.on_violation.get("action", "refuse")
        elif has_medium:
            action = "sanitize"
        else:
            action = "allow"

        # Build output
        if action == "refuse":
            sanitized = self.on_violation.get(
                "message",
                "I cannot provide this response due to safety policies."
            )
            self.logger.warning(f"Output REFUSED — {len(violations)} violation(s)")
        elif action == "sanitize":
            sanitized = self._sanitize(response, violations)
            self.logger.warning(f"Output SANITIZED — {len(violations)} violation(s)")
        else:
            sanitized = response

        return {
            "valid": action != "refuse",
            "violations": violations,
            "sanitized_output": sanitized,
            "action": action,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        """Detect personally identifiable information."""
        violations = []
        for pii_type, pattern in self.pii_patterns.items():
            matches = re.findall(pattern, text)
            if matches:
                violations.append({
                    "validator": "pii",
                    "pii_type": pii_type,
                    "reason": f"Response contains {pii_type.replace('_', ' ')}",
                    "severity": "high",
                    "matches": matches,
                    "category": "pii",
                })
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        """Detect harmful instructions or dangerous content in output."""
        violations = []
        for pattern in self.harmful_output_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "harmful_content",
                    "reason": "Response contains potentially harmful instructions",
                    "severity": "high",
                    "category": "harmful_content",
                })
                return violations
        return violations

    def _check_misinformation(self, text: str) -> List[Dict[str, Any]]:
        """Detect absolute unsourced claims that may indicate misinformation."""
        violations = []
        for pattern in self.misinformation_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "misinformation",
                    "reason": "Response contains absolute unsourced claim — may indicate misinformation",
                    "severity": "medium",
                    "category": "misinformation",
                })
                break
        return violations

    def _check_factual_consistency(self, response: str, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Placeholder: compare claims against retrieved sources (future work)."""
        return []

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        """Redact PII and flag misinformation in the response."""
        sanitized = text

        for violation in violations:
            # Redact PII matches
            if violation.get("validator") == "pii":
                pii_type = violation.get("pii_type", "data")
                for match in violation.get("matches", []):
                    sanitized = sanitized.replace(match, f"[REDACTED-{pii_type.upper()}]")

            # Flag misinformation with a note
            elif violation.get("validator") == "misinformation":
                sanitized += (
                    "\n\n⚠️ *Note: Some claims in this response may lack sufficient citations. "
                    "Please verify with primary sources.*"
                )
                break  # Add note once

        return sanitized
