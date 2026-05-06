"""
Input Guardrail
Checks user inputs for safety violations.

Policy categories enforced:
1. harmful_content   - Violence, self-harm, weapons, illegal activity
2. prompt_injection  - Attempts to hijack or override agent instructions
3. off_topic_queries - Queries unrelated to HCI/research purpose
"""

from typing import Dict, Any, List
import re
import logging


class InputGuardrail:
    """
    Guardrail for checking input safety.

    Enforces three policy categories:
    - Harmful content detection (keyword + pattern based)
    - Prompt injection detection
    - Off-topic query detection (HCI relevance check)
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("guardrails.input")

        self.min_length = 5
        self.max_length = 2000

        safety_config = config.get("safety", {})
        self.prohibited_categories = safety_config.get("prohibited_categories", [
            "harmful_content", "prompt_injection", "off_topic_queries",
        ])

        # Policy 1: Harmful content — regex (high) + keywords (medium)
        self.harmful_patterns = [
            r"\b(how to (make|build|create|synthesize) (a )?bomb)\b",
            r"\b(make|build|create).{0,20}(weapon|explosive|poison)\b",
            r"\bstep[- ]by[- ]step.{0,30}(kill|attack|harm)\b",
            r"\b(how to|ways to).{0,20}(kill (my|your)?self|commit suicide|self.harm)\b",
            r"\b(hack into|crack|bypass security|steal (credentials|passwords))\b",
            r"\b(child (porn|abuse|exploit))\b",
        ]
        self.harmful_keywords = [
            "how to kill", "how to hurt", "how to attack",
            "make a bomb", "build a weapon", "synthesize poison",
            "ddos", "ransomware", "malware", "phishing",
        ]

        # Policy 2: Prompt injection patterns
        self.injection_patterns = [
            r"ignore (all |previous |prior |your )?(instructions?|rules?|guidelines?|constraints?)",
            r"forget (everything|all instructions|your instructions)",
            r"(you are now|pretend (you are|to be)|act as).{0,30}(unrestricted|jailbreak|DAN|evil)",
            r"\bsystem\s*:\s*",
            r"\[system\]",
            r"(reveal|show|print|output).{0,20}(system prompt|instructions|your prompt)",
            r"disregard.{0,20}(safety|guidelines|rules)",
            r"\bsudo\b",
            r"override (safety|restrictions|guidelines)",
        ]

        # Policy 3: HCI relevance keywords
        self.hci_keywords = [
            "hci", "human computer interaction", "user interface", "ui", "ux",
            "user experience", "usability", "accessibility", "design",
            "research", "study", "paper", "literature", "review", "survey",
            "trend", "method", "approach", "framework", "model", "theory",
            "interaction", "interface", "prototype", "evaluation", "user study",
            "cognitive", "mental model", "affordance", "feedback", "navigation",
            "ar", "vr", "augmented reality", "virtual reality", "mobile", "web",
            "ai", "machine learning", "chatbot", "voice", "gesture", "touch",
            "visualization", "data", "dashboard", "information", "display",
            "elderly", "children", "novice", "expert", "disability",
            "explainable", "transparent", "trust", "privacy", "ethics",
            "what", "how", "why", "compare", "difference", "best practice",
            "explain", "describe", "summarize", "overview", "latest",
        ]

        self.logger.info("InputGuardrail initialized with 3 policy categories")

    def validate(self, query: str) -> Dict[str, Any]:
        """
        Validate input query against all policy categories.

        Returns dict with: valid, violations, sanitized_input, action, category
        """
        if not query or not query.strip():
            return {
                "valid": False,
                "violations": [{"validator": "length", "reason": "Empty query", "severity": "low"}],
                "sanitized_input": query,
                "action": "block",
                "category": "empty_input",
            }

        violations = []
        query_lower = query.lower().strip()

        violations.extend(self._check_length(query))
        violations.extend(self._check_toxic_language(query_lower))
        violations.extend(self._check_prompt_injection(query_lower))
        violations.extend(self._check_relevance(query_lower))

        has_high = any(v["severity"] == "high" for v in violations)
        has_medium = any(v["severity"] == "medium" for v in violations)

        if has_high:
            action = "block"
        elif has_medium:
            action = "warn"
        else:
            action = "allow"

        category = violations[0].get("validator") if violations else None
        sanitized = query[:self.max_length] if len(query) > self.max_length else query
        is_valid = action != "block"

        if not is_valid:
            self.logger.warning(f"Input blocked — category={category}, preview={query[:60]!r}")

        return {
            "valid": is_valid,
            "violations": violations,
            "sanitized_input": sanitized,
            "action": action,
            "category": category,
        }

    def _check_length(self, query: str) -> List[Dict[str, Any]]:
        violations = []
        if len(query) < self.min_length:
            violations.append({"validator": "length", "reason": f"Query too short (min {self.min_length} chars)", "severity": "low"})
        if len(query) > self.max_length:
            violations.append({"validator": "length", "reason": "Query too long — will be truncated", "severity": "medium"})
        return violations

    def _check_toxic_language(self, text: str) -> List[Dict[str, Any]]:
        """Policy 1 — Harmful content detection."""
        violations = []
        for pattern in self.harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "harmful_content",
                    "reason": "Query contains potentially harmful instructions or content",
                    "severity": "high",
                    "category": "harmful_content",
                })
                return violations
        for keyword in self.harmful_keywords:
            if keyword in text:
                violations.append({
                    "validator": "harmful_content",
                    "reason": f"Query contains harmful keyword: '{keyword}'",
                    "severity": "medium",
                    "category": "harmful_content",
                })
                break
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        """Policy 2 — Prompt injection / jailbreak detection."""
        violations = []
        for pattern in self.injection_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "prompt_injection",
                    "reason": "Query appears to attempt prompt injection or instruction override",
                    "severity": "high",
                    "category": "prompt_injection",
                })
                return violations
        return violations

    def _check_relevance(self, query: str) -> List[Dict[str, Any]]:
        """Policy 3 — Off-topic query detection (warn only, no block)."""
        violations = []
        if not any(kw in query for kw in self.hci_keywords):
            violations.append({
                "validator": "off_topic",
                "reason": "Query does not appear to be related to HCI or research.",
                "severity": "low",
                "category": "off_topic_queries",
            })
        return violations
