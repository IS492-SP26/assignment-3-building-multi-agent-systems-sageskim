"""
AutoGen-Based Orchestrator
Runs all async operations (search + agents) in a single event loop.
"""

import logging
import asyncio
import re
from typing import Dict, Any, List

from src.agents.autogen_agents import create_research_team
from src.guardrails.safety_manager import SafetyManager
from src.tools.web_search import WebSearchTool
from src.tools.paper_search import PaperSearchTool


class AutoGenOrchestrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        self.safety_manager = SafetyManager(config)
        self.workflow_trace: List[Dict[str, Any]] = []

    def process_query(self, query: str) -> Dict[str, Any]:
        self.logger.info(f"Processing query: {query}")

        # Input safety check (sync)
        input_safety = self.safety_manager.check_input_safety(query)
        if not input_safety["safe"]:
            return {
                "query": query,
                "response": input_safety.get("refusal_message", "Request blocked by safety policy."),
                "conversation_history": [],
                "metadata": {
                    "safety_blocked": True,
                    "safety_events": [{"type": "input", "action": input_safety.get("action", "block"), "violations": input_safety.get("violations", [])}],
                    "num_messages": 0,
                    "num_sources": 0,
                }
            }
        query = input_safety.get("query", query)

        try:
            # Run everything in ONE event loop
            result = asyncio.run(self._full_pipeline(query))

            # Output safety check (sync)
            output_safety = self.safety_manager.check_output_safety(result.get("response", ""))
            if output_safety.get("action") in ("sanitize", "refuse"):
                result["response"] = output_safety["response"]
                result.setdefault("metadata", {}).setdefault("safety_events", []).append({
                    "type": "output",
                    "action": output_safety.get("action"),
                    "violations": output_safety.get("violations", []),
                })
            return result

        except Exception as e:
            self.logger.error(f"Error: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True}
            }

    async def _full_pipeline(self, query: str) -> Dict[str, Any]:
        """Run search + agents all inside one event loop."""
        # Fetch search results concurrently using async tool classes
        web_tool = WebSearchTool(provider="tavily", max_results=5)
        paper_tool = PaperSearchTool(max_results=5)

        try:
            web_results_raw = await asyncio.wait_for(web_tool.search(query), timeout=15)
        except Exception as e:
            self.logger.warning(f"Web search failed: {e}")
            web_results_raw = []
        
        try:
            paper_results_raw = await asyncio.wait_for(paper_tool.search(query), timeout=15)
        except Exception as e:
            self.logger.warning(f"Paper search failed: {e}")
            paper_results_raw = []

        # Format results
        web_results = self._format_web(query, web_results_raw)
        paper_results = self._format_papers(query, paper_results_raw)

        # Build task message with pre-fetched evidence
        task_message = f"""Research Query: {query}

Use the following pre-fetched evidence to answer the query:

--- WEB SEARCH RESULTS ---
{web_results}

--- ACADEMIC PAPERS ---
{paper_results}
---

Instructions:
1. Planner: Create a brief research plan based on the evidence above.
2. Researcher: Summarize key findings from the evidence with inline citations.
3. Writer: Write a well-structured research report with a References section at the end.
4. Critic: Give 1-2 sentences of feedback on quality."""

        # Run agent team
        team = create_research_team(self.config)
        result = await team.run(task=task_message)

        def clean(text):
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            return text.replace("TERMINATE", "").strip()

        messages = [
            {"source": msg.source, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in result.messages
        ]

        # Get Writer's response
        final_response = ""
        for msg in messages:
            if msg["source"] == "Writer":
                cleaned = clean(msg["content"])
                if len(cleaned) > 50:
                    final_response = cleaned

        # Fallback to Researcher
        if not final_response:
            for msg in messages:
                if msg["source"] == "Researcher":
                    cleaned = clean(msg["content"])
                    if len(cleaned) > 50:
                        final_response = cleaned

        if not final_response and messages:
            final_response = clean(messages[-1]["content"])

        return self._extract_results(query, messages, final_response)

    def _format_web(self, query: str, results) -> str:
        if isinstance(results, Exception) or not results:
            return "No web search results available."
        output = f"Found {len(results)} web search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            output += f"{i}. {r.get('title', '')}\n   URL: {r.get('url', '')}\n   {r.get('snippet', '')}\n\n"
        return output

    def _format_papers(self, query: str, results) -> str:
        if isinstance(results, Exception) or not results:
            return "No academic papers found."
        output = f"Found {len(results)} academic papers for '{query}':\n\n"
        for i, p in enumerate(results, 1):
            authors = ", ".join([a["name"] for a in p.get("authors", [])[:3]])
            output += f"{i}. {p.get('title', '')}\n   Authors: {authors}\n   Year: {p.get('year', '')} | Citations: {p.get('citation_count', '')}\n   URL: {p.get('url', '')}\n\n"
        return output

    def _extract_results(self, query: str, messages: List[Dict], final_response: str = "") -> Dict[str, Any]:
        plan, research_findings, critique = "", [], ""
        for msg in messages:
            src = msg.get("source", "")
            content = msg.get("content", "")
            if src == "Planner" and not plan:
                plan = content
            elif src == "Researcher":
                research_findings.append(content)
            elif src == "Critic":
                critique = content

        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": max(len(research_findings), 1),
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list(set(m.get("source", "") for m in messages)),
            }
        }

    def get_agent_descriptions(self) -> Dict[str, str]:
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Analyzes pre-fetched evidence from web and academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and provides feedback",
        }
