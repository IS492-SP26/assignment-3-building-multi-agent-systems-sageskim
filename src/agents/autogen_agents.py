"""
AutoGen Agent Implementations

This module provides concrete AutoGen-based implementations of the research agents.
Each agent is implemented as an AutoGen AssistantAgent with specific tools and behaviors.

Based on the AutoGen literature review example:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/examples/literature-review.html
"""

import os
from typing import Dict, Any, List, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_core.tools import FunctionTool
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import ModelFamily
# Import our research tools
from src.tools.web_search import web_search
from src.tools.paper_search import paper_search


def create_model_client(config: Dict[str, Any]) -> OpenAIChatCompletionClient:
    """
    Create model client for AutoGen agents.
    
    Args:
        config: Configuration dictionary from config.yaml
        
    Returns:
        OpenAIChatCompletionClient configured for the specified provider
    """
    model_config = config.get("models", {}).get("default", {})
    provider = model_config.get("provider", "groq")
    
    # Groq configuration (uses OpenAI-compatible API)
    if provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        
        return OpenAIChatCompletionClient(
            model=model_config.get("name", "llama-3.3-70b-versatile"),
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            model_info={
                "json_output": False,
                "vision": False,
                "function_calling": True,
                "family": ModelFamily.UNKNOWN,
                "structured_output": False,
            },
            parallel_tool_calls=False,
        )
    
    # OpenAI configuration
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        return OpenAIChatCompletionClient(
            model=model_config.get("name", "gpt-4o-mini"),
            api_key=api_key,
            base_url=base_url,
        )

    elif provider == "vllm":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        
        return OpenAIChatCompletionClient(
            model=model_config.get("name", "gpt-4o-mini"),
            api_key=api_key,
            base_url=base_url,
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.GPT_4O,
                "structured_output": True,
            },
        )
    
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def create_planner_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Planner Agent using AutoGen.
    
    The planner breaks down research queries into actionable steps.
    It doesn't use tools, but provides strategic direction.
    
    Args:
        config: Configuration dictionary
        model_client: Model client for the agent
        
    Returns:
        AutoGen AssistantAgent configured as a planner
    """
    agent_config = config.get("agents", {}).get("planner", {})
    
    # Load system prompt from config or use default
    default_system_message = """You are a Research Planner. Your job is to break down research queries into clear, actionable steps.

When given a research query, you should:
1. Identify the key concepts and topics to investigate
2. Determine what types of sources would be most valuable (academic papers, web articles, etc.)
3. Suggest specific search queries for the Researcher
4. Outline how the findings should be synthesized

Provide your plan in a structured format with numbered steps.
Be specific about what information to gather and why it's relevant."""

    # Use custom prompt from config if available, otherwise use default
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a task planner. Break down research queries into actionable steps.":
        system_message = custom_prompt
    else:
        system_message = default_system_message

    planner = AssistantAgent(
        name="Planner",
        model_client=model_client,
        description="Breaks down research queries into actionable steps",
        system_message=system_message,
    )
    
    return planner


def create_researcher_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Researcher Agent using AutoGen.
    
    The researcher has access to web search and paper search tools.
    It gathers evidence based on the planner's guidance.
    
    Args:
        config: Configuration dictionary
        model_client: Model client for the agent
        
    Returns:
        AutoGen AssistantAgent configured as a researcher with tool access
    """
    agent_config = config.get("agents", {}).get("researcher", {})
    
    # Load system prompt from config or use default
    default_system_message = """You are a Research Assistant. Your job is to gather high-quality information from academic papers and web sources.

You have access to two tools: web_search and paper_search. When conducting research:
1. IMPORTANT: Call only ONE tool at a time. Never call multiple tools simultaneously.
2. First call web_search to find recent articles and web sources.
3. Then call paper_search to find academic papers.
4. Extract key findings, quotes, and data from results.
5. Note all source URLs and citations.
6. After gathering enough sources (at least 3-5), summarize your findings clearly."""

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a researcher. Find and collect relevant information from various sources.":
        system_message = custom_prompt
    else:
        system_message = default_system_message

    # No tool calling — evidence is pre-fetched by the orchestrator
    researcher = AssistantAgent(
        name="Researcher",
        model_client=model_client,
        description="Analyzes pre-fetched evidence from web and academic sources",
        system_message=system_message,
    )
    
    return researcher


def create_writer_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Writer Agent using AutoGen.
    
    The writer synthesizes research findings into coherent responses with proper citations.
    
    Args:
        config: Configuration dictionary
        model_client: Model client for the agent
        
    Returns:
        AutoGen AssistantAgent configured as a writer
    """
    agent_config = config.get("agents", {}).get("writer", {})
    
    # Load system prompt from config or use default
    default_system_message = """You are a Research Writer. Your job is to synthesize research findings into clear, well-organized responses.

When writing:
1. Start with an overview/introduction
2. Present findings in a logical structure
3. Cite sources inline using [Source: Title/Author]
4. Synthesize information from multiple sources
5. Avoid copying text directly - paraphrase and synthesize
6. Include a references section at the end
7. Ensure the response directly answers the original query

Format your response professionally with clear headings, paragraphs, in-text citations, and a References section at the end."""

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a writer. Synthesize research findings into a coherent report.":
        system_message = custom_prompt
    else:
        system_message = default_system_message

    writer = AssistantAgent(
        name="Writer",
        model_client=model_client,
        description="Synthesizes research findings into coherent, well-cited responses",
        system_message=system_message,
    )
    
    return writer


def create_critic_agent(config: Dict[str, Any], model_client: OpenAIChatCompletionClient) -> AssistantAgent:
    """
    Create a Critic Agent using AutoGen.
    
    The critic evaluates the quality of the research and writing,
    providing feedback for improvement.
    
    Args:
        config: Configuration dictionary
        model_client: Model client for the agent
        
    Returns:
        AutoGen AssistantAgent configured as a critic
    """
    agent_config = config.get("agents", {}).get("critic", {})
    
    # Load system prompt from config or use default
    default_system_message = """You are a Research Critic. Write 1-2 sentences of feedback on the Writer response quality (relevance, evidence, clarity).
Always end your message with: TERMINATE"""

    # Use custom prompt from config if available
    custom_prompt = agent_config.get("system_prompt", "")
    if custom_prompt and custom_prompt != "You are a critic. Evaluate the quality and accuracy of research findings.":
        system_message = custom_prompt
    else:
        system_message = default_system_message

    critic = AssistantAgent(
        name="Critic",
        model_client=model_client,
        description="Evaluates research quality and provides feedback",
        system_message=system_message,
    )
    
    return critic


def create_research_team(config: Dict[str, Any]) -> RoundRobinGroupChat:
    """
    Create the research team as a RoundRobinGroupChat.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RoundRobinGroupChat with all agents configured
    """
    # Create model client (shared by all agents)
    model_client = create_model_client(config)
    
    # Create all agents
    planner = create_planner_agent(config, model_client)
    researcher = create_researcher_agent(config, model_client)
    writer = create_writer_agent(config, model_client)
    critic = create_critic_agent(config, model_client)
    
    # Create termination condition
    termination = TextMentionTermination("TERMINATE")
    
    # Create team with round-robin ordering
    team = RoundRobinGroupChat(
        participants=[planner, researcher, writer, critic],
        termination_condition=termination,
    )
    
    return team

