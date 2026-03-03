"""
LANGGRAPH STATE
Defines the central state object that flows through every node.
Every node reads from this state and returns an updated version of it.
"""

from typing import TypedDict, List, Optional, Literal
from langchain_core.messages import BaseMessage


class PAState(TypedDict):
    """
    Central state object passed between all LangGraph nodes.
    
    messages       — full conversation history (HumanMessage / AIMessage)
    patients       — the processed patient records loaded from JSON
    context        — RAG-retrieved clinical context for current query
    current_query  — the clinician's latest question
    response       — the agent's latest response
    next_node      — routing signal for conditional edges
    """
    messages: List[BaseMessage]
    patients: List[dict]
    context: str
    current_query: str
    response: str
    next_node: str
