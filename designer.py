#!/usr/bin/env python3
"""
designer.py - Modular Designer Agent with graph-based workflow
"""

from dotenv import load_dotenv
from typing import List, Dict, Optional, Annotated
from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import stream_graph, random_uuid

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# Load environment variables for API keys
load_dotenv()

# Memory for graph checkpoints
memory = MemorySaver()

# Define state type for graph
class State(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Tool implementations
tools_list = []

@tool
def color_gen(query: str) -> List[Dict[str, str]]:
    """Generate a palette of colors based on a semantic query"""
    if query.lower().startswith("warm"):
        return [{"red": "#FF0000"}, {"orange": "#FFA500"}, {"yellow": "#FFFF00"}]
    return [{"blue": "#0000FF"}, {"green": "#008000"}, {"purple": "#800080"}]
tools_list.append(color_gen)

@tool
def font_gen(query: str) -> List[Dict[str, str]]:
    """Generate a list of font suggestions based on a semantic query"""
    if query.lower().startswith("warm"):
        return [{"Arial": "sans-serif"}, {"Times New Roman": "serif"}, {"Courier New": "monospace"}]
    return [{"Helvetica": "sans-serif"}, {"Georgia": "serif"}, {"Comic Sans MS": "cursive"}]
tools_list.append(font_gen)

class DesignerAgent:
    """
    Encapsulates the designer agent using LangGraph workflow and LangChain tools.
    """
    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        # Bind LLM with tools
        self.llm = ChatOpenAI(model=model_name, temperature=temperature).bind_tools(tools_list)
        # Graph tool node
        self.tool_node = ToolNode(tools_list)
        # Build workflow graph
        self.app = self._build_workflow().compile(checkpointer=memory)

    def _build_workflow(self) -> StateGraph:
        graph = StateGraph(State)

        # Designer node: calls LLM with prompt template
        def designer_node(state: State) -> State:
            prompt = ChatPromptTemplate.from_messages([
                ("system", 
                    "You are a professional designer specialized in promotional materials. "
                    "Plan and produce color and font suggestions using available tools."),
                MessagesPlaceholder("messages"),
            ])
            chain = prompt | self.llm
            response = chain.invoke({"messages": state["messages"]})
            return {"messages": [AIMessage(content=response)]}

        # Add nodes
        graph.add_node("designer", designer_node)
        graph.add_node("tools", self.tool_node)

        # Define flow
        graph.add_edge(START, "designer")
        graph.add_conditional_edges("designer", tools_condition)
        graph.add_edge("tools", "designer")
        graph.add_edge("designer", END)

        return graph

    def stream(self, user_input: str, recursion_limit: int = 10) -> None:
        """Stream the workflow execution for a given user input."""
        inputs = {"messages": [HumanMessage(content=user_input)]}
        config = RunnableConfig(recursion_limit=recursion_limit, configurable={"thread_id": random_uuid()})
        stream_graph(self.app, inputs, config)

# Example usage
if __name__ == "__main__":
    agent = DesignerAgent()
    agent.stream("안녕하세요? 어떤 색과 글꼴이 따뜻한 느낌을 줄까요?")