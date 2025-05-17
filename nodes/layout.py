"""
레이아웃 관련 노드 모듈
"""
from langchain_core.messages import AIMessage

from nodes.common import State
from tools.layout import layout_gen


def layout_node(state: State) -> State:
    """레이아웃을 생성하는 노드 함수"""
    result = f"layout: {layout_gen.invoke('simple')}"
    return {"messages": [AIMessage(result)]}
