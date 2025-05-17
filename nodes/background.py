"""
배경 관련 노드 모듈
"""
from langchain_core.messages import AIMessage

from nodes.common import State
from tools.background import background_gen


def background_node(state: State) -> State:
    """배경을 생성하는 노드 함수"""
    result = f"background: {background_gen.invoke('gradient')}"
    return {"messages": [AIMessage(str(result))]}
