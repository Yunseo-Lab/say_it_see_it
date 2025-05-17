"""
그리기 관련 노드 모듈
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from nodes.common import State, drawer_tools
from utils import load_system_template


def drawer(state: State) -> State:
    """그림을 생성하는 노드 함수"""
    system_template = load_system_template("/Users/localgroup/Documents/workspace/say_it_see_it/prompts/drawer_prompt.yaml")

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_template),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    llm = ChatOpenAI(model="gpt-4o-mini")
    # llm_with_tools = llm.bind_tools(drawer_tools)
    chain = prompt | llm
    return {"messages": [chain.invoke({"messages": state["messages"]})]}
