"""
공통 상태 및 도구 정의 모듈
"""
from typing import Annotated, Tuple
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_teddynote.tools.tavily import TavilySearch
from pydantic import Field
from dotenv import load_dotenv

from tools.color import color_gen
from tools.background import background_gen
from tools.layout import layout_gen

# 환경 변수 로드
load_dotenv()

# --- 툴 인스턴스화 ---
tavily = TavilySearch(max_results=3)
designer_tools = [tavily]
drawer_tools = [color_gen]

# --- 상태 정의 ---
class State(TypedDict):
    canvas_size: Annotated[Tuple[int, int],
        Field(
            ...,
            gt=0,
            description="캔버스 크기 (width, height) 형식의 튜플, 단위는 픽셀",
            example=(800, 600),
        ),
    ]
    messages: Annotated[list, add_messages]
