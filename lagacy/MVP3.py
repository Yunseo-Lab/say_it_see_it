# langgraph_design_pipeline.py
# 목적: 자동 디자인 파이프라인을 LangGraph + 전략 패턴 + 클래스 기반으로 설계하여
#       재사용성과 확장성을 극대화함

from abc import ABC, abstractmethod
from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph

# =============================
# 0. 상태 정의
# =============================
class DesignState(TypedDict, total=False):
    bg: str
    layout: Dict[str, tuple]
    color_palette: List[str]
    font: str
    decoration: str
    status: str

# =============================
# 1. 공통 인터페이스 정의
# =============================
class DesignStep(ABC):
    """
    모든 디자인 작업 단계 클래스는 DesignStep 인터페이스를 구현해야 함.
    LangGraph에 연결될 때 이 인터페이스의 execute()가 호출됨.
    """
    @abstractmethod
    def execute(self, state: DesignState) -> DesignState:
        pass

# =============================
# 2. 도구 인터페이스 및 구현 정의
# =============================

class Tool(ABC):
    """모든 디자인 도구의 공통 인터페이스"""
    @abstractmethod
    def apply(self, state: DesignState) -> DesignState:
        pass

class ColorGen(Tool):
    def apply(self, state: DesignState) -> DesignState:
        state["color_palette"] = ["#FF9900", "#333333"]
        print("[ColorGen] 색상 팔레트 적용")
        return state

class FontGen(Tool):
    def apply(self, state: DesignState) -> DesignState:
        state["font"] = "Pretendard"
        print("[FontGen] 폰트 적용")
        return state

class DecoGen(Tool):
    def apply(self, state: DesignState) -> DesignState:
        state["decoration"] = "sparkle"
        print("[DecoGen] 데코레이션 적용")
        return state

# =============================
# 3. 개별 디자인 단계 클래스 정의
# =============================

class BackgroundDesigner(DesignStep):
    """배경을 생성하는 초기 단계"""
    def execute(self, state: DesignState) -> DesignState:
        print("[BackgroundDesigner] 배경 생성 중...")
        state["bg"] = "gradient-blue"
        return state

class LayoutDesigner(DesignStep):
    """레이아웃을 설계하고 툴을 활용하는 단계"""
    def __init__(self, tools: List[Tool]):
        self.tools = tools

    def execute(self, state: DesignState) -> DesignState:
        print("[LayoutDesigner] 레이아웃 설계 중...")
        state["layout"] = {
            "title": (10, 10, 300, 60),
            "image": (10, 80, 300, 200)
        }
        for tool in self.tools:
            state = tool.apply(state)
        return state

class Designer(DesignStep):
    """색상, 폰트, 데코레이션 등을 종합적으로 결정하는 단계"""
    def __init__(self, tools: List[Tool]):
        self.tools = tools

    def execute(self, state: DesignState) -> DesignState:
        print("[Designer] 종합 디자인 적용 중...")
        for tool in self.tools:
            state = tool.apply(state)
        return state

class Finish(DesignStep):
    """최종 마무리 및 결과 출력 단계"""
    def execute(self, state: DesignState) -> DesignState:
        print("[Finish] 최종 디자인 결과 완료 ✅")
        state["status"] = "done"
        return state

# =============================
# 4. 그래프 생성기 클래스 정의
# =============================

class GraphPlanner:
    """
    디자인 파이프라인 전체를 LangGraph 기반으로 구성하는 클래스
    """
    def __init__(self):
        # 도구 인스턴스화
        layout_tools: List[Tool] = []  # 필요 시 LayoutTool 인터페이스도 정의 가능
        design_tools: List[Tool] = [ColorGen(), FontGen(), DecoGen()]

        # 각 노드 이름과 실행 객체를 매핑
        self.steps: Dict[str, DesignStep] = {
            "BackgroundDesigner": BackgroundDesigner(),
            "LayoutDesigner": LayoutDesigner(layout_tools),
            "Designer": Designer(design_tools),
            "FINISH": Finish()
        }

    def build_graph(self):
        """
        LangGraph를 구성하고 노드 및 흐름을 등록함
        """
        graph = StateGraph()

        # 각 노드 등록
        for name, step in self.steps.items():
            graph.add_node(name, step.execute)

        # 진입점과 전이 정의
        graph.set_entry_point("BackgroundDesigner")
        graph.add_edge("BackgroundDesigner", "LayoutDesigner")
        graph.add_edge("LayoutDesigner", "Designer")
        graph.add_edge("Designer", "FINISH")

        # 종료 상태 정의
        graph.set_finish_point("FINISH")

        return graph.build()

# =============================
# 5. 메인 실행 함수
# =============================
if __name__ == "__main__":
    planner = GraphPlanner()
    design_graph = planner.build_graph()

    # 초기 상태로 그래프 실행
    initial_state: DesignState = {}
    final_state = design_graph.invoke(initial_state)

    print("\n🎯 최종 디자인 결과:")
    for k, v in final_state.items():
        print(f"{k}: {v}")
