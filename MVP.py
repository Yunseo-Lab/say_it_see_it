from abc import ABC, abstractmethod
from typing import Dict, Any, TypedDict, Optional, Tuple
import langgraph

# === 데이터 타입 정의 ===

class State(TypedDict, total=False):
    query: str                            # 사용자 쿼리 (예: "소화가 잘되는 우유")
    size: str                             # 홍보물 사이즈 (예: "1080x1920")
    photo: Optional[str]                  # 제품 사진 경로 (예: "milk.png")
    layout: Dict[str, Tuple[int, int, int, int]]  # 요소별 위치 정보 (x1, y1, x2, y2)
    text: Dict[str, str]                  # 생성된 문구 (title, description 등)
    bg: str                               # 생성된 배경 이미지 경로
    style: Dict[str, Any]                 # 폰트 및 색상 스타일 정보
    deco: Dict[str, Any]                  # 장식 요소 정보 (스티커, 사진 등)
    result: str                           # 최종 렌더링 결과 (HTML)

# === 인터페이스 정의 ===

class LayoutGenerator(ABC):
    """
    레이아웃 생성기 인터페이스
    입력: query(str), size(str), bg(str|None)
    출력: Dict[str, Tuple[int, int, int, int]] (요소별 bounding box)
    """
    @abstractmethod
    def generate(self, query: str, size: str, bg: Optional[str] = None) -> Dict[str, Tuple[int, int, int, int]]: pass

class LayoutRefiner(ABC):
    """
    기존 레이아웃을 보정하는 인터페이스
    입력: 기존 layout, query, size, bg
    출력: 보정된 layout (요소 위치 변경/추가)
    """
    @abstractmethod
    def refine(self, layout: Dict[str, Tuple[int, int, int, int]], query: str, size: str, bg: Optional[str] = None) -> Dict[str, Tuple[int, int, int, int]]: pass

class CopyWriter(ABC):
    """
    문구 생성기 인터페이스
    입력: query, layout
    출력: text dict (title, description 등)
    """
    @abstractmethod
    def write(self, query: str, layout: Dict[str, Any]) -> Dict[str, str]: pass

class BackgroundMaker(ABC):
    """
    배경 생성기 인터페이스
    입력: theme(str) or query
    출력: 배경 이미지 파일 경로
    """
    @abstractmethod
    def create(self, theme: str) -> str: pass

class FontStyler(ABC):
    """
    폰트 스타일러 인터페이스
    입력: text, bg 이미지
    출력: font 정보 dict (예: 폰트, 색상)
    """
    @abstractmethod
    def style(self, text: Dict[str, str], bg: str) -> Dict[str, Any]: pass

class Decorator(ABC):
    """
    장식 요소 삽입기 인터페이스
    입력: layout, photo(optional)
    출력: deco dict (스티커, 사진 위치 등)
    """
    @abstractmethod
    def decorate(self, layout: Dict[str, Any], photo: Optional[str] = None) -> Dict[str, Any]: pass

class Renderer(ABC):
    """
    렌더링 엔진 인터페이스
    입력: layout, text, style, deco
    출력: HTML 등 최종 결과물 string
    """
    @abstractmethod
    def render(self, layout: Dict[str, Tuple[int, int, int, int]], text: Dict[str, str], style: Dict[str, Any], deco: Dict[str, Any]) -> str: pass

# === 실제 컴포넌트 구현 ===

class BasicLayout(LayoutGenerator):
    def generate(self, query: str, size: str, bg: Optional[str] = None) -> Dict[str, Tuple[int, int, int, int]]:
        # 기본 위치를 가지는 레이아웃 생성
        return {
            "title": (50, 50, 400, 100),
            "image": (100, 300, 500, 700),
            "description": (100, 800, 500, 950)
        }

class ContentAwareLayout(LayoutGenerator):
    def generate(self, query: str, size: str, bg: Optional[str] = None) -> Dict[str, Tuple[int, int, int, int]]:
        if not bg:
            raise ValueError("ContentAwareLayout requires a background image.")
        # 배경을 고려해 title과 description만 배치
        return {
            "title": (30, 80, 400, 150),
            "description": (30, 600, 700, 750)
        }

class RefineCompleteLayout(LayoutRefiner):
    def refine(self, layout: Dict[str, Tuple[int, int, int, int]], query: str, size: str, bg: Optional[str] = None) -> Dict[str, Tuple[int, int, int, int]]:
        # subtitle 추가, image 위치 약간 이동
        new_layout = layout.copy()
        if "title" in new_layout:
            x1, y1, x2, y2 = new_layout["title"]
            new_layout["subtitle"] = (x1, y2 + 10, x2, y2 + 50)
        if "image" in new_layout:
            x1, y1, x2, y2 = new_layout["image"]
            new_layout["image"] = (x1 + 10, y1 + 10, x2 + 10, y2 + 10)
        return new_layout

class GPTCopyWriter(CopyWriter):
    def write(self, query: str, layout: Dict[str, Any]) -> Dict[str, str]:
        return {"title": "소화가 편한 우유!", "description": "덴마크산 천연원유로 만든 건강한 선택"}

class ThemedBackground(BackgroundMaker):
    def create(self, theme: str) -> str:
        return f"background_for_{theme}.png"

class SimpleFontStyler(FontStyler):
    def style(self, text: Dict[str, str], bg: str) -> Dict[str, Any]:
        return {"font": "NanumGothic", "color": "#333"}

class AutoDecorator(Decorator):
    def decorate(self, layout: Dict[str, Any], photo: Optional[str] = None) -> Dict[str, Any]:
        return {"stickers": ["milk_icon.png"], "photo": photo}

class HTMLRenderer(Renderer):
    def render(self, layout: Dict[str, Tuple[int, int, int, int]], text: Dict[str, str], style: Dict[str, Any], deco: Dict[str, Any]) -> str:
        # HTML로 요소 배치 및 스타일 적용
        html = "<div style='position:relative;'>"
        for key, box in layout.items():
            if key in text:
                x1, y1, x2, y2 = box
                html += f"<div style='position:absolute; left:{x1}px; top:{y1}px; width:{x2 - x1}px; height:{y2 - y1}px; font:{style['font']}; color:{style['color']};'>{text[key]}</div>"
        if deco.get("photo"):
            html += f"<img src='{deco['photo']}' style='position:absolute; left:200px; top:400px;'>"
        html += "</div>"
        return html

# === LangGraph 파이프라인 구성 ===

from langgraph.graph import StateGraph

class Pipeline:
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.components = self._init_components()
        self.graph: StateGraph[State] = StateGraph()

        # 노드 정의
        self.graph.add_node("layout", self.run_layout)
        self.graph.add_node("copy", self.run_copy)
        self.graph.add_node("bg", self.run_bg)
        self.graph.add_node("font", self.run_font)
        self.graph.add_node("check", self.run_check)
        self.graph.add_node("refine", self.run_refine)
        self.graph.add_node("deco", self.run_deco)
        self.graph.add_node("render", self.run_render)

        # 흐름 정의
        self.graph.set_entry_point("layout")
        self.graph.add_edge("layout", "copy")
        self.graph.add_edge("copy", "bg")
        self.graph.add_edge("bg", "font")
        self.graph.add_edge("font", "check")
        self.graph.add_conditional_edges("check", self.is_decolable, {"yes": "deco", "no": "refine"})
        self.graph.add_edge("refine", "copy")
        self.graph.add_edge("deco", "render")
        self.graph.set_finish_point("render")

    def _init_components(self) -> Dict[str, Any]:
        base_layout: LayoutGenerator = BasicLayout() if self.config.get("layout") != "content-aware" else ContentAwareLayout()
        layout_refiner: LayoutRefiner = RefineCompleteLayout()

        return {
            "layout_generator": base_layout,
            "layout_refiner": layout_refiner,
            "copy": GPTCopyWriter(),
            "bg": ThemedBackground(),
            "font": SimpleFontStyler(),
            "deco": AutoDecorator(),
            "renderer": HTMLRenderer()
        }

    def run_layout(self, state: State) -> State:
        layout = self.components["layout_generator"].generate(state["query"], state["size"], state.get("bg"))
        return {**state, "layout": layout}

    def run_refine(self, state: State) -> State:
        layout = self.components["layout_refiner"].refine(state["layout"], state["query"], state["size"], state.get("bg"))
        return {**state, "layout": layout}

    def run_copy(self, state: State) -> State:
        text = self.components["copy"].write(state["query"], state["layout"])
        return {**state, "text": text}

    def run_bg(self, state: State) -> State:
        bg = self.components["bg"].create(state["query"])
        return {**state, "bg": bg}

    def run_font(self, state: State) -> State:
        style = self.components["font"].style(state["text"], state["bg"])
        return {**state, "style": style}

    def run_check(self, state: State) -> State:
        return state  # 분기 판별용 빈 처리

    def is_decolable(self, state: State) -> str:
        layout = state["layout"]
        if "image" not in layout:
            return "no"
        x1, y1, x2, y2 = layout["image"]
        width = x2 - x1
        if width < 100:
            return "no"
        return "yes"

    def run_deco(self, state: State) -> State:
        deco = self.components["deco"].decorate(state["layout"], state.get("photo"))
        return {**state, "deco": deco}

    def run_render(self, state: State) -> State:
        result = self.components["renderer"].render(state["layout"], state["text"], state["style"], state["deco"])
        return {**state, "result": result}

# === 실행 예시 ===

if __name__ == "__main__":
    pipeline = Pipeline(config={"layout": "basic"})
    graph = pipeline.graph.compile()
    result = graph.invoke({"query": "덴마크 소화가 잘되는 우유", "size": "1080x1920", "photo": "milk.png"})
    print(result["result"])
