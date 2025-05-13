from abc import ABC, abstractmethod
from typing import Dict, Any, TypedDict, Optional, Tuple
from langgraph.graph import StateGraph
from openai import OpenAI

# === 데이터 타입 정의 ===

class State(TypedDict, total=False):
    query: str
    size: str
    photo: Optional[str]
    candidates: Dict[str, Any]  # 요소, 폰트, 배경 등 초기 후보군
    layout: Dict[str, Tuple[int, int, int, int]]
    text: Dict[str, str]
    bg: str
    style: Dict[str, Any]
    deco: Dict[str, Any]
    result: str

# === 메인 인터페이스 정의 ===

class ElementPlanner(ABC):
    @abstractmethod
    def plan(self, query: str, size: str) -> Dict[str, Any]: pass

class Integrator(ABC):
    @abstractmethod
    def integrate(self, candidates: Dict[str, Any], query: str) -> Tuple[Dict, Dict, str, Dict]: pass

class Renderer(ABC):
    @abstractmethod
    def render(self, layout: Dict, text: Dict, style: Dict, deco: Dict) -> str: pass

# === ElementPlanner 컴포넌트 인터페이스 정의 ===

class LayoutElementGenerator(ABC):
    @abstractmethod
    def generate(self, query: str, size: str) -> list[str]: pass

class FontCandidateGenerator(ABC):
    @abstractmethod
    def generate(self, query: str) -> list[str]: pass

class BackgroundCandidateGenerator(ABC):
    @abstractmethod
    def generate(self, query: str) -> list[str]: pass

class ColorPaletteGenerator(ABC):
    @abstractmethod
    def generate(self, query: str) -> list[str]: pass  # HEX 코드 리스트 반환

class DecorationCandidateGenerator(ABC):
    @abstractmethod
    def generate(self, query: str) -> list[str]: pass

# === 메인 컴포넌트 구현 ===

class ParallelElementPlanner(ElementPlanner):
    def __init__(self):
        self.layout_gen = SimpleLayoutElementGenerator()
        self.font_gen = FontCandidateByTone()
        self.bg_gen = BackgroundFromTheme()
        self.deco_gen = IconRecommender()
        self.color_gen = SimpleColorPalette()

    def plan(self, query: str, size: str) -> Dict[str, Any]:
        return {
            "layout_elements": self.layout_gen.generate(query, size),
            "font_candidates": self.font_gen.generate(query),
            "background_candidates": self.bg_gen.generate(query),
            "deco_candidates": self.deco_gen.generate(query),
            "color_candidates": self.color_gen.generate(query)
        }

class UnifiedIntegrator(Integrator):
    def __init__(self, llm_client: OpenAI):
        self.llm = llm_client

    def integrate(self, candidates: Dict[str, Any], query: str) -> Tuple[Dict, Dict, str, Dict]:
        prompt = f"""
        사용자 쿼리: "{query}"
        사용 가능한 요소: {candidates['layout_elements']}
        폰트 후보: {candidates['font_candidates']}
        배경 후보: {candidates['background_candidates']}
        컬러 후보: {candidates['color_candidates']}
        장식 후보: {candidates['deco_candidates']}

        위 정보를 바탕으로 아래 항목을 결정해주세요:
        1. 디자인 컨셉명 (ex. "soft nature", "urban bold")
        2. 각 요소별 문구 text["title"], text["description"], ...
        3. 각 요소별 스타일 style["title"], style["description"], ... (font, color 포함)
        4. 요소별 배치 layout["title"] = (x1, y1, x2, y2)
        5. 사용할 배경 (파일명)
        6. 사용할 장식 리스트 (deco["stickers"])

        응답은 다음 형식의 JSON으로 주세요:
        {{
            "concept": "...",
            "text": {{...}},
            "style": {{...}},
            "layout": {{...}},
            "bg": "...",
            "deco": {{ "stickers": [...], "photo": null }}
        }}
        """

        response = self.llm.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )

        result = eval(response.choices[0].message.content)

        return (
            result["layout"],
            result["text"],
            result["bg"],
            {
                "style": result["style"],
                "deco": result["deco"]
            }
        )

class HTMLRenderer(Renderer):
    def render(self, layout: Dict, text: Dict, style: Dict, deco: Dict) -> str:
        html = "<div style='position:relative;'>"
        for key, box in layout.items():
            if key in text:
                x1, y1, x2, y2 = box
                html += f"<div style='position:absolute; left:{x1}px; top:{y1}px; width:{x2 - x1}px; height:{y2 - y1}px; font:{style['font']}; color:{style['color']};'>{text[key]}</div>"
        if deco.get("photo"):
            html += f"<img src='{deco['photo']}' style='position:absolute; left:200px; top:400px;'>"
        html += "</div>"
        return html

# === ElementPlanner 컴포넌트 구현 ===

class SimpleLayoutElementGenerator(LayoutElementGenerator):
    def generate(self, query: str, size: str) -> list[str]:
        # 크기나 키워드 기반으로 요소 결정
        elements = ["title", "description"]
        if "이미지" in query or "제품" in query:
            elements.append("image")
        return elements

class FontCandidateByTone(FontCandidateGenerator):
    def generate(self, query: str) -> list[str]:
        if any(kw in query for kw in ["건강", "따뜻", "편안", "자연"]):
            return ["NanumMyeongjo", "NotoSerifKR", "NanumBarunGothic"]
        elif any(kw in query for kw in ["할인", "이벤트", "역동", "강조"]):
            return ["BlackHanSans", "BMDOHYEON", "SpoqaHanSansNeo"]
        else:
            return ["NanumGothic", "NotoSansKR", "Pretendard"]
        
class BackgroundFromTheme(BackgroundCandidateGenerator):
    def generate(self, query: str) -> list[str]:
        if "여름" in query:
            return ["bg_summer_beach.jpg", "bg_blue_gradient.png"]
        elif "겨울" in query:
            return ["bg_snow.png", "bg_winter_warm.jpg"]
        else:
            return [f"bg_generic_{query[:10]}.jpg"]
        
class SimpleColorPalette(ColorPaletteGenerator):
    def generate(self, query: str) -> list[str]:
        if "자연" in query or "건강" in query:
            return ["#A8D5BA", "#F4EFEA", "#5D737E"]
        elif "세일" in query or "이벤트" in query:
            return ["#FF5F5F", "#FFE066", "#333333"]
        else:
            return ["#FFFFFF", "#222222", "#AAAAAA"]
        
class IconRecommender(DecorationCandidateGenerator):
    def generate(self, query: str) -> list[str]:
        if "우유" in query:
            return ["milk_icon.png", "cow_emoji.png"]
        elif "과일" in query:
            return ["apple_icon.png", "fruit_slice.png"]
        else:
            return ["star.png", "sparkle.png"]
        


# === LangGraph 파이프라인 구성 ===

class UnifiedPipeline:
    def __init__(self):
        self.components = {
            "planner": ParallelElementPlanner(),
            "integrator": UnifiedIntegrator(),
            "renderer": HTMLRenderer()
        }
        self.graph: StateGraph[State] = StateGraph()
        self.graph.add_node("plan", self.run_plan)
        self.graph.add_node("integrate", self.run_integrate)
        self.graph.add_node("render", self.run_render)
        self.graph.set_entry_point("plan")
        self.graph.add_edge("plan", "integrate")
        self.graph.add_edge("integrate", "render")
        self.graph.set_finish_point("render")

    def run_plan(self, state: State) -> State:
        candidates = self.components["planner"].plan(state["query"], state["size"])
        return {**state, "candidates": candidates}

    def run_integrate(self, state: State) -> State:
        layout, text, bg, style_and_deco = self.components["integrator"].integrate(state["candidates"], state["query"])
        return {
            **state,
            "layout": layout,
            "text": text,
            "bg": bg,
            "style": {"font": style_and_deco["font"], "color": style_and_deco["color"]},
            "deco": {"stickers": style_and_deco["stickers"], "photo": state.get("photo")}
        }

    def run_render(self, state: State) -> State:
        result = self.components["renderer"].render(state["layout"], state["text"], state["style"], state["deco"])
        return {**state, "result": result}

# === 실행 예시 ===

if __name__ == "__main__":
    pipeline = UnifiedPipeline()
    graph = pipeline.graph.compile()
    result = graph.invoke({"query": "덴마크 소화가 잘되는 우유", "size": "1080x1920", "photo": "milk.png"})
    print(result["result"])
