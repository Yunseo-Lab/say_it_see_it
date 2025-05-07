def run_test_case(name: str, input_state: State):
    print(f"\n=== {name} ===")
    pipeline = Pipeline(config={"layout": "basic"})  # content-aware 도 테스트 가능
    graph = pipeline.graph.compile()
    result = graph.invoke(input_state)
    print(result["layout"])
    print("→ render 결과 요약:", result["result"][:120], "...")


if __name__ == "__main__":
    # 🎯 테스트 케이스 1: image 넓이가 너무 좁아 deco 불가 → refine 진입
    test_case_1 = {
        "query": "여름철 갈증 해소 탄산수",
        "size": "1080x1920",
        "photo": "sparkling.png",
        "layout": {  # image가 너무 작음
            "title": (50, 50, 400, 100),
            "image": (100, 300, 130, 320),
            "description": (100, 800, 500, 950)
        }
    }

    # 🎯 테스트 케이스 2: 넓은 image → deco 가능
    test_case_2 = {
        "query": "오트밀 쿠키 출시",
        "size": "1080x1080",
        "photo": "cookie.png"
    }

    # 🎯 테스트 케이스 3: bg가 있는 content-aware 레이아웃
    test_case_3 = {
        "query": "눈 건강에 좋은 블루베리",
        "size": "1080x1920",
        "photo": "blueberry.png",
        "bg": "bluebg.jpg"
    }

    # 실행
    run_test_case("작은 image → refine", test_case_1)
    run_test_case("정상적인 기본 layout", test_case_2)
    run_test_case("content-aware 배경 사용", test_case_3)