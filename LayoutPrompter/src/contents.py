from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import Dict, Any
from pydantic import create_model, Field

load_dotenv()

client = OpenAI()

def generate_contents(user_prompt: str, layout: list) -> dict:
    """
    사용자의 프롬프트와 레이아웃 정보를 받아 홍보물의 내용을 생성합니다.
    Structured Outputs 기능을 사용하여 정확한 형식의 응답을 보장합니다.
    
    Args:
        user_prompt (str): 사용자의 프롬프트
        layout (list): 예시) ["title", "description", "logo"]
    Returns:
        dict: 생성된 홍보물 내용(딕셔너리)
    """
    
    # 동적으로 Pydantic 모델 생성
    fields: Dict[str, Any] = {}
    for item in layout:
        fields[item] = (str, Field(..., description=f"The {item} content for the poster"))
    
    PosterModel = create_model('PosterModel', **fields)
    
    system_text = f"""Generate promotional content according to the following layout: {', '.join(layout)}.
Write the promotional message based on the user's prompt.
Create appropriate content for each item in korean."""
    
    try:
        # Structured Outputs: Pydantic 모델 전달하여 파싱 보장
        response = client.responses.parse(
            model="gpt-4o-mini",
            input=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_prompt},
            ],
            text_format=PosterModel,
        )
        return dict(response.output_parsed)
    except Exception as e:
        print(f"Error in generating poster content: {e}")
        return {item: f"Default {item}" for item in layout}

# Example usage
if __name__ == "__main__":
    user_prompt = "가나 초콜렛에 대한 홍보물 제작"
    layout = ["title", "description", "logo"]
    content = generate_contents(user_prompt, layout)
    print(type(content))  # <class 'dict'>
    print(content)