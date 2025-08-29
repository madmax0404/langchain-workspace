from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def generate_python_code(prompt):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "당신은 Python 프로그래머입니다. 주어진 문제를 해결하는 간단한 Python 코드를 생성하세요.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            },
        ],
        response_format={"type": "text"},
        temperature=1,
        max_completion_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    return response.choices[0].message.content

# python 코드 추출 함수
import re

def extract_code(text):
    """응답에서 Python 코드 부분만 추출"""
    code_block = re.search(r"```python(.*?)```", text, re.DOTALL)
    return code_block.group(1).strip() if code_block else None

def execute_python_code(code):
    """생성된 Python 코드를 실행하고 결과를 반환"""
    local_vars = {}
    try:
        exec(code, {}, local_vars)
        return local_vars.get("result")  # 'result' 변수를 코드에서 반환하도록 설정
    except Exception as e:
        return f"코드 실행 중 오류 발생: {str(e)}"

if __name__ == "__main__":
    # 사용자 질문
    user_question = "윤년을 구하는 함수를 만들어줘."
    answer = generate_python_code(user_question)
    print("answer: ", answer)
    extracted_code = extract_code(answer)
    print("extracted_code: ", extracted_code)

    if extracted_code:
        result = execute_python_code(extracted_code)
        print("실행 결과: ", result)
    else:
        print("코드 추출 실패.")