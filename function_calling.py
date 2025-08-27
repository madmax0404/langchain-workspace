import json
from openai import OpenAI
from dotenv import load_dotenv
import os


def get_current_weather(location, unit="fahrenheit"):
    if "seoul" in location.lower():
        return json.dumps({"location": "Seoul", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]

# .env 파일 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

messages = [
    {"role": "user", "content": [{"type": "text", "text": "서울의 날씨는 어떤가요?"}]}
]

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=messages,
    tools=tools,
)

print(response.to_json(indent=2))

available_functions = {
    "get_current_weather": get_current_weather,
}
for tool_call in response.choices[0].message.tool_calls:
    # 함수를 실행
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)
    # function_args = tool_call.function.arguments
    # print(function_args)
    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit")
    )

    print(function_response)