from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-mini")
# response = llm.invoke("Hello, world!")
# print(response.content)

# 맥락 유지
messages = [
    SystemMessage("당신은 친절한 상담사입니다. 사용자의 질문에 부드러운 톤으로 응답하세요."),
    HumanMessage("안녕하세요 저는 한종윤입니다."),
    AIMessage("안녕하세요 종윤님. 무엇을 도와드릴까요?"),
    HumanMessage("제 이름을 기억하시나요?")
]

# ai_message = llm.invoke(messages)
# print(ai_message.content)

for chunk in llm.stream(messages):
    print(chunk.content, end="", flush=True)