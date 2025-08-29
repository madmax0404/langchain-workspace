from openai import OpenAI
from dotenv import load_dotenv
import os

# .env 파일 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# response = client.chat.completions.create(
#     model="gpt-4.1-mini",
#     messages=[
#         {
#             "role": "developer",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "당신은 챗봇입니다. 사용자의 질문에 대해 친절하게 응답해주세요.",
#                 }
#             ],
#         },
#         {
#             "role": "user",
#             "content": [{"type": "text", "text": "오늘의 날씨를 알려주세요."}],
#         },
#         {
#             "role": "assistant",
#             "content": [{"type": "text", "text": '''도와드릴게요! 어느 지역의 “오늘” 날씨를 보고 싶으신가요?

# 아래 중 몇 가지만 알려주시면 바로 안내해 드릴게요:
# - 지역: 도시/구·동 또는 우편번호 (예: 서울시 마포구, 부산 해운대, 06000)
# - 시간대: 지금, 오전/오후, 하루 종일 중 어떤 요약이 필요하신가요?
# - 단위: 섭씨(°C) 또는 화씨(°F)
# - 추가 정보 필요 여부: 강수확률/우산 여부, 바람, 미세먼지/초미세먼지, 자외선, 체감온도, 옷차림 추천 등

# 예시로 이렇게 적어주셔도 좋아요:
# - “서울 종로구 오늘 하루 종일, 섭씨로, 우산 필요한지와 미세먼지만 알려줘”
# - “부산 해운대 지금 기준, 간단 요약으로”

# 원하시면 활동 목적(출퇴근/야외운동/등산/아이와 외출 등)을 알려주시면 더 맞춤형으로 옷차림·휴대품 팁까지 드릴게요.'''}],
#         },
#         {
#             "role": "user",
#             "content": [{
#                 "type": "text",
#                 "text": "서울 강남의 날씨는 어때? json 형식으로 출력해줘. 예시) {location: '서울', temperature: '25C'}"
#             }],
#         },
#     ],
#     response_format={
#         "type": "json_object"
#     },
#     temperature=1,
#     max_completion_tokens=2048,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stream=True,
# )

# # print(response.choices[0].message.content)
# # 스트리밍 방식 응답받기.

# for chunk in response:
#     content = chunk.choices[0].delta.content
#     if (content is not None):
#         print(content, end="", flush=True)

img_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dc/The_White_House_-_54409525537_%28cropped%29_%28cropped%29.jpg/250px-The_White_House_-_54409525537_%28cropped%29_%28cropped%29.jpg"

response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text","text": "이미지에 대해 설명해 주세요."}, {"type": "image_url", "image_url": {"url": img_url}}
            ],
        }
    ],
    response_format={"type": "text"},
    temperature=1,
    max_completion_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    # stream=True,
)

print(response.choices[0].message.content)

