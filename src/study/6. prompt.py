from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder, FewShotPromptTemplate
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_chroma import Chroma

load_dotenv()
llm = ChatOpenAI()

# template = "{location}의 맛집을 10개 이상 추천해주세요. \n ### 응답예시 ### \n 번호. 음식 - 설명"
# prompt = PromptTemplate.from_template(template)
# prompt = prompt.format(location="강남")
# print(prompt)

# 생성자 방식의 템플릿 생성
# prompt = PromptTemplate(
#     template=template,
#     input_variables=["location"] # 유효성 검사
# )
# chain = prompt | llm
# chain.invoke({"location": "강남"})

# template = "{location1}과 {location2}의 맛집을 10개 이상 추천해주세요. \n ### 응답예시 ### \n 번호. 음식 - 설명"
# prompt = PromptTemplate.from_template(
#     template=template,
#     partial_variables={
#         "location1": "강남"
#     }
# )
# chain = prompt | llm
# chain.invoke({"location2": "역삼동"})

# template1 = "{location}의 맛집을 10개 이상 추천해주세요. \n ### 응답예시 ### \n 번호. 음식 - 설명"
# prompt1 = PromptTemplate.from_template(template1)

# template2 = "\n그 후 맛집이 어떤 유형의 음식을 파는지도 알려주세요. ex) 양식, 중식"
# prompt2 = PromptTemplate.from_template(template2)

# template3 = "\n그 후 맛집에서 파는 베스트 메뉴를 추천해주세요."
# combined_prompt = (prompt1 + prompt2 + template3)

# chain = combined_prompt | llm
# chain.invoke({"location": "압구정"})

# ChatPromptTemplate
# chat_template = ChatPromptTemplate.from_messages(
#     [
#         ("system", "당신은 헬스 트레이너입니다. 신규회원에 대해 친절히 상담해주세요."),
#         ("human", "안녕하세요."),
#         ("ai", "안녕하세요 회원님! 운동하러 오셨나요?"),
#         MessagesPlaceholder(variable_name="conversation"),
#         ("human", "{user_input}")
#     ]
# )

# chain = chat_template | llm
# chain.invoke({"user_input": "네 ^^",
#               "conversation": [
#                   ("human", "PT 1회 체험 가능한가요?"),
#                   ("ai", "네, 당연히 가능합니다! PT 1회 체험을 통해 저와 함께 운동을 경험해 보시고, 운동 스타일과 목표에 맞는 플랜을 세울 수 있습니다. 언제 편하신가요?"),
#                   ("human", "지금 당장요."),
#                   ("ai", "그럼 지금 바로 해보실까요?")
#               ]})

# few-shot prompt template
examples = [
    {
        "question": "세상에서 가장 가난한 왕은?",
        "answer": "최저임금!."
    },
    {
        "question": "오리가 얼면?",
        "answer": "언덕!"
    },
    {
        "question": "김밥이 죽으면 뭐가 될까?",
        "answer": "김밥천국!"
    },
    {
        "question": "고구려가 백제한테 이긴 이유는?",
        "answer": "고구려가 백제보다 ‘고’(高)퀄리티니까!"
    },
    {
        "question": "컴퓨터가 싫어하는 바람은?",
        "answer": "윈도우 업데이트!"
    },
    {
        "question": "소금이 죽으면 뭐가 될까?",
        "answer": "죽염!"
    },
    {
        "question": "개발자가 좋아하는 소스는?",
        "answer": "오픈소스!"
    }
]

example_prompt = PromptTemplate.from_template("질문: {question}\n{answer}")
prompt = FewShotPromptTemplate(
    examples=examples, # 사용할 예제들
    example_prompt=example_prompt, # 예제 포맷팅에 사용할 템플릿
    suffix="질문: {input}", # 예제 뒤에 추가될 접미사
    input_variables=["input"]
)
chain = prompt | llm
# chain.invoke({"input": "소금이 죽으면?"})

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples, # 사용할 예제
    OpenAIEmbeddings(), # 임베딩 모델
    Chroma, # 임베딩(벡터) 저장소
    k=2, # 생성할 예제 수
)
question = "섹시한 소금은?" # 요염
selected_examples = example_selector.select_examples({"question": question})
print(f"입력과 가장 유사한 예제: {question}")
for example in selected_examples:
    print(f"question: {example["question"]}")
    print(f"answer: {example["answer"]}")
    
prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    suffix="질문: {question}\n answer:"
)
chain = prompt | llm
chain.invoke({"question": "가장 섹시한 소금은?"})