from dotenv import load_dotenv
load_dotenv()
# 1. 프롬프트 템플릿
from langchain_core.prompts import PromptTemplate
# 탬플릿내용
question = "{location}의 맛집을 10개이상 추천해주세요. \n ### 응답예시 ### \n 번호. 음식점 - 설명"
prompt_template = PromptTemplate.from_template(question)
# 2. llm설정
from langchain_openai import ChatOpenAI
llm = ChatOpenAI()
# 3. 출력파서 설정
from langchain_core.output_parsers import StrOutputParser
output_parser = StrOutputParser()
# 체인구성
# | => lcel -> 랭체인의 컴포넌트를 체인으로 구성하는 역할의 연산자
chain = prompt_template | llm | output_parser
input = {"location":"강남 역삼"}
#chain.invoke(input) # 체인을 호출하고 결과를 반환하는 함수
# batch
#  - 리스트를 입력으로 전달하여 체인을 호출하는 방식
#  - 리스트의 각 입력에 대한 결과를 마찬가지로 리스트형태로 반환
input1 = input
input2 = {"location","선릉 압구정"}
chain.batch([input1, input2])