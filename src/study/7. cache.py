from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

llm = ChatOpenAI()

template = "{location}의 맛집을 10개 이상 추천해주세요. \n ### 응답예시 ### \n 번호. 음식 - 설명"
prompt = PromptTemplate.from_template(template)

chain = prompt | llm

# 인메모리 캐시 설정
from langchain.globals import set_llm_cache
# from langchain_community.cache import InMemoryCache
# set_llm_cache(InMemoryCache())
from langchain_community.cache import SQLiteCache
import os
if not os.path.exists("cache"):
    os.makedirs("cache")
    
set_llm_cache(SQLiteCache(database_path="cache/llm_cache.db"))

response = chain.invoke({"location": "강남"})
response = chain.invoke({"location": "강남"}) # 캐시에 저장된 응답 재사용. 단, 캐시 메모리는 현재 메모리에 생성되었기 때문에 메서드 종료후 함께 소멸한다.