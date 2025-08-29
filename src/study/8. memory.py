from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, ConversationSummaryBufferMemory, ConversationSummaryMemory
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory.vectorstore_token_buffer_memory import ConversationVectorStoreTokenBufferMemory
from langchain_chroma import Chroma
# from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_openai import OpenAIEmbeddings

load_dotenv()
llm = ChatOpenAI()
# embedder = HuggingFaceInstructEmbeddings(
#     query_instruction="Represent the query for retrieval:"
# )
embedder = OpenAIEmbeddings(model="text-embedding-3-large")
chroma = Chroma(collection_name="demo", embedding_function=embedder)
retriever = chroma.as_retriever(search_type="similarity_score_threshold", search_kwargs={"k":5, "score_threshold":0.01})
conversation_memory = ConversationVectorStoreTokenBufferMemory(
    return_messages=True,
    llm=llm,
    retriever=retriever,
    max_token_limit=1000
)

# 1. ConversationBufferMemory
# memory = ConversationBufferMemory()

# 4-1. ConversationSummaryMemory
# memory = ConversationSummaryMemory(llm=llm, return_messages=True)
# 4-2. ConversationSummaryBufferMemory
# 최근 100개의 토큰에 해당하는 채팅정보를 유지하고, 그 이후의 값들은 요약하는 설정.
# memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=100, return_messages=True)

# memory.save_context(
#     inputs={
#         "human": "서울에서 제일 맛있는 돈까스 집을 추천해줘."
#     },
#     outputs={
#         "ai": "안녕하세요! 서울에는 맛있는 돈까스 집이 많아요! 분위기 좋은 맛집이 좋으세요, 가성비 있는 곳이 좋으세요?"
#     }
# )

# memory.save_context(
#     inputs={
#         "human": "가성비 맛집으로 추천해줘."
#     },
#     outputs={
#         "ai": "가성비 좋은 돈까스 맛집으로는 '명동 고등어덮밥'이 있어요! 이 식당은 붐비는 명동 중심가에 위치해 있고, 신선한 재료와 푸짐한 양으로 유명해요. 가격도 적당하니 한번 가보시는 건 어떨까요?"
#     }
# )

conversation_memory.save_context({"Human": "서울에서 제일 맛있는 돈까스 맛집을 추천해줘"},
                                  {"AI": "안녕하세요! 서울에는 맛있는 돈까스 집이 많아요! 분위기 좋은 곳을 원하시나요, 아니면 가성비 좋은 곳을 찾으시나요?"}
)
conversation_memory.save_context({"Human": "돈까스가 무슨 분위기야. 가성비 맛집으로 추천해줘"},
                                  {"AI": "그렇다면 김밥천국을 추천합니다!"}
)
load_data = conversation_memory.load_memory_variables({"input": "돈까스 맛집을 추천해줘."})
print(load_data)

# llm에 대화 이력을 유지
# conversation = ConversationChain(llm=llm, memory=conversation_memory)
# response = conversation.predict(input="명동 고등어덮밥 말고는 없어?")

# print(response)