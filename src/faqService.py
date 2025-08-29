from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableLambda
import uuid
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv()

# FastAPI 앱생성
app = FastAPI()

app.add_middleware(
    CORSMiddleware,  # 웹브라우저와 파이썬 서버간에 통신을 허용하기 위한 미들웨어
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1단계 - 문서 로드
loader = PyMuPDFLoader("../FAQ_SOURCE_DATA/교보핫트랙스고객응대메뉴얼.pdf")
docs = loader.load()

# 2단계 - 텍스트 스플리터
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
split_docs = text_splitter.split_documents(docs)

# 3단계 - 문서 임베딩 도구 및 벡터스토어 생성
embeddings = OpenAIEmbeddings()
vector_store = Chroma.from_documents(
    documents=split_docs, embedding=embeddings, persist_directory="./chroma_db"
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 4단계 - 메모리 저장소 추가
memory = ConversationBufferMemory(return_messages=True)

# 5단계 - 프롬프트 생성
prompt = ChatPromptTemplate.from_messages(
    messages=[
        (
            "system",
            """You are an assistant for question-answering tasks.
Use the following retrieved context to answer the question.
If you don't know the answer, just say you don't know.
Answer in Korean.""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "# Question:\n{query}\n# retrieved_docs:\n{retrieved_docs}\n# Answer:",
        )
    ]
)

# 6. 언어모델 생성
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

# 7. 세션별 대화 기록 관리
store = {}


# 세션 id를 기반으로 대화기록을 가져오거나 생성하는 함수.
def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    return store[session_id]


# 8. 입력 변환을 위한 RunnableLambda 설정
query_extractor = RunnableLambda(lambda x: x["query"])

chain = prompt | llm | StrOutputParser()

# 9. RunnableWithMessageHistory를 생성하여 메세지 히스토리 바인딩
chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="query",
    history_messages_key="chat_history",
)


# 사용자의 요청을 받을 데이터 모델
class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: Request, response: Response, chat_request: ChatRequest):
    # 클라이언트의 session_id 추출
    session_id = request.cookies.get("session_id")
    if session_id is None:
        session_id = str(uuid.uuid4())
        response.set_cookie(key="session_id", value=session_id, httponly=True)

    try:
        retrieved_docs = retriever.invoke(chat_request.question)

        async def response_generator() -> AsyncGenerator[str, None]:
            for chunk in chain_with_history.stream(
                input={
                    "query": chat_request.question,
                    "retrieved_docs": retrieved_docs,
                },
                config={"configurable": {"session_id": session_id}},
            ):
                yield chunk
                await asyncio.sleep(0)

        return StreamingResponse(response_generator(), media_type="text/plain")
    except Exception as e:
        print("에러 발생.", str(e))
        raise HTTPException(status_code=500, detail=str(e))


# 서버 실행 명령어
# uvicorn 파일명:app --host 0.0.0.0 --port 8000 --reload
# uvicorn faqService:app --host 0.0.0.0 --port 8000 --reload
