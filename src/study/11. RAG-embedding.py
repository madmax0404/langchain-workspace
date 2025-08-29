from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain.storage import LocalFileStore
from langchain.embeddings import CacheBackedEmbeddings

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
# vector = embedding_model.embed_query("오늘의 날씨는?")
# print(type(vector))
# print(len(vector)) # 1536개의 배열
# print(vector[:10])

# vector = embedding_model.embed_documents(["안녕하세요?", "안녕하쇼?", "안녕", "하이", "hello"])
# print(type(vector))
# print(len(vector))
# for i in range(len(vector)):
#     print(len(vector[i]))

store = LocalFileStore("./cache/")

cached_embedder = CacheBackedEmbeddings.from_bytes_store(
    namespace=embedding_model.model,
    document_embedding_cache=store,
    underlying_embeddings=embedding_model,
    key_encoder="blake2b"
)

vector = cached_embedder.embed_documents(["안녕하세요?", "안녕하쇼?", "안녕", "하이", "hello"])
print(type(vector))
print(len(vector))
for i in range(len(vector)):
    print(len(vector[i]))