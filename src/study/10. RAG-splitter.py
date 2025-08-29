from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import TextLoader

raw_documents = TextLoader(file_path="textSplitterExample.txt", encoding="utf-8").load()
# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=5,
#     is_separator_regex=False,
#     chunk_overlap=1,
#     length_function=len
# )
# documents = text_splitter.split_documents(raw_documents)

# print(len(documents))
# print(documents)
# print(documents[0])

# Recursive
# text_splitter = RecursiveCharacterTextSplitter(
#     separators=["\n\n", "\n", " "],
#     chunk_size=50,
#     chunk_overlap=20,
#     length_function=len,
#     is_separator_regex=False
# )
# documents = text_splitter.split_documents(raw_documents)

# print(len(documents))
# print(documents[0])
# print(documents[1])
# print(documents)

text_splitter = TokenTextSplitter.from_tiktoken_encoder(
    chunk_size=100,
    chunk_overlap=20,
    encoding_name="gpt2"
)
documents = text_splitter.split_documents(raw_documents)

print(len(documents))
print(documents[0])
print(documents[1])