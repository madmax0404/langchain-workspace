# 1. 텍스트 파일 로드
from langchain_community.document_loaders import TextLoader, PyPDFLoader, WebBaseLoader

loader = TextLoader("example.txt")
documents = loader.load()

# print(documents)

# 2. PDF 문서 로드
loader = PyPDFLoader("쬬조_세미프로젝트 보고서.pdf")
documents = loader.load()

# print(documents[7].page_content)

# 3. 웹문서 로드
import bs4

loader = WebBaseLoader(
    web_paths=["https://khedu.co.kr/main/main.kh"],
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(["section", "div"])),
    header_template={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
    },
)
loader.requests_kwargs = {"verify": False}
documents = loader.load()

print(documents)