
# All Documents loader in python


# 1. TextLoader

# The simplest loader — reads plain text files.


from langchain.document_loaders import TextLoader

# Load a single text file
loader = TextLoader("example.txt")
documents = loader.load()

print(documents[0].page_content)  # The text content
print(documents[0].metadata)      # {'source': 'example.txt'}


#----------------------------------------------------------------------------------------------------------------

# 2. CSVLoader

# Loads CSV files row by row or as a whole.


from langchain.document_loaders import CSVLoader

# Load each row as a separate document
loader = CSVLoader(file_path="data.csv")
documents = loader.load()

# Or specify a column to use as content
loader = CSVLoader(
    file_path="data.csv",
    source_column="id",           # Which column to track as source
    content_columns=["name", "description"]  # Which columns to include
)
documents = loader.load()

for doc in documents:
    print(doc.page_content)  # "name: John | description: Developer"
    print(doc.metadata)      # {'source': '1', 'row': 0}


#----------------------------------------------------------------------------------------------------------------

# 3. JSONLoader

# Loads JSON files with JMESPath queries for extraction.


from langchain.document_loaders import JSONLoader
import json

# Sample JSON: [{"title": "A", "content": "..."}, {"title": "B", ...}]
loader = JSONLoader(
    file_path="data.json",
    jq_schema=".[]",              # JMESPath: iterate over array
    content_key="content",        # Which key contains the text
    metadata_func=lambda record, metadata: {
        **metadata,
        "title": record.get("title")
    }
)
documents = loader.load()

# For nested JSON
loader = JSONLoader(
    file_path="nested.json",
    jq_schema=".articles[]",      # Extract from nested path
    content_key="body"
)


#----------------------------------------------------------------------------------------------------------------

# 4. PyPDFLoader / PDFPlumberLoader

# Extract text from PDF files.


from langchain.document_loaders import PyPDFLoader, PDFPlumberLoader

# Basic PDF loading (one doc per page)
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Lazy loading for large PDFs (memory efficient)
for doc in loader.lazy_load():
    print(f"Page {doc.metadata['page']}: {doc.page_content[:100]}")

# PDFPlumber — better for tables and complex layouts
loader = PDFPlumberLoader("document.pdf")
documents = loader.load()

print(documents[0].metadata)  # {'source': 'doc.pdf', 'page': 0}


#----------------------------------------------------------------------------------------------------------------

# 5. UnstructuredPDFLoader

# Advanced PDF parsing with Unstructured.io (handles images, tables).


from langchain.document_loaders import UnstructuredPDFLoader

# "single" = one document, "elements" = per element, "paged" = per page
loader = UnstructuredPDFLoader(
    "document.pdf",
    mode="elements",              # Extract individual elements
    strategy="hi_res"             # Use OCR for scanned PDFs
)
documents = loader.load()

for doc in documents:
    print(f"Type: {doc.metadata['category']}")  # Title, NarrativeText, Table, etc.
    print(doc.page_content)


#----------------------------------------------------------------------------------------------------------------

# 6. DirectoryLoader

# Load all files from a directory recursively.


from langchain.document_loaders import DirectoryLoader, TextLoader

# Load all .txt files
loader = DirectoryLoader(
    path="./documents",
    glob="**/*.txt",              # Pattern matching
    loader_cls=TextLoader,        # Which loader to use per file
    show_progress=True,
    use_multithreading=True       # Faster for many files
)
documents = loader.load()

print(f"Loaded {len(documents)} documents")

# Load multiple file types with different loaders
from langchain.document_loaders import PythonLoader

loader = DirectoryLoader(
    path="./src",
    glob="**/*.py",
    loader_cls=PythonLoader      # Preserves Python-specific formatting
)


#----------------------------------------------------------------------------------------------------------------

# 7. WebBaseLoader

# Scrape web pages.


from langchain.document_loaders import WebBaseLoader

# Single URL
loader = WebBaseLoader("https://python.langchain.com/docs/get_started/introduction")
documents = loader.load()

# Multiple URLs
urls = [
    "https://example.com/page1",
    "https://example.com/page2"
]
loader = WebBaseLoader(urls)
documents = loader.load()

# With custom headers (for authenticated pages)
loader = WebBaseLoader(
    "https://api.example.com/docs",
    header_template={
        "User-Agent": "Mozilla/5.0",
        "Authorization": "Bearer token123"
    }
)

# Extract specific parts using BeautifulSoup
from bs4 import BeautifulSoup

loader = WebBaseLoader("https://example.com")
loader.bs_get_text_kwargs = {"separator": " ", "strip": True}
documents = loader.load()


#----------------------------------------------------------------------------------------------------------------

# 8. RecursiveUrlLoader

# Crawl entire websites recursively.


from langchain.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup

def extractor(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n", strip=True)

loader = RecursiveUrlLoader(
    url="https://docs.python.org/3/",
    max_depth=2,                  # How many links deep to follow
    extractor=extractor,
    prevent_outside=True,           # Stay within the domain
    use_async=True,                # Faster with async
    timeout=10
)
documents = loader.load()

print(f"Crawled {len(documents)} pages")


#----------------------------------------------------------------------------------------------------------------

# 9. SitemapLoader

# Load from XML sitemaps (great for SEO-optimized sites).


from langchain.document_loaders import SitemapLoader

loader = SitemapLoader(
    web_path="https://example.com/sitemap.xml",
    filter_urls=["https://example.com/blog/"],  # Only load blog posts
    parsing_function=lambda content, url, meta: {
        "page_content": content,
        "metadata": {**meta, "source": url}
    }
)
documents = loader.load()


#----------------------------------------------------------------------------------------------------------------

# 10. GitLoader

# Load files from Git repositories.


from langchain.document_loaders import GitLoader

# Clone and load
loader = GitLoader(
    clone_url="https://github.com/langchain-ai/langchain.git",
    repo_path="./langchain_repo",
    branch="master",
    file_filter=lambda file_path: file_path.endswith(".md")  # Only markdown
)
documents = loader.load()

# Load existing local repo
loader = GitLoader(
    repo_path="./my_project",
    file_filter=lambda file_path: "test" not in file_path  # Exclude tests
)


#----------------------------------------------------------------------------------------------------------------

# 11. NotionDBLoader / NotionDirectoryLoader

# Load from Notion.


from langchain.document_loaders import NotionDBLoader

loader = NotionDBLoader(
    integration_token="secret_xxx",
    database_id="abc123",
    request_timeout_sec=30
)
documents = loader.load()

# Or load from exported Notion HTML
from langchain.document_loaders import NotionDirectoryLoader
loader = NotionDirectoryLoader("path/to/notion/export")


#----------------------------------------------------------------------------------------------------------------

# 12. ConfluenceLoader

# Load from Atlassian Confluence.


from langchain.document_loaders import ConfluenceLoader

loader = ConfluenceLoader(
    url="https://your-domain.atlassian.net/wiki",
    username="your-email@example.com",
    api_key="your-api-token",
    space_key="DEV",              # Specific Confluence space
    limit=50,                     # Pages per request
    max_pages=100                 # Total pages to load
)
documents = loader.load()


#----------------------------------------------------------------------------------------------------------------

# 13. ArxivLoader

# Load academic papers from arXiv.


from langchain.document_loaders import ArxivLoader

# Search by query
loader = ArxivLoader(query="attention is all you need", load_max_docs=2)
documents = loader.load()

# Or by paper ID
loader = ArxivLoader(query="1706.03762", load_max_docs=1)
documents = loader.load()

for doc in documents:
    print(doc.metadata)  # {'Published': '2017-06-12', 'Title': 'Attention Is All You Need', ...}
    print(doc.page_content[:200])


#----------------------------------------------------------------------------------------------------------------

# 14. WikipediaLoader

# Load Wikipedia articles.


from langchain.document_loaders import WikipediaLoader

loader = WikipediaLoader(
    query="Artificial intelligence",
    load_max_docs=2,              # Number of articles
    lang="en"                     # Language
)
documents = loader.load()

print(documents[0].metadata)  # {'summary': '...', 'title': 'Artificial intelligence'}


#----------------------------------------------------------------------------------------------------------------

# 15. YoutubeLoader

# Transcribe YouTube videos.


from langchain.document_loaders import YoutubeLoader

# Requires: pip install youtube-transcript-api
loader = YoutubeLoader.from_youtube_url(
    "https://youtube.com/watch?v=abc123",
    add_video_info=True           # Include title, author, length
)
documents = loader.load()

print(documents[0].metadata)  # {'source': 'abc123', 'title': '...', 'author': '...'}


#----------------------------------------------------------------------------------------------------------------

# 16. UnstructuredFileLoader

# Universal loader for many formats (docs, pptx, images, etc.).


from langchain.document_loaders import UnstructuredFileLoader

# Auto-detects file type
loader = UnstructuredFileLoader(
    "document.docx",
    mode="elements",              # or "single", "paged"
    strategy="fast"               # "fast", "hi_res", "ocr_only"
)
documents = loader.load()

# Supported: .docx, .pptx, .xlsx, .html, .eml, .msg, .jpg, .png, etc.


#----------------------------------------------------------------------------------------------------------------

# 17. AzureBlobStorageContainerLoader / S3FileLoader

# Cloud storage loaders.


from langchain.document_loaders import AzureBlobStorageContainerLoader

loader = AzureBlobStorageContainerLoader(
    conn_str="DefaultEndpointsProtocol=https;...",
    container="my-container",
    prefix="documents/"           # Only load from this prefix
)
documents = loader.load()

# AWS S3
from langchain.document_loaders import S3FileLoader

loader = S3FileLoader(
    bucket="my-bucket",
    key="path/to/file.pdf",
    aws_access_key_id="...",
    aws_secret_access_key="..."
)


#----------------------------------------------------------------------------------------------------------------

# 18. DataFrameLoader

# Load from pandas DataFrames.


from langchain.document_loaders import DataFrameLoader
import pandas as pd

df = pd.DataFrame({
    "product": ["Laptop", "Phone"],
    "description": ["High-performance", "5G enabled"],
    "price": [999, 699]
})

loader = DataFrameLoader(df, page_content_column="description")
documents = loader.load()

for doc in documents:
    print(doc.page_content)   # "High-performance"
    print(doc.metadata)       # {'product': 'Laptop', 'price': 999}


#----------------------------------------------------------------------------------------------------------------

# 19. PlaywrightURLLoader

# JavaScript-rendered pages (SPA, React apps).


from langchain.document_loaders import PlaywrightURLLoader

loader = PlaywrightURLLoader(
    urls=["https://spa-app.example.com"],
    remove_selectors=["header", "footer", "nav"],  # Remove clutter
    continue_on_failure=True
)
documents = loader.load()


#----------------------------------------------------------------------------------------------------------------

# 20. TwitterTweetLoader

# Load tweets (requires Twitter API v2).


from langchain.document_loaders import TwitterTweetLoader

loader = TwitterTweetLoader.from_bearer_token(
    oauth2_bearer_token="YOUR_BEARER_TOKEN",
    twitter_users=["elonmusk", "sama"],
    number_tweets=50
)
documents = loader.load()


#----------------------------------------------------------------------------------------------------------------

# 21. AirbyteLoader

# Load from 300+ sources via Airbyte.


from langchain.document_loaders import AirbyteStripeLoader

loader = AirbyteStripeLoader(
    config={
        "client_secret": "sk_test_...",
        "account_id": "acct_..."
    },
    stream_name="customers"
)
documents = loader.load()


#----------------------------------------------------------------------------------------------------------------

# 22. FirestoreLoader / BigQueryLoader

# Google Cloud loaders.


from langchain.document_loaders import FirestoreLoader

loader = FirestoreLoader(
    source="users",
    project="my-project"
)
documents = loader.load()

# BigQuery
from langchain.document_loaders import BigQueryLoader

loader = BigQueryLoader(
    query="SELECT title, content FROM blog_posts LIMIT 10",
    project="my-project",
    page_content_columns=["content"],
    metadata_columns=["title"]
)


#----------------------------------------------------------------------------------------------------------------

# 23. SlackDirectoryLoader / DiscordChatLoader

# Messaging platform loaders.


from langchain.document_loaders import SlackDirectoryLoader

loader = SlackDirectoryLoader("path/to/slack/export")
documents = loader.load()

from langchain.document_loaders import DiscordChatLoader
loader = DiscordChatLoader("path/to/discord/export.json")


#----------------------------------------------------------------------------------------------------------------

# 24. Email Loaders


from langchain.document_loaders import UnstructuredEmailLoader

# .eml files
loader = UnstructuredEmailLoader("email.eml", mode="elements")
documents = loader.load()

# Outlook .msg
from langchain.document_loaders import OutlookMessageLoader
loader = OutlookMessageLoader("email.msg")


#----------------------------------------------------------------------------------------------------------------

# 25. Custom Document Loader

# Build your own loader by extending `BaseLoader`.

```python
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from typing import Iterator, List

class APIJsonLoader(BaseLoader):
    def __init__(self, api_url: str, headers: dict = None):
        self.api_url = api_url
        self.headers = headers or {}
    
    def lazy_load(self) -> Iterator[Document]:
        import requests
        
        response = requests.get(self.api_url, headers=self.headers)
        data = response.json()
        
        for item in data:
            yield Document(
                page_content=item.get("content", ""),
                metadata={
                    "source": self.api_url,
                    "id": item.get("id"),
                    "created_at": item.get("created_at")
                }
            )
    
    def load(self) -> List[Document]:
        return list(self.lazy_load())

# Usage
loader = APIJsonLoader("https://api.example.com/posts")
documents = loader.load()


#----------------------------------------------------------------------------------------------------------------


# Best Practices

# 1. Always use `lazy_load()` for large datasets to avoid memory issues

# 2. Combine with text splitters — loaders give raw docs, splitters chunk them:
   

   from langchain.text_splitter import RecursiveCharacterTextSplitter
   splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
   chunks = splitter.split_documents(documents)
   


# 3. Preserve metadata — always include source info for traceability

# 4. Handle errors gracefully — use `continue_on_failure=True` when available

# 5. Pre-process content — clean HTML, remove boilerplate before embedding




#----------------------------------------------------------------------------------------------------------------