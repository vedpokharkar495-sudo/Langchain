

# ALL RETRIVER's IN LANGCHAIN '


#----------------------------------------------------------------------------------------------------------------


# 1. VectorStoreRetriever

# The most common retriever — uses semantic similarity via embeddings.


from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Setup: Create sample documents
documents = [
    Document(page_content="Python is a high-level programming language known for readability.", metadata={"source": "python-docs"}),
    Document(page_content="JavaScript runs in browsers and enables interactive web pages.", metadata={"source": "js-docs"}),
    Document(page_content="Rust provides memory safety without garbage collection.", metadata={"source": "rust-docs"}),
    Document(page_content="Python's asyncio library enables concurrent programming.", metadata={"source": "python-docs"}),
    Document(page_content="TypeScript adds static typing to JavaScript.", metadata={"source": "ts-docs"}),
]

# Split and embed
splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
chunks = splitter.split_documents(documents)

# Create vector store
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings()
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",      # "similarity", "mmr", "similarity_score_threshold"
    search_kwargs={"k": 2}         # Return top 2 results
)

# Retrieve
results = retriever.get_relevant_documents("How does Python handle concurrency?")
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")

#----------------------------------------------------------------------------------------------------------------

# Search Types:


# MMR (Maximal Marginal Relevance) — balances relevance with diversity
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # 0=diverse, 1=relevant only
)

# Similarity with score threshold
retriever = vectorstore.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.7, "k": 5}
)

# Fetch more candidates for MMR re-ranking
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 20}  # Fetch 20, return top 3 diverse
)

#----------------------------------------------------------------------------------------------------------------



# 2. MultiQueryRetriever

# Generates multiple query variations to improve retrieval coverage.


from langchain.retrievers import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI(temperature=0),
    prompt=None  # Uses default prompt to generate 3 variations
)

# For query "python async", it generates:
# 1. "How does Python handle asynchronous programming?"
# 2. "Python concurrency and asyncio library"
# 3. "Async/await patterns in Python"
# Then retrieves from all variations and deduplicates

results = multi_retriever.get_relevant_documents("Tell me about async Python")
for doc in results:
    print(f"Retrieved: {doc.page_content[:100]}...")

#----------------------------------------------------------------------------------------------------------------

# Custom Query Generation:


from langchain.prompts import PromptTemplate

custom_prompt = PromptTemplate(
    input_variables=["question"],
    template="""Generate 4 different versions of the question to retrieve relevant documents.
Original question: {question}

Your task: Create variations that capture different aspects, synonyms, or rephrasings.
Output each question on a new line numbered 1-4."""
)

multi_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=ChatOpenAI(temperature=0.7),
    prompt=custom_prompt
)


#----------------------------------------------------------------------------------------------------------------


# 3. ContextualCompressionRetriever

# Compresses retrieved documents to only relevant parts.


from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

base_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Uses LLM to extract only relevant sentences from each document
compressor = LLMChainExtractor.from_llm(ChatOpenAI(temperature=0))

compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever
)

results = compression_retriever.get_relevant_documents("Python memory management")
for doc in results:
    print(f"Compressed: {doc.page_content}")
    print(f"Original source: {doc.metadata.get('source')}\n")

#----------------------------------------------------------------------------------------------------------------


# With LLMChainFilter (removes irrelevant docs entirely):


from langchain.retrievers.document_compressors import LLMChainFilter

# Instead of extracting, this filters out whole documents
filter_compressor = LLMChainFilter.from_llm(ChatOpenAI(temperature=0))

filter_retriever = ContextualCompressionRetriever(
    base_compressor=filter_compressor,
    base_retriever=base_retriever
)

results = filter_retriever.get_relevant_documents("Rust ownership")
print(f"Filtered down to {len(results)} relevant documents")

#----------------------------------------------------------------------------------------------------------------


# With EmbeddingsFilter (faster, no LLM call):


from langchain.retrievers.document_compressors import EmbeddingsFilter

embeddings_filter = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.76
)

embed_compression_retriever = ContextualCompressionRetriever(
    base_compressor=embeddings_filter,
    base_retriever=base_retriever
)


#----------------------------------------------------------------------------------------------------------------


# 4. ParentDocumentRetriever

# Retrieves small chunks for search, returns full parent documents.


from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# Child splitter: small chunks for embedding/retrieval
child_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)

# Parent splitter: larger chunks that get returned
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)

# Store for parent documents
store = InMemoryStore()

retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,           # Where child chunks are embedded
    docstore=store,                   # Where parent docs are stored
    child_splitter=child_splitter,
    parent_splitter=parent_splitter
)

# Add documents — stores both parent and child
retriever.add_documents(documents)

# Retrieves child chunks, returns their parent documents
results = retriever.get_relevant_documents("Python features")
for doc in results:
    print(f"Parent doc length: {len(doc.page_content)} chars")
    print(f"Content: {doc.page_content[:200]}...\n")


#----------------------------------------------------------------------------------------------------------------


# 5. EnsembleRetriever

# Combines multiple retrievers with weighted scores (RRF - Reciprocal Rank Fusion).


from langchain.retrievers import EnsembleRetriever
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever

# Semantic retriever (dense)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Keyword retriever (sparse) — BM25
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# Combine: 60% weight to dense, 40% to sparse
ensemble_retriever = EnsembleRetriever(
    retrievers=[dense_retriever, bm25_retriever],
    weights=[0.6, 0.4],
    search_kwargs={"k": 4}
)

results = ensemble_retriever.get_relevant_documents("Python async programming")
for doc in results:
    print(f"Source: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:100]}...\n")


#----------------------------------------------------------------------------------------------------------------


# 6. BM25Retriever

# Classic keyword-based retrieval (no embeddings needed).


from langchain.retrievers import BM25Retriever

# Initialize with documents
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 3

# Or with texts
texts = [doc.page_content for doc in chunks]
bm25_retriever = BM25Retriever.from_texts(texts)
bm25_retriever.k = 2

results = bm25_retriever.get_relevant_documents("JavaScript typing")
for doc in results:
    print(f"BM25 Score-based: {doc.page_content[:100]}...")


#----------------------------------------------------------------------------------------------------------------


# 7. TfidfRetriever

# TF-IDF based retrieval — good baseline, no embeddings.


from langchain.retrievers import TFIDFRetriever

tfidf_retriever = TFIDFRetriever.from_documents(chunks)
tfidf_retriever.k = 3

results = tfidf_retriever.get_relevant_documents("Rust safety")
for doc in results:
    print(f"TF-IDF: {doc.page_content[:100]}...")


#----------------------------------------------------------------------------------------------------------------


# 8. SelfQueryRetriever

# Uses LLM to convert natural language to structured metadata filters.


from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.schema import Document

# Documents with rich metadata
docs_with_meta = [
    Document(page_content="Python tutorial for beginners", metadata={"topic": "programming", "level": "beginner", "year": 2023}),
    Document(page_content="Advanced Rust patterns", metadata={"topic": "programming", "level": "advanced", "year": 2024}),
    Document(page_content="JavaScript ES6 features", metadata={"topic": "web", "level": "intermediate", "year": 2023}),
    Document(page_content="React hooks deep dive", metadata={"topic": "web", "level": "advanced", "year": 2024}),
]

vectorstore2 = Chroma.from_documents(docs_with_meta, OpenAIEmbeddings())

# Define metadata fields for the LLM to use
metadata_field_info = [
    AttributeInfo(
        name="topic",
        description="The subject area of the document",
        type="string"
    ),
    AttributeInfo(
        name="level",
        description="Difficulty level: beginner, intermediate, or advanced",
        type="string"
    ),
    AttributeInfo(
        name="year",
        description="Publication year",
        type="integer"
    ),
]

document_content_description = "Programming tutorials and documentation"

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(temperature=0),
    vectorstore=vectorstore2,
    document_contents=document_content_description,
    metadata_field_info=metadata_field_info,
    verbose=True,
    search_kwargs={"k": 2}
)

# LLM converts this to: filter={"$and": [{"level": {"$eq": "advanced"}}, {"year": {"$gte": 2024}}]}
results = retriever.get_relevant_documents("Show me advanced programming content from 2024")

for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Metadata: {doc.metadata}\n")


#----------------------------------------------------------------------------------------------------------------


# 9. TimeWeightedVectorStoreRetriever

# Recency-biased retrieval — newer documents rank higher.


from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
import time

# Create documents with timestamps
docs_time = [
    Document(page_content="Python 3.8 released with walrus operator", metadata={"last_accessed": time.time() - 100000}),
    Document(page_content="Python 3.12 released with improved error messages", metadata={"last_accessed": time.time()}),
    Document(page_content="Python 3.10 added pattern matching", metadata={"last_accessed": time.time() - 50000}),
]

vectorstore3 = FAISS.from_documents(docs_time, OpenAIEmbeddings())

retriever = TimeWeightedVectorStoreRetriever(
    vectorstore=vectorstore3,
    decay_rate=0.01,           # Higher = faster decay (more recency bias)
    k=2
)

results = retriever.get_relevant_documents("Python latest features")
for doc in results:
    print(f"Content: {doc.page_content}")
    print(f"Last accessed: {doc.metadata.get('last_accessed')}\n")


#----------------------------------------------------------------------------------------------------------------


# 10. MultiVectorRetriever

# Stores multiple vectors per document (e.g., summaries, chunks, questions).


from langchain.retrievers import MultiVectorRetriever
from langchain.storage import InMemoryByteStore
from langchain.schema import Document
import uuid

# Create base vectorstore
vectorstore4 = Chroma(collection_name="multi", embedding_function=OpenAIEmbeddings())

# Store for parent documents
store = InMemoryStore()
id_key = "doc_id"

retriever = MultiVectorRetriever(
    vectorstore=vectorstore4,
    docstore=store,
    id_key=id_key,
    search_kwargs={"k": 2}
)

# For each document, create multiple representations
doc_ids = [str(uuid.uuid4()) for _ in documents]

# Original chunks (child vectors)
for i, doc in enumerate(documents):
    doc.metadata[id_key] = doc_ids[i]
    # Add original
    vectorstore4.add_documents([doc])
    # Add summary as separate vector pointing to same parent
    summary = Document(
        page_content=f"Summary: {doc.page_content[:50]}...",
        metadata={id_key: doc_ids[i], "type": "summary"}
    )
    vectorstore4.add_documents([summary])
    # Store parent in docstore
    store.mset([(doc_ids[i], doc)])

results = retriever.get_relevant_documents("programming languages")
for doc in results:
    print(f"Retrieved: {doc.page_content[:100]}...")


#----------------------------------------------------------------------------------------------------------------


# 11. MergerRetriever (LongContextReorder)

# Reorders retrieved documents to combat "lost in the middle" effect.


from langchain.retrievers import MergerRetriever
from langchain.document_transformers import LongContextReorder

# Create multiple retrievers
retriever1 = vectorstore.as_retriever(search_kwargs={"k": 3})
retriever2 = BM25Retriever.from_documents(chunks)
retriever2.k = 3

# Merge results
lotr = MergerRetriever(retrievers=[retriever1, retriever2])

# Reorder: most relevant at start AND end (LLMs perform better at edges)
reordering = LongContextReorder()
results = lotr.get_relevant_documents("Python features")

# Apply reordering
reordered_docs = reordering.transform_documents(results)

print("Original order:", [r.page_content[:30] for r in results])
print("Reordered:", [r.page_content[:30] for r in reordered_docs])


#----------------------------------------------------------------------------------------------------------------


# 12. WebResearchRetriever

# Automatically searches the web and retrieves relevant pages.


from langchain.retrievers import WebResearchRetriever
from langchain.utilities import GoogleSearchAPIWrapper

search = GoogleSearchAPIWrapper()

web_retriever = WebResearchRetriever.from_llm(
    vectorstore=vectorstore,
    llm=ChatOpenAI(temperature=0),
    search=search,
    num_search_results=3
)

results = web_retriever.get_relevant_documents("Latest Python 3.13 features")
for doc in results:
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content: {doc.page_content[:200]}...\n")


#----------------------------------------------------------------------------------------------------------------


# 13. ArxivRetriever

# Retrieves academic papers from arXiv.


from langchain.retrievers import ArxivRetriever

arxiv_retriever = ArxivRetriever(
    load_max_docs=2,
    get_full_documents=True  # Download full PDF content
)

results = arxiv_retriever.get_relevant_documents("attention is all you need transformer")
for doc in results:
    print(f"Title: {doc.metadata.get('Title')}")
    print(f"Authors: {doc.metadata.get('Authors')}")
    print(f"Published: {doc.metadata.get('Published')}")
    print(f"Content preview: {doc.page_content[:200]}...\n")


#----------------------------------------------------------------------------------------------------------------


# 14. PubMedRetriever

# Retrieves biomedical literature from PubMed.


from langchain.retrievers import PubMedRetriever

pubmed_retriever = PubMedRetriever(
    top_k_results=3
)

results = pubmed_retriever.get_relevant_documents("machine learning drug discovery")
for doc in results:
    print(f"Title: {doc.metadata.get('Title')}")
    print(f"PMID: {doc.metadata.get('uid')}")
    print(f"Abstract: {doc.page_content[:300]}...\n")


#----------------------------------------------------------------------------------------------------------------


# 15. WikipediaRetriever

# Retrieves Wikipedia articles.


from langchain.retrievers import WikipediaRetriever

wiki_retriever = WikipediaRetriever(
    lang="en",
    load_max_docs=2,
    load_all_available_meta=True
)

results = wiki_retriever.get_relevant_documents("Artificial Intelligence")
for doc in results:
    print(f"Title: {doc.metadata.get('title')}")
    print(f"Summary: {doc.page_content[:300]}...\n")


#----------------------------------------------------------------------------------------------------------------


# 16. KayAiRetriever / MetalRetriever

# Third-party AI-native retrievers.


from langchain.retrievers import KayAiRetriever

kay_retriever = KayAiRetriever.create(
    dataset_id="company",
    data_types=["10-K", "10-Q"],
    num_contexts=3
)

results = kay_retriever.get_relevant_documents("Apple revenue 2023")


#----------------------------------------------------------------------------------------------------------------


# 17. Custom Retriever

# Build your own retriever by extending `BaseRetriever`.


from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
import random

class RandomSampleRetriever(BaseRetriever):
    """Returns random documents — useful for testing/baselines."""
    
    documents: List[Document]
    k: int = 3
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Return k random documents."""
        return random.sample(self.documents, min(self.k, len(self.documents)))
    
    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Async version."""
        return self._get_relevant_documents(query, run_manager=run_manager)

# Usage
custom_retriever = RandomSampleRetriever(documents=chunks, k=2)
results = custom_retriever.get_relevant_documents("anything")
for doc in results:
    print(f"Random: {doc.page_content[:50]}...")


#----------------------------------------------------------------------------------------------------------------

# Advanced Custom: Hybrid Score Retriever


from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
import numpy as np

class HybridScoreRetriever(BaseRetriever):
    """Combines semantic + keyword scores with custom weighting."""
    
    vectorstore: any  # Chroma/FAISS instance
    bm25_retriever: any
    alpha: float = 0.5  # Weight for semantic (1-alpha for keyword)
    k: int = 3
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Get semantic results
        semantic_docs = self.vectorstore.similarity_search_with_score(query, k=self.k*2)
        
        # Get BM25 results
        keyword_docs = self.bm25_retriever.get_relevant_documents(query)
        
        # Score fusion (simplified RRF)
        scores = {}
        for doc, score in semantic_docs:
            doc_id = doc.metadata.get('source', doc.page_content[:50])
            scores[doc_id] = {"doc": doc, "score": self.alpha * (1 / (1 + score))}
        
        for i, doc in enumerate(keyword_docs):
            doc_id = doc.metadata.get('source', doc.page_content[:50])
            if doc_id in scores:
                scores[doc_id]["score"] += (1 - self.alpha) * (1 / (i + 1))
            else:
                scores[doc_id] = {"doc": doc, "score": (1 - self.alpha) * (1 / (i + 1))}
        
        # Sort by combined score
        sorted_results = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [r["doc"] for r in sorted_results[:self.k]]

# Usage
hybrid = HybridScoreRetriever(
    vectorstore=vectorstore,
    bm25_retriever=BM25Retriever.from_documents(chunks),
    alpha=0.6,
    k=3
)

results = hybrid.get_relevant_documents("Python async")
for doc in results:
    print(f"Hybrid: {doc.page_content[:100]}...")

#----------------------------------------------------------------------------------------------------------------


# Best Practices


# Production pattern: Multi-layer retrieval
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter

# 1. Start with ensemble for recall
ensemble = EnsembleRetriever(
    retrievers=[
        vectorstore.as_retriever(search_kwargs={"k": 10}),
        BM25Retriever.from_documents(chunks, k=10)
    ],
    weights=[0.7, 0.3]
)

# 2. Compress for precision
compressor = EmbeddingsFilter(
    embeddings=OpenAIEmbeddings(),
    similarity_threshold=0.8
)

final_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=ensemble
)

# 3. Use in RAG chain
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    retriever=final_retriever,
    return_source_documents=True
)



#----------------------------------------------------------------------------------------------------------------


