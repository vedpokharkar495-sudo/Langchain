

# All Text Splitter in LANGCHAIN 
#----------------------------------------------------------------------------------------------------------------

# 1. RecursiveCharacterTextSplitter

# The most versatile and recommended splitter — tries multiple separators recursively.


from langchain.text_splitter import RecursiveCharacterTextSplitter

# Basic usage
text = """# Introduction to Python

Python is a high-level programming language.

## Variables
Variables store data values. In Python, you don't need to declare types.

## Functions
Functions are reusable blocks of code.

def greet(name):
    return f"Hello, {name}!"

This function takes a name parameter and returns a greeting.

## Classes
Classes are blueprints for objects.

class Person:
    def __init__(self, name):
        self.name = name
    
    def say_hello(self):
        return f"Hello, I'm {self.name}"

## Conclusion
Python is versatile and easy to learn."""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,        # Target size in characters
    chunk_overlap=50,      # Overlap between chunks (preserves context)
    length_function=len,   # How to measure length (len, tiktoken, etc.)
    separators=["\n\n", "\n", " ", ""]  # Priority order of separators
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ({len(chunk)} chars) ---")
    print(chunk[:100] + "...\n")

# Output: Respects markdown structure, splits at headers first, then paragraphs, then sentences


#----------------------------------------------------------------------------------------------------------------

# Advanced: With Token Counting


import tiktoken

splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base",  # OpenAI's encoding
    chunk_size=100,               # Tokens, not characters
    chunk_overlap=20
)

# Or for specific model
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=1000,
    chunk_overlap=200
)


#----------------------------------------------------------------------------------------------------------------

# 2. CharacterTextSplitter

# Simplest splitter — splits on a single character (default is `\n\n`).


from langchain.text_splitter import CharacterTextSplitter

text = "Paragraph one.\n\nParagraph two.\n\nParagraph three with more content here.\n\nParagraph four."

splitter = CharacterTextSplitter(
    separator="\n\n",      # Split on double newlines (paragraphs)
    chunk_size=50,
    chunk_overlap=10,
    length_function=len
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")


#----------------------------------------------------------------------------------------------------------------

# Single Character Split


# Split on every period (sentences)
splitter = CharacterTextSplitter(
    separator=".",
    chunk_size=100,
    chunk_overlap=0
)


#----------------------------------------------------------------------------------------------------------------

# 3. TokenTextSplitter

# Splits by token count using tiktoken (OpenAI's tokenizer).


from langchain.text_splitter import TokenTextSplitter

text = """
Natural language processing (NLP) is a subfield of linguistics, computer science, 
and artificial intelligence concerned with the interactions between computers and 
human language, in particular how to program computers to process and analyze 
large amounts of natural language data.
"""

splitter = TokenTextSplitter(
    encoding_name="cl100k_base",  # GPT-4 / GPT-3.5 encoding
    chunk_size=20,                # Tokens per chunk
    chunk_overlap=5               # Overlapping tokens
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk}")
    print(f"Token count: {len(chunk.split())}\n")


#----------------------------------------------------------------------------------------------------------------

# 4. MarkdownHeaderTextSplitter

# Preserves markdown structure by splitting on headers and keeping header metadata.


from langchain.text_splitter import MarkdownHeaderTextSplitter

markdown_text = """# Project Documentation

## Installation

Run `pip install mypackage` to install.

### Requirements
- Python 3.8+
- numpy
- pandas

## Usage

### Basic Example
```
import mypackage
result = mypackage.run()
```

Advanced Configuration
You can configure the package using a YAML file.

API Reference

Class: MyClass
This is the main class.

Method: run()
Executes the main logic.

Method: stop()
Stops execution.

Contributing
Please read CONTRIBUTING.md."""

headers_to_split_on = [
("#", "Header 1"),
("##", "Header 2"),
("###", "Header 3"),
("####", "Header 4")
]

splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

chunks = splitter.split_text(markdown_text)

for i, chunk in enumerate(chunks):
print(f"--- Chunk {i+1} ---")
print(f"Content: {chunk.page_content[:100]}...")
print(f"Metadata: {chunk.metadata}\n")

Output metadata includes:

{'Header 1': 'Project Documentation', 'Header 2': 'Installation', 'Header 3': 'Requirements'}


#----------------------------------------------------------------------------------------------------------------

## 5. **PythonCodeTextSplitter**

# Optimized for Python code — splits on class/function definitions.


from langchain.text_splitter import PythonCodeTextSplitter

python_code = '''
"""Module docstring."""
import os
import sys

GLOBAL_VAR = 42

def helper_function(x):
    """Helper docstring."""
    return x * 2

class DataProcessor:
    """Process data efficiently."""
    
    def __init__(self, source):
        self.source = source
        self.data = []
    
    def load(self):
        """Load data from source."""
        with open(self.source, 'r') as f:
            self.data = f.readlines()
        return self
    
    def process(self):
        """Process loaded data."""
        return [line.strip() for line in self.data]

class Analyzer:
    """Analyze processed data."""
    
    def analyze(self, data):
        return {"count": len(data), "unique": len(set(data))}

def main():
    processor = DataProcessor("data.txt")
    processor.load().process()
    analyzer = Analyzer()
    result = analyzer.analyze(processor.data)
    print(result)

if __name__ == "__main__":
    main()
'''

splitter = PythonCodeTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_text(python_code)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(chunk[:200])
    print()

# Splits at class/function boundaries, keeping structure intact

#----------------------------------------------------------------------------------------------------------------

# 6. RecursiveJsonSplitter

# Splits JSON while preserving structure — great for API responses.


from langchain.text_splitter import RecursiveJsonSplitter
import json

json_data = {
    "company": "TechCorp",
    "employees": [
        {"name": "Alice", "department": "Engineering", "skills": ["Python", "Go", "Rust"]},
        {"name": "Bob", "department": "Design", "skills": ["Figma", "Sketch", "Adobe XD"]},
        {"name": "Charlie", "department": "Engineering", "skills": ["Java", "Kotlin", "Spring"]}
    ],
    "products": {
        "web": {"name": "WebApp", "tech": ["React", "Node.js", "PostgreSQL"]},
        "mobile": {"name": "MobileApp", "tech": ["Flutter", "Firebase"]}
    },
    "metadata": {
        "founded": 2015,
        "locations": ["SF", "NY", "London", "Tokyo", "Berlin"],
        "revenue": {"2021": 1000000, "2022": 2500000, "2023": 5000000}
    }
}

splitter = RecursiveJsonSplitter(
    max_chunk_size=300,  # Max characters per chunk
    min_chunk_size=50
)

# Method 1: Split the JSON object directly
chunks = splitter.split_json(json_data)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(json.dumps(chunk, indent=2))
    print()

# Method 2: Create documents with metadata
texts = splitter.split_text(json_data=json_data)
for text in texts:
    print(text[:200])


#----------------------------------------------------------------------------------------------------------------

# 7. NLTKTextSplitter

# Uses NLTK's sentence tokenizer for natural language.


from langchain.text_splitter import NLTKTextSplitter

text = """
Dr. Smith went to Washington. He arrived at 3 p.m. 
The meeting was held at the U.S. Capitol building. 
Mr. Johnson presented the report. It was well received.
Despite the rain, everyone attended. The conference lasted 3 days.
"""

splitter = NLTKTextSplitter(
    separator="\n",
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.strip()}")

# NLTK correctly handles "Dr.", "U.S.", "Mr." — doesn't split on those periods


#----------------------------------------------------------------------------------------------------------------

# 8. SpacyTextSplitter

# Uses spaCy's sentence segmentation (more accurate than NLTK).


from langchain.text_splitter import SpacyTextSplitter

text = """
Apple Inc. is planning to open a new office in the U.S. 
The company was founded by Steve Jobs in 1976. 
It is headquartered in Cupertino, California. 
The new office will employ 500 people. 
Dr. Tim Cook announced this yesterday.
"""

splitter = SpacyTextSplitter(
    separator="",
    chunk_size=150,
    chunk_overlap=30,
    pipeline="en_core_web_sm"  # spaCy model to use
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.strip()}")

# spaCy handles "Apple Inc.", "U.S.", "Dr." correctly


#----------------------------------------------------------------------------------------------------------------

# 9. Language

# Pre-configured splitters for specific programming languages.


from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language
)

# Python
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=300,
    chunk_overlap=50
)

# JavaScript
js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS,
    chunk_size=300,
    chunk_overlap=50
)

# Available languages:
# PYTHON, JS, TS, JAVA, KOTLIN, C, CPP, GO, RUST, RUBY, PHP, SWIFT, MARKDOWN, HTML, SOL, CSHARP

js_code = '''
function calculateTotal(items) {
    let total = 0;
    for (const item of items) {
        total += item.price * item.quantity;
    }
    return total;
}

class ShoppingCart {
    constructor() {
        this.items = [];
    }
    
    addItem(item) {
        this.items.push(item);
    }
    
    removeItem(id) {
        this.items = this.items.filter(item => item.id !== id);
    }
    
    getTotal() {
        return calculateTotal(this.items);
    }
}

const cart = new ShoppingCart();
cart.addItem({id: 1, price: 10, quantity: 2});
console.log(cart.getTotal());
'''

chunks = js_splitter.split_text(js_code)
for i, chunk in enumerate(chunks):
    print(f"--- JS Chunk {i+1} ---")
    print(chunk[:200])
    print()


#----------------------------------------------------------------------------------------------------------------

# 10. HTMLHeaderTextSplitter

# Splits HTML by header tags, preserving structure.


from langchain.text_splitter import HTMLHeaderTextSplitter

html_text = """
<!DOCTYPE html>
<html>
<head><title>My Page</title></head>
<body>
    <h1>Main Title</h1>
    <p>Introduction paragraph here.</p>
    
    <h2>Section One</h2>
    <p>Content for section one.</p>
    <p>More content here.</p>
    
    <h3>Subsection 1.1</h3>
    <p>Details about subsection.</p>
    
    <h2>Section Two</h2>
    <p>Content for section two.</p>
    
    <h3>Subsection 2.1</h3>
    <p>More details.</p>
</body>
</html>
"""

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3")
]

splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

chunks = splitter.split_text(html_text)

for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i+1} ---")
    print(f"Content: {chunk.page_content[:100]}...")
    print(f"Metadata: {chunk.metadata}\n")


#----------------------------------------------------------------------------------------------------------------

# 11. HTMLSectionSplitter

# Splits HTML by semantic sections (div, section, article).


from langchain.text_splitter import HTMLSectionSplitter

html_text = """
<html>
<body>
    <article>
        <h1>News Article</h1>
        <p>Breaking news content here.</p>
    </article>
    <section>
        <h2>Related Stories</h2>
        <p>Story one content.</p>
        <p>Story two content.</p>
    </section>
    <div>
        <h2>Comments</h2>
        <p>User comment one.</p>
    </div>
</body>
</html>
"""

splitter = HTMLSectionSplitter(
    headers_to_split_on=[("h1", "Header 1"), ("h2", "Header 2")]
)

chunks = splitter.split_text(html_text)

for chunk in chunks:
    print(f"Content: {chunk.page_content[:80]}...")
    print(f"Metadata: {chunk.metadata}\n")


#----------------------------------------------------------------------------------------------------------------

# 12. LatexTextSplitter

# Splits LaTeX documents respecting commands and environments.


from langchain.text_splitter import LatexTextSplitter

latex_text = r"""
\documentclass{article}
\begin{document}

\title{My Research Paper}
\author{John Doe}
\maketitle

\begin{abstract}
This is the abstract of my paper. It summarizes the key contributions.
\end{abstract}

\section{Introduction}
The introduction provides background and motivation.

\subsection{Problem Statement}
We address the following problem...

\begin{equation}
E = mc^2
\end{equation}

\section{Methodology}
Our approach uses deep learning.

\begin{itemize}
    \item Data collection
    \item Model training
    \item Evaluation
\end{itemize}

\section{Results}
The results show significant improvement.

\end{document}
"""

splitter = LatexTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)

chunks = splitter.split_text(latex_text)

for i, chunk in enumerate(chunks):
    print(f"--- LaTeX Chunk {i+1} ---")
    print(chunk[:200])
    print()


#----------------------------------------------------------------------------------------------------------------

# 13. SemanticChunker

# Splits based on semantic similarity (embeddings) — keeps related sentences together.


from langchain.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

text = """
Machine learning is a subset of artificial intelligence. 
It enables systems to learn from data. 
Deep learning is a type of machine learning. 
It uses neural networks with many layers. 
Python is a popular programming language. 
It was created by Guido van Rossum. 
Python has a simple syntax. 
Many data scientists use Python for analysis. 
The Eiffel Tower is in Paris. 
It was completed in 1889. 
It is a famous landmark in France. 
Tourism is important for the French economy.
"""

# Requires OpenAI API key
splitter = SemanticChunker(
    OpenAIEmbeddings(),
    breakpoint_threshold_type="percentile",  # or "standard_deviation", "interquartile"
    breakpoint_threshold_amount=85           # Adjust sensitivity
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Semantic Chunk {i+1} ---")
    print(chunk.strip())
    print()

# Groups: ML/AI sentences together, Python sentences together, Paris/tourism together


#----------------------------------------------------------------------------------------------------------------

# 14. AgenticChunker

# Uses an LLM to intelligently decide chunk boundaries.


from langchain.text_splitter import AgenticChunker
from langchain_openai import ChatOpenAI

text = """
Introduction to Neural Networks

Neural networks are inspired by biological neurons. 
They consist of layers of interconnected nodes. 
Each connection has a weight that is adjusted during training.

Backpropagation Algorithm

The backpropagation algorithm calculates gradients. 
It uses the chain rule of calculus. 
Gradients flow backward through the network. 
This allows efficient weight updates.

Convolutional Neural Networks

CNNs are designed for image processing. 
They use convolutional layers to detect features. 
Pooling layers reduce spatial dimensions. 
Popular architectures include ResNet and VGG.

Recurrent Neural Networks

RNNs process sequential data. 
They have memory of previous inputs. 
LSTM and GRU are variants that solve vanishing gradient problems.
"""

splitter = AgenticChunker(
    ChatOpenAI(temperature=0),
    max_chunk_size=300
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Agentic Chunk {i+1} ---")
    print(chunk[:200])
    print()


#----------------------------------------------------------------------------------------------------------------

# 15. NLPTextSplitter (Experimental)

# Uses transformer models for semantic splitting.


from langchain.text_splitter import NLPTextSplitter

text = """
Artificial intelligence continues to evolve rapidly. 
New models are released every month. 
Companies invest billions in AI research. 
The technology impacts healthcare, finance, and education. 
However, ethical concerns remain. 
Bias in AI systems is a major issue. 
Privacy is another critical consideration. 
Regulations are being developed worldwide. 
The EU AI Act is a comprehensive framework. 
It classifies AI systems by risk level.
"""

splitter = NLPTextSplitter(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}: {chunk.strip()}")


#----------------------------------------------------------------------------------------------------------------

# 16. Custom Text Splitter

# Build your own splitter by extending the base class.


from langchain.text_splitter import TextSplitter
from langchain.schema import Document
from typing import List

class RegexTextSplitter(TextSplitter):
    """Split text using custom regex patterns."""
    
    def __init__(self, pattern: str, **kwargs):
        super().__init__(**kwargs)
        self.pattern = pattern
        import re
        self.regex = re.compile(pattern)
    
    def split_text(self, text: str) -> List[str]:
        """Split text using the regex pattern."""
        chunks = self.regex.split(text)
        # Remove empty strings and strip whitespace
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def create_documents(self, texts: List[str], metadatas: List[dict] = None) -> List[Document]:
        """Create documents with metadata."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, (text, metadata) in enumerate(zip(texts, _metadatas)):
            chunks = self.split_text(text)
            for j, chunk in enumerate(chunks):
                doc_metadata = {
                    **metadata,
                    "chunk_index": j,
                    "splitter": "RegexTextSplitter"
                }
                documents.append(Document(page_content=chunk, metadata=doc_metadata))
        return documents

# Usage: Split on chapter markers
text = """
CHAPTER 1
This is the first chapter content.
It has multiple paragraphs.

CHAPTER 2
This is the second chapter.
More content here.

CHAPTER 3
Final chapter content.
"""

splitter = RegexTextSplitter(
    pattern=r"CHAPTER \d+",
    chunk_size=1000,  # Not used but required by base class
    chunk_overlap=0
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"--- Chapter Chunk {i+1} ---")
    print(chunk)
    print()

#----------------------------------------------------------------------------------------------------------------

# 17. Splitting Documents (Not Just Text)

# All splitters work with `Document` objects too.


from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

documents = [
    Document(
        page_content="Long content here...",
        metadata={"source": "doc1.pdf", "page": 1}
    ),
    Document(
        page_content="Another long content...",
        metadata={"source": "doc2.pdf", "page": 1}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

# Preserves metadata in each chunk
chunks = splitter.split_documents(documents)

for chunk in chunks:
    print(f"Content: {chunk.page_content[:50]}...")
    print(f"Metadata: {chunk.metadata}")  # Includes original metadata + chunk info
    print()



#----------------------------------------------------------------------------------------------------------------

# Best Practices

# 1. Use RecursiveCharacterTextSplitter as default — it handles most cases well

# 2. Match chunk size to your embedding model — OpenAI: 8191 tokens, BGE: 512 tokens

# 3. Always use overlap — 10-20% of chunk size preserves context across boundaries

# 4. Use language-specific splitters for code — they respect syntax boundaries

# 5. Use semantic chunking for long documents — keeps related content together

# 6. Preserve metadata — always track source, page, header in chunk metadata

#----------------------------------------------------------------------------------------------------------------


# Production-ready pattern
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader

# Load
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# Split with optimal settings for RAG
splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    model_name="gpt-4",
    chunk_size=500,      # ~375 words
    chunk_overlap=100,   # 20% overlap
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_documents(documents)

print(f"Original docs: {len(documents)}")
print(f"Chunks: {len(chunks)}")
print(f"Avg chunk size: {sum(len(c.page_content) for c in chunks) / len(chunks):.0f} chars")


#----------------------------------------------------------------------------------------------------------------


