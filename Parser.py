
# Output parser in langchain 

# 1. PydanticOutputParser

# The most powerful parser — extracts structured data into Pydantic models with validation.


from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from pydantic import BaseModel, Field
from typing import List

# Define your schema
class Movie(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    genres: List[str] = Field(description="List of genres")
    rating: float = Field(description="IMDb rating out of 10")

class MovieList(BaseModel):
    movies: List[Movie] = Field(description="List of recommended movies")

parser = PydanticOutputParser(pydantic_object=MovieList)

template = """Recommend 3 movies similar to {query}.
{format_instructions}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Simulated LLM output (in practice, use real LLM)
llm_output = '''
{
    "movies": [
        {"title": "Inception", "year": 2010, "genres": ["Sci-Fi", "Thriller"], "rating": 8.8},
        {"title": "Interstellar", "year": 2014, "genres": ["Sci-Fi", "Drama"], "rating": 8.7},
        {"title": "The Matrix", "year": 1999, "genres": ["Sci-Fi", "Action"], "rating": 8.7}
    ]
}
'''

result = parser.parse(llm_output)
print(result.movies[0].title)  # "Inception"
print(result.movies[0].year)   # 2010


#----------------------------------------------------------------------------------------------------------------

#2. StructuredOutputParser

# Returns a dictionary with predefined fields — simpler than Pydantic, no validation.


from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Define what fields you expect
schemas = [
    ResponseSchema(name="answer", description="Answer to the user's question"),
    ResponseSchema(name="confidence", description="Confidence score from 0-100"),
    ResponseSchema(name="sources", description="List of sources used")
]

parser = StructuredOutputParser.from_response_schemas(schemas)

template = """Answer the following question: {question}
{format_instructions}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["question"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm_output = '''
```json
{
    "answer": "Python was created by Guido van Rossum",
    "confidence": 98,
    "sources": ["python.org", "wikipedia.org"]
}
```

'''

result = parser.parse(llm_output)
print(result["answer"])      # "Python was created by Guido van Rossum"
print(result["confidence"])  # "98"



#----------------------------------------------------------------------------------------------------------------

## 3. **CommaSeparatedListOutputParser**

# Parses comma-separated values into a Python list.

from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

template = """List {count} programming languages.
{format_instructions}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["count"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm_output = "Python, JavaScript, Rust, Go, TypeScript"
result = parser.parse(llm_output)
print(result)  # ['Python', 'JavaScript', 'Rust', 'Go', 'TypeScript']



#----------------------------------------------------------------------------------------------------------------

# 4. EnumOutputParser

# Ensures output matches one of predefined enum values.


from langchain.output_parsers import EnumOutputParser
from enum import Enum

class Sentiment(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

parser = EnumOutputParser(enum=Sentiment)

template = """Analyze the sentiment of this review: {review}
Choose one: {options}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["review"],
    partial_variables={"options": parser.get_format_instructions()}
)

llm_output = "positive"
result = parser.parse(llm_output)
print(result)           # <Sentiment.POSITIVE: 'positive'>
print(result.value)     # "positive"



#----------------------------------------------------------------------------------------------------------------

# 5. BooleanOutputParser

# Parses various text representations of true/false into Python booleans.


from langchain.output_parsers import BooleanOutputParser

parser = BooleanOutputParser()

# Handles: yes/no, true/false, 1/0, on/off (case insensitive)
outputs = ["Yes", "NO", "true", "False", "1", "0"]

for output in outputs:
    result = parser.parse(output)
    print(f"{output} -> {result}")
# Yes -> True
# NO -> False
# true -> True
# False -> False
# 1 -> True
# 0 -> False



#----------------------------------------------------------------------------------------------------------------

# 6. DatetimeOutputParser

# Parses date/time strings into Python datetime objects.


from langchain.output_parsers import DatetimeOutputParser
from datetime import datetime

parser = DatetimeOutputParser()

template = """When was {event}? Provide the date.
{format_instructions}"""

prompt = PromptTemplate(
    template=template,
    input_variables=["event"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

llm_output = "1969-07-20"
result = parser.parse(llm_output)
print(result)           # 1969-07-20 00:00:00
print(type(result))     # <class 'datetime.datetime'>



#----------------------------------------------------------------------------------------------------------------

# 7. RegexParser

# Extracts data using custom regular expressions.


from langchain.output_parsers import RegexParser

# Extract name, age, and email using regex groups
parser = RegexParser(
    regex=r"Name: (.*?)\nAge: (\d+)\nEmail: (.*?)\n",
    output_keys=["name", "age", "email"]
)

llm_output = '''
Name: Alice Johnson
Age: 28
Email: alice@example.com
'''

result = parser.parse(llm_output)
print(result["name"])   # "Alice Johnson"
print(result["age"])    # "28"
print(result["email"])  # "alice@example.com"



#----------------------------------------------------------------------------------------------------------------

# 8. OutputFixingParser

# Wraps another parser and uses an LLM to fix malformed output.


from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from langchain.llms import OpenAI
from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")

base_parser = PydanticOutputParser(pydantic_object=Person)

# Wrap with fixer — if parsing fails, LLM attempts to fix it
parser = OutputFixingParser.from_llm(parser=base_parser, llm=OpenAI())

# Malformed JSON (missing quotes around name, extra comma)
bad_output = '{name: John, age: 30,}'

try:
    result = parser.parse(bad_output)
    print(result.name)  # "John"
    print(result.age)   # 30
except Exception as e:
    print(f"Failed even after fixing: {e}")



#----------------------------------------------------------------------------------------------------------------

# 9. RetryOutputParser

# Similar to OutputFixingParser but with configurable retry logic.


from langchain.output_parsers import RetryOutputParser
from langchain.llms import OpenAI

base_parser = PydanticOutputParser(pydantic_object=Person)
parser = RetryOutputParser.from_llm(parser=base_parser, llm=OpenAI())

# Will retry up to max_retries times if parsing fails
result = parser.parse_with_prompt(bad_output, prompt_value=prompt)



#----------------------------------------------------------------------------------------------------------------

# 10. JSONOutputParser

# Parses any JSON output into Python dicts/lists (LangChain newer versions / LCEL).


from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class Product(BaseModel):
    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    tags: List[str] = Field(description="Product tags")

parser = JsonOutputParser(pydantic_object=Product)

llm_output = '''
```json
{
    "name": "Wireless Headphones",
    "price": 99.99,
    "tags": ["electronics", "audio", "wireless"]
}
```

'''

result = parser.parse(llm_output)
print(result["name"])   # "Wireless Headphones"
print(result["price"])  # 99.99




#----------------------------------------------------------------------------------------------------------------

## 11. **StrOutputParser**

# Simply returns the raw string output (default behavior, useful in chains).


from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()

llm_output = "This is the raw text from the LLM."
result = parser.parse(llm_output)
print(result)  # "This is the raw text from the LLM."
print(type(result))  # <class 'str'>



#----------------------------------------------------------------------------------------------------------------

# 12. XMLOutputParser

# Parses XML-formatted LLM output into dictionaries.


from langchain.output_parsers import XMLOutputParser

parser = XMLOutputParser()

llm_output = '''
<response>
    <title>Introduction to Machine Learning</title>
    <author>Andrew Ng</author>
    <chapters>
        <chapter>Linear Regression</chapter>
        <chapter>Neural Networks</chapter>
        <chapter>SVM</chapter>
    </chapters>
</response>
'''

result = parser.parse(llm_output)
print(result["title"])     # "Introduction to Machine Learning"
print(result["chapters"])  # OrderedDict or list of chapters



#----------------------------------------------------------------------------------------------------------------

# 13. YamlOutputParser

# Parses YAML output (useful for human-readable structured data).


from langchain.output_parsers import YamlOutputParser

parser = YamlOutputParser(pydantic_object=Person)

llm_output = '''
name: Sarah Connor
age: 29
'''

result = parser.parse(llm_output)
print(result.name)  # "Sarah Connor"
print(result.age)   # 29



#----------------------------------------------------------------------------------------------------------------

# 14. PandasDataFrameOutputParser

# Directly parses output into a pandas DataFrame.


from langchain.output_parsers import PandasDataFrameOutputParser
import pandas as pd

# Define expected columns
parser = PandasDataFrameOutputParser(
    dataframe=pd.DataFrame({
        "product": pd.Series(dtype="str"),
        "sales": pd.Series(dtype="int"),
        "region": pd.Series(dtype="str")
    })
)

llm_output = '''
product,sales,region
Laptop,150,North
Phone,300,South
Tablet,80,East
'''

df = parser.parse(llm_output)
print(df)
#    product  sales region
# 0   Laptop    150  North
# 1    Phone    300  South
# 2   Tablet     80   East




#----------------------------------------------------------------------------------------------------------------

# 1. Always use `get_format_instructions()` — it tells the LLM exactly how to format output

# 2. Wrap with `OutputFixingParser` for production — LLMs sometimes produce malformed JSON

# 3. Start simple — use `JsonOutputParser` or `StructuredOutputParser` before Pydantic

# 4. Test edge cases — LLMs may add markdown fences (`json), extra text, or comments

# 5. Use LCEL chains — modern LangChain syntax makes parser integration cleaner:
    

#----------------------------------------------------------------------------------------------------------------


from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

chain = (
    ChatPromptTemplate.from_template("Tell me about {topic} in JSON format")
    | ChatOpenAI()
    | JsonOutputParser()
)

result = chain.invoke({"topic": "quantum computing"})




#----------------------------------------------------------------------------------------------------------------
