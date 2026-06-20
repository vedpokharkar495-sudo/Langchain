# chat with open AI model 

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

response = llm.invoke("What is Artificial Intelligence?")

print(response.content)


