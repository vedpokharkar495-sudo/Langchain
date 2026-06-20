# Prompt Templates
# Instead of writing prompts manually every time, use templates.

# =========================================

# Without template:

# question = "Explain Python"

# prompt = f"Explain {question}"

# =========================================


# With LangChain:

from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Explain the following topic: {topic}"
)

formatted = prompt.invoke({"topic": "Python"})

print(formatted.text) #Explain the following topic: Python






