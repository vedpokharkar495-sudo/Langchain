# Langchain




---

## What is LangChain?

**Simple Definition**

LangChain is a framework that helps developers build applications powered by Large Language Models (LLMs).

Think of it as a manager that coordinates everything needed for an AI application.


---

**Why Do We Need LangChain?**

Suppose you want to build a chatbot.

The chatbot should:

answer questions

search PDFs

remember previous messages

search a database

use tools

browse documents


Doing this manually becomes difficult.

LangChain provides ready-made components.


---

## Main Components of LangChain

Think of LangChain like LEGO blocks.

LangChain

      ┌───────────────┐
      │ Prompt        │
      └───────────────┘
               │
               ▼
      ┌───────────────┐
      │ LLM           │
      └───────────────┘
               │
               ▼
      ┌───────────────┐
      │ Output Parser │
      └───────────────┘
               │
               ▼
        Final Response

As you learn more, you'll add:

Memory

Chains

Embeddings

Vector Databases

Retrievers

Agents

Tools



---

Core Concept 1: LLM

An LLM is simply an AI model.

Examples:

GPT-4

GPT-5 family

Llama

Mistral

Gemini


It takes text as input.

Question
     │
     ▼
LLM
     │
     ▼
Answer

Example:

Input:

What is Python?

Output:

Python is a programming language.


---

Core Concept 2: Prompt

A prompt is simply the instruction you give the LLM.

Example:

Explain Machine Learning.

or

Explain Machine Learning like I'm 10 years old.

Better prompt = Better answer.


---

Core Concept 3: Model

LangChain connects to different models.

Example:

OpenAI

Anthropic

Llama

Gemini

Mistral

Changing the model often requires changing only a few lines of code.


---

First LangChain Program

Install

pip install langchain
pip install langchain-openai
pip install python-dotenv


---

Create a .env file:

OPENAI_API_KEY=your_api_key

---

Temperature

Temperature controls creativity.

0.0

Very factual.

0.5

Balanced.

1.0

Creative.

Example:

Question:

Write a story.

Temperature = 0

Simple story

Temperature = 1

Creative and imaginative story


---

Prompt Templates

Instead of writing prompts manually every time, use templates.




---

Why Prompt Templates?

Suppose 1,000 users ask different questions.

Instead of writing:

"Explain Python"

"Explain AI"

"Explain ML"

"Explain SQL"

You write one reusable template:

"Explain {topic}"


---



