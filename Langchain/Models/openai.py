import os

from dotenv import load_dotenv
from langchain_openai import OpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Missing OPENAI_API_KEY in your .env file.")

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)

prompt = "Explain what a large language model is in one sentence."
response = llm.invoke(prompt)
print(response)

load_dotenv()

model = OpenAI(model="gpt4")
result = model.invoke("what is llm")
print(result)
print(result.content)