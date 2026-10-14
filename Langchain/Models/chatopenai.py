from langchain_openai import ChatOpenAi

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAi(model="gpt-4")
result = model.invoke("what is llm")
print(result)
print(result.content)