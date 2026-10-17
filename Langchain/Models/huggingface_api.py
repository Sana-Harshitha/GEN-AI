

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    max_new_tokens=100,
    provider="auto",
    timeout=60,
)

model = ChatHuggingFace(llm=llm)

try:
    response = model.invoke(
        "What is an LLM? Explain in 3 simple sentences."
    )
    print(response.content)

except Exception as e:
    print(f"Error: {e}")
