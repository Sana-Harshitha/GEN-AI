
# DONT DOWNLOAD 

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=100,
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




    