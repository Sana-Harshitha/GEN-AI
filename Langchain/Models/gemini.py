
# import os
# from dotenv import load_dotenv
# from google import genai

# load_dotenv()

# client = genai.Client(
#     api_key=os.getenv("GOOGLE_API_KEY")
# )

# print("Available Gemini models:\n")

# for model in client.models.list():
#     name = model.name or ""
#     methods = getattr(model, "supported_actions", None) or []

#     if "flash" in name.lower():
#         print(f"Model: {name}")
#         print(f"Supported actions: {methods}")
#         print("-" * 40)




import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_retries=0,
    timeout=60,
)

try:
    start = time.time()

    response = model.invoke(
        "Explain RAG in 3 simple sentences."
    )

    print(response.content)
    print(f"\nResponse time: {time.time() - start:.2f} seconds")

except Exception as e:
    print(f"Error: {e}")



