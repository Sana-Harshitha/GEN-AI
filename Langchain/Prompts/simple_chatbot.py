

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_retries=0,
    timeout=60,
)


chat_history = []
while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    chat_history.append(user_input)
    result = model.invoke(chat_history)
    chat_history.append(result)
    print("AI:", result.text)
