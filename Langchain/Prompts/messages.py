
from langchain_core.messages import SystemMessage,AIMessage,HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_retries=0,
    timeout=60,
)


chat_history = [
    SystemMessage(content="You are a helpful assistant")
]
while True:
    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=user_input))
    result = model.invoke(chat_history)
    chat_history.append(AIMessage(content=result.text))
    print("AI:", result.text)
