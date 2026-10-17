from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

# Load your environment variables (make sure GOOGLE_API_KEY is in your .env file)
load_dotenv()

# 1. Initialize the Google Gemini model
model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_retries=0,
    timeout=60,
)

# 2. Define the Chat Prompt Template using tuples and MessagesPlaceholder (matching the video's pattern)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer all queries accurately."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# 3. Initialize the chat history list to store past turns
chat_history = []

print("--- AI Chatbot Initialized (Type 'exit' to quit) ---")

# 4. Run the interactive conversational loop
while True:
    try:
        user_input = input("\nYou: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nGoodbye!")
        break

    if not user_input:
        continue

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    # 5. Format the prompt template by passing the running history and current user input
    prompt_messages = chat_prompt.invoke({
        "chat_history": chat_history,
        "user_input": user_input
    })

    # 6. Invoke the model directly with the structured prompt messages (No Chains)
    result = model.invoke(prompt_messages)

    # Extract the response content safely depending on the SDK version
    response_text = result.content if hasattr(result, "content") else result.text
    print("AI:", response_text)

    # 7. Append the current turn to the chat history for context in the next loop
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response_text))