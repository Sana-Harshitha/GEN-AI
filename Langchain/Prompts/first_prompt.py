from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    max_retries=0,
    timeout=60,
)
prompt_template=PromptTemplate(
    template=(
        "Please summarize the research paper titled '{paper_input}' with the following specifications:\n"
        "1. Explanation Style: {style_input}\n"
        "2. Explanation Length: {length_input}\n\n"
        "Guidelines:\n"
        "- Include relevant mathematical equations if present in the paper.\n"
        "- Explain the mathematical concepts using simple, intuitive code snippets where applicable.\n"
        "- Use relatable analogies to simplify complex ideas.\n"
        "- If certain information is not available in the paper, respond with 'Insufficent information available' instead of guessing.\n"
        "- Ensure the summary is clear, accurate, and aligned with the provided style and length."
    
    ),
    input_variables=["paper_input", "style_input", "length_input"],
)

print("--- Research Paper Summarizer ---")

# 3. Take dynamic inputs from the user via the terminal
paper_input = input("Enter the research paper name (e.g., Attention Is All You Need): ")
style_input = input("Enter the explanation style (e.g., Code Heavy, Math Heavy, Simple and Intuitive): ")
length_input = input("Enter the summary length (e.g., Short, Medium, Long): ")

# 4. Fill the template with user inputs
formatted_prompt = prompt_template.format(
    paper_input=paper_input,
    style_input=style_input,
    length_input=length_input
)

print("\nGenerating summary, please wait...\n")

# 5. Invoke the model with the formatted prompt
result = model.invoke(formatted_prompt)

# 6. Display the output content
print("--- Summary Output ---")
print(result.text)
