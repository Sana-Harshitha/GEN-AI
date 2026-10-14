import os
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline


# Configure the local Hugging Face pipeline with TinyLlama
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 100
    }
)

# Wrap it in LangChain's ChatHuggingFace interface
model = ChatHuggingFace(llm=llm)

# Invoke the model with a prompt
result = model.invoke("What is the capital of India")

# Print the generated text response
print(result.content)