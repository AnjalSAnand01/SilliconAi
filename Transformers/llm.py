from llama_cpp import Llama

# Path to your downloaded GGUF model
model_path = "/workspaces/SilliconAi/LLM_Models/mistral-7b-instruct.Q4_0.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=2048,     # Context size
    n_threads=2,    # Match your CPU core count
)

# Prompt the model
response = llm("How a n-channel Mosfet Works", max_tokens=200)
print(response["choices"][0]["text"].strip())