from llm.interface import get_llm

llm = get_llm(
    backend="ollama",
    model="llama3.2:1b",
)

print("Backend initialized:", type(llm).__name__)

query = (
    "Plan a 3-day trip from Seattle to Yakima "
    "for 2 people with a budget of $1500."
)

constraints = llm.extract_constraints(query)

print("\nExtracted constraints:")
print(constraints)

print("\nOllama integration test completed.")
