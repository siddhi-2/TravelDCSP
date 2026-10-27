
from ollama import chat

response = chat(
    model="llama3.2:1b",
    messages=[
        {
            "role": "system",
            "content": "You are a travel-planning assistant."
        },
        {
            "role": "user",
            "content": (
                "Suggest an inexpensive three-day trip "
                "for two people from Seattle to Yakima."
            )
        }
    ]
)

print(response.message.content)
