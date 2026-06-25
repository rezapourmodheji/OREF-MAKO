from openai import OpenAI

client = OpenAI(api_key="...")

response = client.responses.create(
    model="gpt-5",
    input="Write a Python function to compute Fibonacci numbers."
)

print(response.output_text)