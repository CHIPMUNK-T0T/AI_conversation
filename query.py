import requests

url = "http://127.0.0.1:1234/v1/chat/completions"

payload = {
    "model": "llama-3.1-swallow-8b-instruct-v0.5",
    "messages": [
        {
            "role": "user",
            "content": "こんにちは。自己紹介してください。"
        }
    ],
    "temperature": 0.7
}

response = requests.post(url, json=payload)
response.raise_for_status()

data = response.json()
print(data["choices"][0]["message"]["content"])


from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio"
)

res = client.chat.completions.create(
    model="llama-3.1-swallow-8b-instruct-v0.5",
    messages=[
        {"role": "user", "content": "こんにちは。面白い話して。"}
    ],
)

print(res.choices[0].message.content)
