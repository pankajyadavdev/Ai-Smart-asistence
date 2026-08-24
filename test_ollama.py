import requests


url = "http://localhost:11434/api/generate"

data = {
    "model": "llama3.2:3b" 
    "prompt": "Explain Artificial Intelligence in two sentences.",
    "stream": False
}


response = requests.post(
    url,
    json=data,
    timeout=120
)


print(response.json()["response"])