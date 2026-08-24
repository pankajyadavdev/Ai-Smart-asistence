import requests
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"

def generate_answer(question, context):

    prompt = f"""
You are an AI College Assistant.

Answer the student's question using ONLY the information
provided in the document context.

Do not make up information.

If the answer is not present in the document, say:

"I could not find this information in the uploaded document."

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=120
    )

    response.raise_for_status()

    result = response.json()

    return result["response"]