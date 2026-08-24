import os
import requests


# ============================================================
# OLLAMA CLOUD
# ============================================================

OLLAMA_URL = "https://ollama.com/api/generate"

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "gpt-oss:20b"
)

API_KEY = os.getenv(
    "OLLAMA_API_KEY"
)


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, context):

    if not API_KEY:
        raise ValueError(
            "OLLAMA_API_KEY is missing. "
            "Add it in Streamlit Cloud → "
            "Manage app → Settings → Secrets."
        )

    prompt = f"""
You are an AI College Assistant.

Answer the student's question using ONLY
the information provided in the document context.

Rules:

1. Do not make up information.
2. Do not use outside knowledge.
3. If the answer is not present in the context,
   say exactly:

"I could not find this information in the uploaded document."

4. Give a clear and student-friendly answer.
5. Use bullet points when appropriate.

DOCUMENT CONTEXT:
{context}

STUDENT QUESTION:
{question}

ANSWER:
"""

    try:

        response = requests.post(
            OLLAMA_URL,

            headers={
                "Authorization": (
                    f"Bearer {API_KEY}"
                ),
                "Content-Type": (
                    "application/json"
                ),
            },

            json={
                "model": MODEL_NAME,
                "prompt": prompt,
                "stream": False,
            },

            timeout=180,
        )

    except requests.exceptions.Timeout:

        raise RuntimeError(
            "Ollama request timed out. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to Ollama Cloud."
        )

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Ollama request failed: {error}"
        )


    # ========================================================
    # CHECK HTTP RESPONSE
    # ========================================================

    if response.status_code != 200:

        raise RuntimeError(
            f"Ollama API error "
            f"({response.status_code}):\n"
            f"{response.text}"
        )


    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    try:

        result = response.json()

    except ValueError:

        raise RuntimeError(
            "Ollama returned an invalid JSON response:\n"
            f"{response.text}"
        )


    # ========================================================
    # CHECK RESPONSE
    # ========================================================

    if "error" in result:

        raise RuntimeError(
            f"Ollama error: "
            f"{result['error']}"
        )


    if "response" not in result:

        raise RuntimeError(
            "Ollama response does not contain "
            "'response'.\n"
            f"Response: {result}"
        )


    answer = result["response"].strip()


    if not answer:

        raise RuntimeError(
            "Ollama returned an empty answer."
        )


    return answer
