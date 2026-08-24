import os
import requests


OLLAMA_URL = "https://ollama.com/api/generate"

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "gpt-oss:20b"
)

API_KEY = os.getenv(
    "OLLAMA_API_KEY"
)


def summarize_pdf(documents):

    if not API_KEY:
        raise RuntimeError(
            "OLLAMA_API_KEY is missing. "
            "Add it in Streamlit Cloud Secrets."
        )

    if not documents:
        raise ValueError(
            "No document content was provided."
        )

    # Build document text
    parts = []

    for doc in documents:

        text = doc.get("text", "").strip()

        if text:

            parts.append(
                f"Page {doc.get('page', 1)}:\n{text}"
            )

    context = "\n\n".join(parts)

    if not context:
        raise ValueError(
            "No text could be extracted from the PDF."
        )

    # Keep the prompt manageable
    context = context[:20000]

    prompt = f"""
You are an AI college study assistant.

Create a clear and useful summary of the
following study material.

Requirements:

- Use ONLY the provided document.
- Do not invent information.
- Include the main concepts.
- Include important definitions.
- Include important facts.
- Use headings and bullet points.
- Make the summary useful for exam preparation.
- Do not mention information outside the document.

DOCUMENT:

{context}

SUMMARY:
"""

    try:

        response = requests.post(
            OLLAMA_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json",
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
            "Summary generation timed out."
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to Ollama Cloud."
        )

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Ollama request failed: {error}"
        )

    if response.status_code != 200:

        raise RuntimeError(
            f"Ollama API error "
            f"{response.status_code}:\n"
            f"{response.text}"
        )

    try:

        result = response.json()

    except ValueError:

        raise RuntimeError(
            "Ollama returned invalid JSON."
        )

    if "error" in result:

        raise RuntimeError(
            f"Ollama error: {result['error']}"
        )

    summary = result.get(
        "response",
        ""
    ).strip()

    if not summary:

        raise RuntimeError(
            "Ollama returned an empty summary."
        )

    return summary
