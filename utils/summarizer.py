import os
import requests
import streamlit as st


OLLAMA_URL = "https://ollama.com/api/generate"


# Get API key from Streamlit Cloud Secrets
API_KEY = st.secrets.get(
    "OLLAMA_API_KEY",
    os.getenv("OLLAMA_API_KEY")
)

MODEL_NAME = st.secrets.get(
    "OLLAMA_MODEL",
    os.getenv(
        "OLLAMA_MODEL",
        "gpt-oss:20b-cloud"
    )
)


def summarize_pdf(documents):

    if not API_KEY:
        raise RuntimeError(
            "OLLAMA_API_KEY was not found. "
            "Check Streamlit Cloud → "
            "Manage app → Settings → Secrets."
        )

    if not documents:
        raise ValueError(
            "No document content was provided."
        )

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

    context = context[:20000]

    prompt = f"""
You are an AI college study assistant.

Summarize the following study material.

Rules:
- Use only the provided material.
- Do not invent information.
- Include important concepts.
- Include important definitions.
- Include important facts.
- Use headings and bullet points.
- Make it useful for exam preparation.

STUDY MATERIAL:

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

    except requests.exceptions.RequestException as error:

        raise RuntimeError(
            f"Could not connect to Ollama: {error}"
        )

    if response.status_code != 200:

        raise RuntimeError(
            f"Ollama API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    try:
        result = response.json()
    except ValueError:

        raise RuntimeError(
            f"Invalid Ollama response: "
            f"{response.text}"
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
            f"Ollama returned no summary. "
            f"Response: {result}"
        )

    return summary
