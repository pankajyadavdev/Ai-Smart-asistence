import requests


# ============================================================
# CONFIG
# ============================================================

OLLAMA_URL = "http://localhost:11434/api/generate"

MODEL_NAME = "llama3.2:3b"

TIMEOUT = 600

# For small PDFs, use one request.
SMALL_PDF_PAGES = 10

# Larger PDFs are processed in larger sections.
LARGE_BATCH_SIZE = 10

# Maximum generated tokens.
SECTION_TOKENS = 350
FINAL_TOKENS = 700


# ============================================================
# CHECK OLLAMA
# ============================================================

def check_ollama():

    try:

        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=10,
        )

        response.raise_for_status()

        models = response.json().get(
            "models",
            []
        )

        model_names = [
            model.get("name", "")
            for model in models
        ]

        if MODEL_NAME not in model_names:

            return False, (
                f"Model '{MODEL_NAME}' was not found. "
                f"Available models: {model_names}"
            )

        return True, ""

    except requests.exceptions.ConnectionError:

        return False, (
            "Ollama is not running. "
            "Start Ollama and try again."
        )

    except Exception as error:

        return False, str(error)


# ============================================================
# OLLAMA REQUEST
# ============================================================

def ask_ollama(
    prompt,
    max_tokens,
):

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,

            # Keep the model loaded between requests.
            "keep_alive": "10m",

            "options": {
                "temperature": 0.2,
                "num_predict": max_tokens,

                # Helps prevent unnecessary long context.
                "num_ctx": 8192,
            },
        },
        timeout=TIMEOUT,
    )

    response.raise_for_status()

    data = response.json()

    answer = data.get(
        "response",
        "",
    ).strip()

    if not answer:

        raise RuntimeError(
            "Ollama returned an empty response."
        )

    return answer


# ============================================================
# BUILD TEXT
# ============================================================

def build_text(documents):

    parts = []

    for document in documents:

        text = document.get(
            "text",
            "",
        ).strip()

        if not text:
            continue

        page = document.get(
            "page",
            "?",
        )

        parts.append(
            f"Page {page}:\n{text}"
        )

    return "\n\n".join(parts)


# ============================================================
# SUMMARIZE ONE SECTION
# ============================================================

def summarize_section(documents):

    text = build_text(documents)

    if not text:
        return ""

    prompt = f"""
You are a concise study assistant.

Summarize the following PDF content.

Focus only on:
- main ideas
- important concepts
- definitions
- important facts
- formulas
- technical details

Remove repetition.

Do not invent information.

Use short headings and bullet points.

Keep the summary concise.

PDF CONTENT:

{text}

SUMMARY:
"""

    return ask_ollama(
        prompt,
        SECTION_TOKENS,
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

def create_final_summary(
    summaries,
):

    if not summaries:
        return ""

    combined = "\n\n".join(
        summaries
    )

    prompt = f"""
Create a concise study guide from
the following section summaries.

Use:

# Overview

# Main Topics

# Important Concepts

# Important Definitions

# Important Facts

# Key Takeaways

Rules:
- Remove duplicate information.
- Keep important technical details.
- Do not invent information.
- Keep it concise.
- Make it useful for exam revision.

SECTION SUMMARIES:

{combined}

FINAL STUDY GUIDE:
"""

    return ask_ollama(
        prompt,
        FINAL_TOKENS,
    )


# ============================================================
# MAIN PDF SUMMARIZER
# ============================================================

def summarize_pdf(documents):

    if not documents:

        return "No PDF content was found."

    # --------------------------------------------------------
    # Remove empty documents
    # --------------------------------------------------------

    documents = [
        document
        for document in documents
        if document.get(
            "text",
            "",
        ).strip()
    ]

    if not documents:

        return (
            "No readable text was found "
            "in this PDF."
        )

    # --------------------------------------------------------
    # Check Ollama
    # --------------------------------------------------------

    ok, error = check_ollama()

    if not ok:

        raise RuntimeError(error)

    # --------------------------------------------------------
    # SMALL PDF
    #
    # One request instead of multiple requests.
    # --------------------------------------------------------

    if len(documents) <= SMALL_PDF_PAGES:

        return summarize_section(
            documents
        )

    # --------------------------------------------------------
    # LARGE PDF
    # --------------------------------------------------------

    batches = []

    for i in range(
        0,
        len(documents),
        LARGE_BATCH_SIZE,
    ):

        batch = documents[
            i:i + LARGE_BATCH_SIZE
        ]

        batches.append(batch)

    section_summaries = []

    # --------------------------------------------------------
    # Summarize sections
    # --------------------------------------------------------

    for batch in batches:

        summary = summarize_section(
            batch
        )

        if summary:

            section_summaries.append(
                summary
            )

    if not section_summaries:

        return (
            "The model did not generate "
            "a summary."
        )

    # --------------------------------------------------------
    # Only one section
    # --------------------------------------------------------

    if len(section_summaries) == 1:

        return section_summaries[0]

    # --------------------------------------------------------
    # Final consolidation
    # --------------------------------------------------------

    return create_final_summary(
        section_summaries
    )
