import json
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def verify_question(
    question,
    source_chunks,
):
    """
    Verify whether a generated question and
    answer are supported by the PDF source.

    Uses the local Ollama model.
    """

    if not question:
        return {
            "verified": False,
            "reason": "Question is empty.",
            "source": "",
            "page": None,
        }

    if not source_chunks:
        return {
            "verified": False,
            "reason": "No source material available.",
            "source": "",
            "page": None,
        }

    question_text = question.get(
        "question",
        "",
    )

    answer = question.get(
        "answer",
        "",
    )

    if not question_text:
        return {
            "verified": False,
            "reason": "Question text is missing.",
            "source": "",
            "page": None,
        }

    if not answer:
        return {
            "verified": False,
            "reason": "Answer is missing.",
            "source": "",
            "page": None,
        }

    # --------------------------------------------------------
    # BUILD SOURCE CONTEXT
    # --------------------------------------------------------

    source_parts = []

    for chunk in source_chunks:

        source = chunk.get(
            "source",
            "Unknown",
        )

        page = chunk.get(
            "page",
            1,
        )

        text = chunk.get(
            "text",
            "",
        )

        source_parts.append(
            f"""
SOURCE: {source}
PAGE: {page}

TEXT:
{text}
"""
        )

    source_context = "\n\n".join(
        source_parts
    )

    # --------------------------------------------------------
    # VERIFICATION PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are a strict academic source verifier.

Determine whether the following question
and answer are supported by the source material.

QUESTION:
{question_text}

ANSWER:
{answer}

SOURCE MATERIAL:
{source_context}

RULES:

1. Use ONLY the source material.
2. Do NOT use outside knowledge.
3. The answer must be supported by the source.
4. If the source does not support the answer,
   mark verified as false.
5. Identify the best source and page.
6. Return ONLY valid JSON.
7. Do not use Markdown.

Return:

{{
    "verified": true,
    "reason": "Short explanation",
    "source": "filename.pdf",
    "page": 1
}}

OR:

{{
    "verified": false,
    "reason": "The source does not support the answer.",
    "source": "",
    "page": null
}}
"""

    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0
        },
    }

    try:

        response = requests.post(
            OLLAMA_URL,
            json=data,
            timeout=180,
        )

        response.raise_for_status()

        result = response.json()

        content = result.get(
            "response",
            "",
        )

        if not content:
            raise ValueError(
                "Ollama returned an empty response."
            )

        content = clean_json_response(
            content
        )

        verification = json.loads(
            content
        )

        return {
            "verified": bool(
                verification.get(
                    "verified",
                    False,
                )
            ),
            "reason": verification.get(
                "reason",
                "No verification reason provided.",
            ),
            "source": verification.get(
                "source",
                "",
            ),
            "page": verification.get(
                "page",
                None,
            ),
        }

    except Exception as error:

        return {
            "verified": False,
            "reason": (
                "Verification failed: "
                + str(error)
            ),
            "source": "",
            "page": None,
        }


def verify_questions(
    questions,
    source_chunks,
):
    """
    Verify all generated questions.
    """

    verified_questions = []

    for question in questions:

        verification = verify_question(
            question,
            source_chunks,
        )

        question[
            "verified"
        ] = verification[
            "verified"
        ]

        question[
            "verification_reason"
        ] = verification[
            "reason"
        ]

        if verification.get(
            "source"
        ):

            question[
                "source"
            ] = verification[
                "source"
            ]

        if verification.get(
            "page"
        ) is not None:

            question[
                "page"
            ] = verification[
                "page"
            ]

        verified_questions.append(
            question
        )

    return verified_questions


def clean_json_response(
    content
):
    """
    Remove Markdown code fences from
    Ollama responses.
    """

    content = content.strip()

    content = re.sub(
        r"^```json\s*",
        "",
        content,
        flags=re.IGNORECASE,
    )

    content = re.sub(
        r"^```\s*",
        "",
        content,
    )

    content = re.sub(
        r"\s*```$",
        "",
        content,
    )

    return content.strip()
