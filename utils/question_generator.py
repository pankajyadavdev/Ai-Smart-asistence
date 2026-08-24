import os
import json
import re
import requests


OLLAMA_URL = "https://ollama.com/api/generate"

MODEL_NAME = os.getenv(
    "OLLAMA_MODEL",
    "gpt-oss:20b"
)

API_KEY = os.getenv(
    "OLLAMA_API_KEY"
)


def generate_questions(
    context,
    num_questions=5,
    difficulty="Medium",
    question_type="MCQ"
):

    if not API_KEY:
        raise ValueError(
            "OLLAMA_API_KEY is missing."
        )

    if not context or not context.strip():
        raise ValueError(
            "No study material was provided."
        )

    if question_type == "MCQ":

        format_text = """
[
  {
    "question": "Question text",
    "options": [
      "Option A",
      "Option B",
      "Option C",
      "Option D"
    ],
    "answer": "Option A",
    "explanation": "Short explanation"
  }
]
"""

    else:

        format_text = """
[
  {
    "question": "Question text",
    "answer": "Correct answer",
    "explanation": "Short explanation"
  }
]
"""

    prompt = f"""
You are a college exam question generator.

Create exactly {num_questions} questions.

Difficulty:
{difficulty}

Question type:
{question_type}

Use ONLY the study material below.

Do not invent information.

Return ONLY valid JSON.
Do not use markdown.
Do not add text before or after the JSON.

Required JSON format:

{format_text}

STUDY MATERIAL:

{context}
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
            "Question generation timed out."
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
            f"Ollama error: "
            f"{result['error']}"
        )


    raw = result.get(
        "response",
        ""
    ).strip()


    if not raw:

        raise ValueError(
            "Ollama returned an empty response."
        )


    questions = parse_questions(raw)


    if not questions:

        raise ValueError(
            "Could not convert Ollama's response "
            "into questions.\n\n"
            f"Ollama response:\n{raw}"
        )


    return questions[:num_questions]


def parse_questions(text):

    # Remove markdown code blocks
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"```",
        "",
        text
    )

    text = text.strip()


    # Try complete JSON
    try:

        data = json.loads(text)

        if isinstance(data, list):

            return clean_questions(data)

    except json.JSONDecodeError:
        pass


    # Find JSON array inside response
    start = text.find("[")
    end = text.rfind("]")


    if start != -1 and end != -1:

        json_text = text[
            start:end + 1
        ]

        try:

            data = json.loads(
                json_text
            )

            if isinstance(data, list):

                return clean_questions(
                    data
                )

        except json.JSONDecodeError:
            pass


    return []


def clean_questions(data):

    questions = []


    for item in data:

        if not isinstance(
            item,
            dict
        ):
            continue


        question = str(
            item.get(
                "question",
                ""
            )
        ).strip()


        answer = str(
            item.get(
                "answer",
                ""
            )
        ).strip()


        if not question or not answer:
            continue


        options = item.get(
            "options",
            []
        )


        if not isinstance(
            options,
            list
        ):
            options = []


        options = [
            str(option).strip()
            for option in options
            if str(option).strip()
        ]


        questions.append(
            {
                "question": question,
                "options": options,
                "answer": answer,
                "explanation": str(
                    item.get(
                        "explanation",
                        ""
                    )
                ).strip()
            }
        )


    return questions
