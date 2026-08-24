import json
import re
import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:3b"


def generate_questions(
    context,
    num_questions=5,
    difficulty="Medium",
    question_type="MCQ",
):

    if not context or not context.strip():
        raise ValueError(
            "No study material was provided."
        )

    if question_type == "MCQ":

        format_instruction = """
Return ONLY valid JSON.

Use exactly this format:

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

        format_instruction = """
Return ONLY valid JSON.

Use exactly this format:

[
  {
    "question": "Question text",
    "answer": "Correct answer",
    "explanation": "Short explanation"
  }
]
"""

    prompt = f"""
You are an exam question generator.

Create exactly {num_questions} questions
from the study material below.

Difficulty:
{difficulty}

Question type:
{question_type}

IMPORTANT:
- Use ONLY the provided study material.
- Do not invent facts.
- Create clear college-level questions.
- Return ONLY JSON.
- Do not use markdown.
- Do not add explanations outside the JSON.

{format_instruction}

STUDY MATERIAL:

{context}
"""

    data = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    response = requests.post(
        OLLAMA_URL,
        json=data,
        timeout=180,
    )

    response.raise_for_status()

    result = response.json()

    raw = result.get("response", "").strip()

    if not raw:
        raise ValueError(
            "Ollama returned an empty response."
        )

    questions = parse_questions(raw)

    if not questions:
        raise ValueError(
            "Could not create questions from "
            "the Ollama response."
        )

    return questions[:num_questions]


def parse_questions(text):

    # Remove markdown code blocks
    text = re.sub(
        r"```json",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"```",
        "",
        text,
    )

    text = text.strip()

    # --------------------------------------------
    # Try normal JSON
    # --------------------------------------------

    try:

        data = json.loads(text)

        if isinstance(data, list):
            return clean_questions(data)

    except json.JSONDecodeError:
        pass

    # --------------------------------------------
    # Find JSON array inside response
    # --------------------------------------------

    start = text.find("[")
    end = text.rfind("]")

    if start != -1 and end != -1:

        json_text = text[start:end + 1]

        try:

            data = json.loads(json_text)

            if isinstance(data, list):
                return clean_questions(data)

        except json.JSONDecodeError:
            pass

    return []


def clean_questions(data):

    questions = []

    for item in data:

        if not isinstance(item, dict):
            continue

        question = str(
            item.get("question", "")
        ).strip()

        answer = str(
            item.get("answer", "")
        ).strip()

        if not question or not answer:
            continue

        cleaned = {
            "question": question,
            "answer": answer,
            "explanation": str(
                item.get(
                    "explanation",
                    ""
                )
            ).strip(),
        }

        options = item.get(
            "options",
            []
        )

        if isinstance(options, list):

            cleaned["options"] = [
                str(option).strip()
                for option in options
                if str(option).strip()
            ]

        else:

            cleaned["options"] = []

        questions.append(cleaned)

    return questions
