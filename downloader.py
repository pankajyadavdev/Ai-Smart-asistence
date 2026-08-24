from io import BytesIO


def create_questions_text(
    questions,
):
    """
    Create a question paper without answers.
    """

    lines = [
        "AI GENERATED EXAM QUESTIONS",
        "=" * 50,
        "",
    ]

    for index, question in enumerate(
        questions,
        start=1,
    ):

        lines.append(
            f"{index}. "
            f"{question.get('question', '')}"
        )

        options = question.get(
            "options",
            [],
        )

        for option_index, option in enumerate(
            options
        ):

            letter = chr(
                65 + option_index
            )

            lines.append(
                f"   {letter}. {option}"
            )

        lines.append("")

    return "\n".join(lines)


def create_answer_key(
    questions,
):
    """
    Create an answer key.
    """

    lines = [
        "ANSWER KEY",
        "=" * 50,
        "",
    ]

    for index, question in enumerate(
        questions,
        start=1,
    ):

        lines.append(
            f"{index}. "
            f"{question.get('answer', '')}"
        )

    return "\n".join(lines)


def create_questions_and_answers(
    questions,
):
    """
    Create questions, answers,
    explanations and sources.
    """

    lines = [
        "EXAM QUESTIONS AND ANSWERS",
        "=" * 50,
        "",
    ]

    for index, question in enumerate(
        questions,
        start=1,
    ):

        lines.append(
            f"{index}. "
            f"{question.get('question', '')}"
        )

        options = question.get(
            "options",
            [],
        )

        for option_index, option in enumerate(
            options
        ):

            letter = chr(
                65 + option_index
            )

            lines.append(
                f"   {letter}. {option}"
            )

        lines.append(
            f"Answer: "
            f"{question.get('answer', '')}"
        )

        lines.append(
            f"Explanation: "
            f"{question.get('explanation', '')}"
        )

        source = question.get(
            "source",
            "",
        )

        page = question.get(
            "page",
            "",
        )

        if source:
            lines.append(
                f"Source: {source} "
                f"- Page {page}"
            )

        lines.append("")

    return "\n".join(lines)


def text_to_bytes(text):
    """
    Convert text into downloadable bytes.
    """

    return BytesIO(
        text.encode("utf-8")
    )
