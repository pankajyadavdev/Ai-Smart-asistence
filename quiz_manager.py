def calculate_score(questions, user_answers):

    correct = 0
    wrong = 0
    not_attempted = 0

    results = []

    for i, question in enumerate(questions):

        user_answer = user_answers.get(i)

        # --------------------------------------------
        # NOT ATTEMPTED
        # --------------------------------------------

        if user_answer is None:
            user_answer = ""

        user_answer = str(user_answer).strip()

        if not user_answer:
            status = "not_attempted"
            not_attempted += 1
            is_correct = False

        else:

            correct_answer = question.get(
                "answer",
                ""
            )

            if correct_answer is None:
                correct_answer = ""

            correct_answer = str(
                correct_answer
            ).strip()

            # ----------------------------------------
            # CHECK ANSWER
            # ----------------------------------------

            is_correct = (
                user_answer.lower()
                == correct_answer.lower()
            )

            if is_correct:
                status = "correct"
                correct += 1
            else:
                status = "wrong"
                wrong += 1

        # --------------------------------------------
        # SAVE RESULT
        # --------------------------------------------

        results.append(
            {
                "question": question.get(
                    "question",
                    ""
                ),

                "user_answer": (
                    user_answer
                    if user_answer
                    else "Not Attempted"
                ),

                "correct_answer": question.get(
                    "answer",
                    ""
                ),

                "is_correct": is_correct,

                "status": status,
            }
        )

    # --------------------------------------------
    # TOTAL
    # --------------------------------------------

    total = len(questions)

    percentage = (
        (correct / total) * 100
        if total > 0
        else 0
    )

    return {
        "correct": correct,
        "wrong": wrong,
        "not_attempted": not_attempted,
        "total": total,
        "percentage": round(
            percentage,
            2
        ),
        "results": results,
    }
