"""Raw questionnaire score calculations."""


def calculate_fabq_scores(answers: dict[int, int]) -> dict[str, float]:
    """Calculate FABQ physical activity and work subscale scores."""
    pa = sum(answers.get(item, 0) for item in (2, 3, 4, 5))
    work = sum(answers.get(item, 0) for item in (6, 7, 9, 10, 11, 12, 15))
    return {"fabq_pa": float(pa), "fabq_w": float(work)}


def calculate_quickdash_score(answers: dict[int, int]) -> dict[str, float]:
    """Calculate QuickDASH score from valid responses."""
    valid_responses = [value for value in answers.values() if 1 <= value <= 5]
    if len(valid_responses) < 10:
        raise ValueError("QuickDASH requires at least 10 valid responses")

    score = ((sum(valid_responses) / len(valid_responses)) - 1) * 25
    return {"score": round(score, 2)}
