from config import SEVERITY_WEIGHTS

def combine_severity_scores(scores: dict) -> float:
    print("Calculating severity score")
    if not scores:
        return 0.0

    weighted_sum = 0.0
    total_weight = 0.0

    for model_name, score in scores.items():
        weight = SEVERITY_WEIGHTS.get(model_name, 1.0)  # default weight=1 if not specified
        weighted_sum += score * weight
        total_weight += weight

    return round(weighted_sum / total_weight, 3)
