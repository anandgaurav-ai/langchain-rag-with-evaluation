from typing import Dict

def compute_confidence(
        retrieval_score: float,
        faithfulness: Dict,
        answer_coverage: float | None = None
) -> Dict:
    """
    Composite confidence score
    Faithfulness is a hard gate
    """

    faith_score = faithfulness["faithfulness_score"]
    verdict = faithfulness["verdict"]

    # ---------------------------
    # 1. Hard Safety Gate
    # ---------------------------

    if verdict == "hallucinated":
        return {
            "confidence": 0.0
            "reason": "Low faithfulness",
            "verdict": "refuse"
        }

    # ---------------------------
    # 2. Optional coverage signal
    # ---------------------------
    coverage = answer_coverage if answer_coverage is not None else 0.5

    # ---------------------------
    # 3. Weighted Confidence
    # ---------------------------
    confidence = (
    0.4 * retrieval_score +
    0.4 * faith_score +
    0.2 * coverage
    )

    confidence = round(min(confidence, 1),3)

    return {
        "confidence": confidence,
        "faithfulness": faith_score,
        "retrieval_score": round(retrieval_score, 3),
        "verdict": "answer"
    }