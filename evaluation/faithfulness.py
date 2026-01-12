import re
from collections import Counter
from typing import List, Dict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Single shared embedder (important for performance)
_Embedder = SentenceTransformer("all-MiniLM-L6-v2")

STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "and", "in", "on", "for", "with",
    "by", "at", "from", "as", "that", "this", "it", "are", "was", "were",
    "be", "or", "if", "but", "not", "which", "can", "has", "have", "had"
}

def _normalize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS]

# ------------------------
# 1. Lexical Overlap
# -------------------------
def lexical_overlap(answer:str, context:str) -> float:
    answer_tokens = _normalize(answer)
    context_tokens = _normalize(context)

    if not answer_tokens:
        return 0.0

    overlap = Counter(answer_tokens) & Counter(context_tokens)

    return sum(overlap.values())/len(answer_tokens)


# ------------------------
# 2. Semantic Similarity
# -------------------------
def semantic_similarity(answer: str, context: str) -> float:
    embeddings = _Embedder.encode([answer, context])
    return float(
        cosine_similarity([embeddings[0], embeddings[1]])[0][0]
    )

# ------------------------
# 3. Citation Coverage (Chunk level grounding)
# -------------------------
def citation_coverage(answer: str, chunks: List[str]) -> float:
    sentences = re.split(r"[.!?]", answer)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return 0.0

    supported = 0

    for sentence in sentences:
        sent_emb = _Embedder.encode(sentence)
        chunk_embs = _Embedder.encode(chunks)

        sims = cosine_similarity(
            [sent_emb], chunk_embs
        )[0]

        if np.max(sims) >= 0.6: # grounding threshold
            supported += 1

    return supported/len(sentences)


# ------------------------
# 4. Hybrid faithfulness score
# -------------------------
def faithfulness_score(
        answer: str,
        context: str,
        chunks: List[str]
) -> Dict:

    lex = lexical_overlap(answer, context)
    sem = semantic_similarity(answer, context)
    cite = citation_coverage(answer, chunks)

    score = (
        0.3 * lex +
        0.4 * sem +
        0.3 * cite
    )

    if score >= 0.75:
        verdict = "faithful"

    elif score >=0.5:
        verdict = "weakly_grounded"
    else:
        verdict = "hallucinated"

    return {
        "faithfulness_score": round(score, 3),
        "lexical_overlap": round(lex, 3),
        "semantic_similarity": round(sem, 3),
        "citation_coverage": round(cite, 3),
        "verdict": verdict
    }


