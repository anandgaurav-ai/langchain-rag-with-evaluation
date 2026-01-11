import json
from pathlib import Path

from rag.retriever_langchain import LangChainRetriever
from evaluation.retrieval_metrics import recall_at_k


def load_eval_queries(path:Path):
    """
    Loads evaluation queries from a JSONL file.
    Each line must contain:
    - query
    - relevant doc
    """
    queries = []
    with path.open("r", encoding = "utf-8") as f:
        for line in f:
            queries.append(json.loads(line))
    return queries

def main():
    eval_file = Path("data/eval_queries.jsonl")

    if not eval_file.exist():
        raise FileNotFoundError(
            "data/eval_queries.jsonl not found"
        )

    # 1. Load evaluation data
    eval_queries = load_eval_queries(eval_file)

    # 2. Initialize Retriever
    retriever = LangChainRetriever()

    # 3. Metrics Storage
    recall_at_1 = []
    recall_at_3 = []
    recall_at_5 = []

    # 4. Run evaluation
    for item in eval_queries:
        query = item["query"]
        expected_doc = item["relevant_doc"]

        _, _, retrieved_sources = retriever.retrieve(query, top_k = 5)

        recall_at_1.append(
            recall_at_k(retrieved_sources, expected_doc, k=1)
        )

        recall_at_3.append(
            recall_at_k(retrieved_sources, expected_doc, k=3)
        )

        recall_at_5.append(
            recall_at_k(retrieved_sources, expected_doc, k=5)
        )

# 5. Aggregate results
def avg(values):
    return sum(values) / len(values) if values else 0.0

print("\n Retrieval Evaluation results")
print("----------------------------")
print(f"Recall@1: {avg(recall_at_1):.2f}")
print(f"Recall@3: {avg(recall_at_3):.2f}")
print(f"Recall#5: {avg(recall_at_5):.2f}")
print("-----------------------------")

if __name__ =="__main__":
    main()