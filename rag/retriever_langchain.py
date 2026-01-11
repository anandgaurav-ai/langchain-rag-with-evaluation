from pathlib import Path
from typing import List, Tuple

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

class LangChainRetriever:
    def __init__(self, data_dir: str = "data/raw_docs"):
        self.data_dir = Path(data_dir)
        self.vectorstore = None
        self._build_index()

    def _build_index(self):
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Data Directory not found: {self.data_dir}"
            )

        loader = DirectoryLoader(
            str(self.data_dir),
            glob="**/*.txt",
            loader_cls = TextLoader
        )

        documents: List[Document] = loader.load()

        if not documents:
            raise ValueError("No documents found for indexing.")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 300,
            chunk_overlap = 50
        )

        chunks: List[Document] = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name ="sentence-transformers/all-miniLM-L6-v2"
        )

        self.vectorstore = FAISS.from_documents(
            chunks,
            embedding = embeddings
        )

    def retrieve(
            self, query: str, top_k: int = 5,
    ) -> Tuple[str, float, List[str]]:
        if self.vectorstore is None:
            raise RuntimeError("Vector store is not initialized.")

        results = self.vectorstore.similarity_search_with_score(
            query, k = top_k
        )

        retrieved_chunks = []
        scores = []
        sources = []

        for doc, score in results:
            retrieved_chunks.append(doc.page_content)
            scores.append(score)
            sources.append(doc.metadata.get("source", "unknown"))

        context = "\n\n".join(retrieved_chunks)
        retrieval_score = 1/(1 + sum(scores) / len(scores))

        return context, retrieval_score, sources