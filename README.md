# 🧠 LangChain RAG with Explicit Evaluation & Hallucination Control

This repository demonstrates how to build a **LangChain-based RAG system** while **retaining explicit control over evaluation, hallucination detection, confidence scoring, and refusal logic**.

Unlike typical LangChain demos, this project intentionally avoids end-to-end chains that hide decision-making logic.

---

## 🎯 Why this project exists

Most RAG examples:
- always answer
- hide retrieval and evaluation inside frameworks
- do not measure hallucinations
- do not refuse when uncertain

This project shows how to use Langchain without losing control**.

It builds on a prior **from-scratch RAG implementation** and answers a key engineering question:

> *How do we adopt a framework for speed without sacrificing safety and evaluation?*

---

## 🧩 Key Design Principles

- Frameworks are used **selectively**, not blindly
- Retrieval is delegated to LangChain abstractions
- Evaluation, confidence, and decisions remain **explicit**
- The system prefers **refusal over hallucination**
- Offline evaluation is separated from online inference

---