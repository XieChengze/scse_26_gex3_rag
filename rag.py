"""
Entry point for the RAG graded exercise.

Run:

    python rag.py
    python rag.py "Your own question about an IT policy?"

The single knowledge-base file is loaded exclusively through
policy_loader.load_policy_documents(); the agent answers through the RAG tool
defined in rag_tool.py and the reflection loop defined in agent.py.
"""

import sys

from policy_loader import load_policy_documents


documents = load_policy_documents()


print(f"Loaded {len(documents)} policy documents from the approved policy file.")



import rag_tool
from agent import MODEL_NAME, answer_with_reflection


# The exercise's target question: the knowledge base mentions contacting the
# IT Service Desk but never gives a phone number, so the correct answer is to
# state that the number is not known - never to invent one.
DEFAULT_QUESTION = (
    "What exact phone number should I call for the "
    "University IT Service Desk?"
)


def show_retrieval_preview(question: str) -> None:
    print("\nRETRIEVED EVIDENCE (top policy blocks):")
    results = rag_tool.vector_store.similarity_search(
        question, k=rag_tool.RETRIEVAL_K
    )
    for document in results:
        metadata = document.metadata
        print("\n--------------------")
        print(
            f"Source: {metadata.get('source')} | "
            f"{metadata.get('policy_id')} | "
            f"topic: {metadata.get('topic')}"
        )
        print(document.page_content)


def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_QUESTION

    print(f"\nUsing model: {MODEL_NAME}")

    print("\n" + "=" * 72)
    print("QUESTION:")
    print(question)
    print("=" * 72)

    show_retrieval_preview(question)

    final_answer = answer_with_reflection(question, verbose=True)

    print("\n" + "=" * 72)
    print("FINAL ANSWER:")
    print("=" * 72)
    print(final_answer)


if __name__ == "__main__":
    main()
