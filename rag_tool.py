"""
RAG retrieval layer for the University IT support agent.

The single approved knowledge base (university_it_policies.txt) is loaded
exclusively through policy_loader.load_policy_documents(), embedded with the
Ollama model "nomic-embed-text" and stored in a local Chroma vector store.
The LangChain tool below is the only way the agent can access policy
information.
"""

import time
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.tools import tool

from policy_loader import load_policy_documents

documents = load_policy_documents()


print(f"Loaded {len(documents)} policy documents.")


# Embedding model served by the local Ollama installation.
EMBEDDING_MODEL = "nomic-embed-text"

COLLECTION_NAME = "university_it_policies"

PERSIST_DIRECTORY = Path(__file__).resolve().parent / ".chroma"

# Number of top-matching policy blocks returned for each question.
RETRIEVAL_K = 4


BATCH_SIZE = 32
MAX_RETRIES = 5


embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)


vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
    persist_directory=str(PERSIST_DIRECTORY),
)


# Index the documents once. The policy id is used as the Chroma document id so
# re-running this file upserts/resumes instead of creating duplicates.
already_indexed = set(vector_store._collection.get(include=[])["ids"])
to_index = [
    document
    for document in documents
    if document.metadata["policy_id"] not in already_indexed
]

print(
    f"{len(already_indexed)} documents already indexed; "
    f"{len(to_index)} remaining."
)

for start in range(0, len(to_index), BATCH_SIZE):
    batch = to_index[start : start + BATCH_SIZE]
    batch_ids = [document.metadata["policy_id"] for document in batch]

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            vector_store.add_documents(batch, ids=batch_ids)
            break
        except Exception as error:  # runner hiccup / temporary connection loss
            if attempt == MAX_RETRIES:
                raise
            wait_seconds = 3 * attempt
            print(
                f"Batch starting at {start} failed on attempt {attempt}: "
                f"{error}. Retrying in {wait_seconds}s ..."
            )
            time.sleep(wait_seconds)

    print(
        f"Indexed {min(start + BATCH_SIZE, len(to_index))}/{len(to_index)} "
        "remaining documents."
    )

print(
    f"Vector store ready with {vector_store._collection.count()} policy "
    f"embeddings ('{PERSIST_DIRECTORY}')."
)


@tool
def search_university_knowledge(question: str) -> str:
    """
    Search the official University IT policy knowledge base for information
    about university IT procedures, policies and instructions (accounts,
    Wi-Fi, VPN, email, printing and storage).

    Returns the most relevant policy blocks, each with its source, policy id
    and topic. The knowledge base contains ONLY the text returned here; it
    does not provide any contact detail that is not explicitly shown.
    """

    results = vector_store.similarity_search(question, k=RETRIEVAL_K)

    retrieved = []

    for document in results:
        metadata = document.metadata
        retrieved.append(
            f"SOURCE: {metadata.get('source')} | "
            f"POLICY ID: {metadata.get('policy_id')} | "
            f"TOPIC: {metadata.get('topic')}\n"
            f"{document.page_content}"
        )

    return "\n\n".join(retrieved)
