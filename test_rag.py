"""
STEP 2 TEST — RAG only (type questions, no microphone needed)
Run: python test_rag.py
Type a question and press Enter. Type 'exit' to quit.
"""
from utils_env import make_openai_client
from agent_agent import rag_search, _answer_from_rag, _enrich_query


def main():
    client = make_openai_client()
    print("\n=== Test RAG ===")
    print("Scrie o întrebare și apasă Enter. Scrie 'exit' pentru a ieși.\n")

    while True:
        try:
            query = input("Întrebare: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() == "exit":
            break

        enriched = _enrich_query(query)
        print(f"[RAG] Query îmbogățit: {enriched[:120]}...", flush=True)

        ctx, sources, ok = rag_search(enriched)

        if not ok:
            print("[RAG] Niciun rezultat relevant găsit în documentație.\n")
            continue

        print(f"[RAG] Surse găsite: {sources}")
        print("[RAG] Trimit la OpenAI...", flush=True)
        answer = _answer_from_rag(client, query, enriched)
        print(f"\nRăspuns: {answer}\n")

    print("Test RAG terminat.")


if __name__ == "__main__":
    main()
