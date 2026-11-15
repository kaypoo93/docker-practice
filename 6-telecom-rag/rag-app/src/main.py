"""
CLI entry point for the telecom RAG chatbot.
Usage: python main.py
"""

try:
    from dotenv import load_dotenv  # type: ignore[reportMissingImports]
except ImportError:  # pragma: no cover - optional dependency for local env files
    def load_dotenv(*args, **kwargs):
        return False

try:
    if __package__:
        from .rag_chain import build_chain  # type: ignore[reportMissingImports]
    else:
        from rag_chain import build_chain  # type: ignore[reportMissingImports]
except ImportError:  # Support running this file directly from the src directory.
    from rag_chain import build_chain  # type: ignore[reportMissingImports]

load_dotenv()


def main():
    print("=== Telecom Customer Care Chatbot (RAG) ===")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    chain = build_chain()

    while True:
        question = input("Customer: ").strip()
        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nAssistant: ", end="", flush=True)
        for chunk in chain.stream(question):
            print(chunk, end="", flush=True)
        print("\n")


if __name__ == "__main__":
    main()