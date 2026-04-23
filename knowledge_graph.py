import os

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_knowledge_base(filename):
    """Load knowledge base from a text file into a list of Q&A pairs."""
    qa_pairs = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            content = f.read()
            # Split by "Q: " to separate entries
            entries = content.split("Q: ")
            for entry in entries:
                if not entry.strip():
                    continue
                
                # Split into Question and Answer
                parts = entry.split("A: ")
                if len(parts) >= 2:
                    question = parts[0].strip()
                    answer = "A: ".join(parts[1:]).strip() # Rejoin in case A: appears in answer
                    qa_pairs.append({"question": question, "answer": answer})
                    
    except FileNotFoundError:
        print(f"Error: {filename} not found.")
    return qa_pairs

KNOWLEDGE_BASE = load_knowledge_base(os.path.join(_BASE_DIR, "knowledge_base.txt"))


