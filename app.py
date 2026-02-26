import chromadb
import ollama
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

# Initialize Groq client
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def build_index(filepath):
    """Load a file, chunk it, embed it, store in ChromaDB."""
    with open(filepath, "r") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_text(raw_text)

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="my_docs")

    collection.delete(where={"source": filepath})

    for i, chunk in enumerate(chunks):
        # Still using Ollama for embeddings (local)
        embedding = ollama.embeddings(model='nomic-embed-text', prompt=chunk)['embedding']
        collection.add(
            ids=[f"{filepath}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filepath}]
        )
    print(f"✅ Indexed {len(chunks)} chunks from {filepath}\n")

def ask(question):
    """Retrieve relevant chunks and generate an answer."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="my_docs")

    # Embed question using Ollama (local)
    question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']
    results = collection.query(query_embeddings=[question_embedding], n_results=2)
    context = "\n\n".join(results['documents'][0])

    # Build prompt
    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

    # 🆕 Groq instead of Ollama for LLM
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


if __name__ == "__main__":
    build_index("sample.txt")
    print("🤖 RAG App ready! Type 'quit' to exit.\n")
    while True:
        question = input("You: ")
        if question.lower() == 'quit':
            break
        answer = ask(question)
        print(f"\nAssistant: {answer}\n")
