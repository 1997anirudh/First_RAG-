import chromadb
import ollama
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from flask import Flask, request, jsonify
import os

app = Flask(__name__)
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def build_index(filepath):
    with open(filepath, "r") as f:
        raw_text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    chunks = splitter.split_text(raw_text)

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="my_docs")
    collection.delete(where={"source": filepath})

    for i, chunk in enumerate(chunks):
        embedding = ollama.embeddings(model='nomic-embed-text', prompt=chunk)['embedding']
        collection.add(
            ids=[f"{filepath}_chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[{"source": filepath}]
        )
    print(f"✅ Indexed {len(chunks)} chunks from {filepath}")

def ask(question):
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="my_docs")

    question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']
    results = collection.query(query_embeddings=[question_embedding], n_results=3)
    context = "\n\n".join(results['documents'][0])
    print(f"DEBUG CONTEXT:\n{context}\n")

    prompt = f"""You are a helpful assistant. Answer using ONLY the context below.
If the answer is not in the context, say "I don't know."

Context:
{context}

Question: {question}
Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Build index on startup
build_index("sample.txt")

@app.route("/ask", methods=["POST"])
def ask_endpoint():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    answer = ask(question)
    return jsonify({"answer": answer})

@app.route("/")
def health():
    return jsonify({"status": "RAG app is running!"})


@app.route("/debug", methods=["POST"])
def debug_endpoint():
    data = request.json
    question = data.get("question", "")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="my_docs")

    question_embedding = ollama.embeddings(model='nomic-embed-text', prompt=question)['embedding']
    results = collection.query(query_embeddings=[question_embedding], n_results=3)

    return jsonify({"chunks": results['documents'][0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
