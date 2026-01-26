import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load documents
with open("campus_docs.txt", "r", encoding="utf-8") as f:
    documents = f.read().split("\n\n")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Create embeddings
doc_embeddings = embedder.encode(documents)

# FAISS index
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Load LLM
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_length=256
)

def get_rag_answer(query: str) -> str:
    # Embed query
    query_embedding = embedder.encode([query])

    # Retrieve top document
    _, indices = index.search(np.array(query_embedding), k=1)
    retrieved_text = documents[indices[0][0]]

    # Generate answer
    prompt = f"""
    Use the context below to answer the question.

    Context:
    {retrieved_text}

    Question:
    {query}
    """

    response = generator(prompt)[0]["generated_text"]
    return response
