from flask import Flask, request, jsonify
import faiss
import numpy as np
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

app = Flask(__name__)

# Load API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# Load FAISS index
index = faiss.read_index("index/startup_index.faiss")

# Load metadata
with open("metadata.json") as f:
    metadata = json.load(f)


def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


@app.route("/search", methods=["POST"])
def search():
    data = request.get_json()
    query = data.get("query")
    k = int(data.get("top_k", 3))

    if not query:
        return jsonify({"error": "Missing 'query'"}), 400

    query_vector = get_embedding(query)
    distances, indices = index.search(np.array([query_vector]), k)

    results = []
    for i, idx in enumerate(indices[0]):
        doc_id = str(idx)
        results.append({
            "rank": i + 1,
            "document": metadata.get(doc_id, ""),
            "distance": float(distances[0][i])
        })

    return jsonify({"query": query, "results": results})

if __name__ == "__main__":
    app.run(debug=True)
