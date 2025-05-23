from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np
import json
import os

# Load API Key
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

documents = [
    "Starting a startup requires solving a real problem.",
    "Learn to cook pasta with this easy recipe.",
    "Fundraising is crucial for a startup's early growth.",
    "You should validate your startup idea before building.",
    "Python is a popular programming language.",
    "Hiring the right team is key to startup success.",
]

def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding, dtype=np.float32)

# Create index
document_embeddings = np.array([get_embedding(doc) for doc in documents])
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

# Save index
os.makedirs("index", exist_ok=True)
faiss.write_index(index, "index/startup_index.faiss")

# Save metadata (doc ID to content)
metadata = {str(i): doc for i, doc in enumerate(documents)}
with open("metadata.json", "w") as f:
    json.dump(metadata, f)

print("✅ FAISS index and metadata saved.")
