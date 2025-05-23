# 🔍 Semantic Text Search with FAISS and OpenAI Embeddings

This project is a mini **vector search engine** built with **FAISS**, **OpenAI Embeddings**, and **Flask**. It demonstrates how to convert text into embeddings (vectors), index them using FAISS, and expose a REST API to search for semantically similar content — just like how search engines, chatbots, or recommendation systems work under the hood.

---

## 🧠 What is a Vector Database?

A **vector database** stores data in the form of **high-dimensional vectors**, which are numerical representations of real-world data like text, images, or audio. These vectors capture the **semantic meaning** of the data, enabling **similarity search** rather than just exact keyword matching.

In this project:
- Each document is embedded into a vector using OpenAI's `text-embedding-ada-002` model.
- These vectors are indexed with **FAISS** (Facebook AI Similarity Search).
- A **Flask API** allows querying the index with a new input and retrieves the most semantically similar results.

---

## 🔧 Features

- 🔗 Semantic text similarity using OpenAI embeddings
- ⚡ Fast vector search using FAISS
- 💾 FAISS index persisted to disk
- 📡 REST API built with Flask
- 🧠 In-memory document metadata mapping (can be extended to SQL/NoSQL)

---

## 🌐 Real-World Use Cases

Vector search is the backbone of many modern AI applications:

| Use Case                       | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| 🔍 **Semantic Search**        | Find documents, emails, or web pages by meaning instead of keywords         |
| 🤖 **Chatbots & RAG**         | Retrieve relevant knowledge chunks to help LLMs generate accurate responses |
| 🖼️ **Image Search**           | Search for similar images using embeddings from models like CLIP            |
| 🎵 **Music/Audio Matching**   | Match audio snippets or detect duplicates                                   |
| 📽️ **Video Retrieval**        | Find visually similar video frames or scenes                                |
| 📚 **Recommendation Systems** | Suggest products or content based on vector proximity                      |

---

## 📁 Project Structure
```
vector-search-app/
│
├── app.py ← Flask REST API
├── build_index.py ← Script to embed docs and build FAISS index
├── index/startup_index.faiss ← Saved FAISS index
├── metadata.json ← Document metadata (ID to text)
├── requirements.txt ← Python dependencies
```

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/arojit/vector-search-app.git
cd vector-search-app
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Your OpenAI API Key
Create a .env file
```bash
OPENAI_API_KEY=xxxxx-xxx-xxx-xxxx
```

### 4. Build the FAISS Index
```bash
python build_index.py
```

### 5. Run the Flask Server
```bash
python app.py
```

### 6. Make a Search Request
```bash
curl --location 'http://127.0.0.1:5000/search' \
--header 'Content-Type: application/json' \
--data '{
    "query": "How to start a startup?",
    "top_k": 3
}'
```

## 📦 Sample Response
```bash
{
  "query": "How to start a startup?",
  "results": [
    {
      "rank": 1,
      "document": "Starting a startup requires solving a real problem.",
      "distance": 0.24
    },
    {
      "rank": 2,
      "document": "You should validate your startup idea before building.",
      "distance": 0.28
    },
    {
      "rank": 3,
      "document": "Fundraising is crucial for a startup’s early growth.",
      "distance": 0.33
    }
  ]
}
```