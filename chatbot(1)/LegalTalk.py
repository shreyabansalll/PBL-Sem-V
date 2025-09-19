#!/usr/bin/env python
# coding: utf-8

# In[16]:


get_ipython().system('pip install sentence-transformers faiss-cpu pandas')


# In[18]:


import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Change these paths to where your files are saved
qa_path = r"/Users/shreya/Downloads/constitution_qa.json"   # Windows example
csv_path = r"/Users/shreya/Downloads/Text.csv"

# Load Constitution Q&A JSON
with open(qa_path, "r", encoding="utf-8") as f:
    qa_data = json.load(f)

# Load CSV dataset
csv_data = pd.read_csv(csv_path)


# In[19]:


knowledge_base = []

# Add Q&A entries
for item in qa_data:
    knowledge_base.append({
        "type": "qa",
        "text": item["question"],
        "answer": item["answer"]
    })

# Add context entries (from CSV)
contexts = csv_data["Text"].dropna().tolist()
for c in contexts:
    knowledge_base.append({
        "type": "context",
        "text": c,
        "answer": c  # fallback: return the text itself
    })

print("Total knowledge base size:", len(knowledge_base))


# In[20]:


from sentence_transformers import SentenceTransformer

# Load model (small, fast, good for FAQs)
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert to embeddings
texts = [item["text"] for item in knowledge_base]
embeddings = model.encode(texts, convert_to_numpy=True)

print("Embedding shape:", embeddings.shape)


# In[21]:


import faiss
import numpy as np

# Dimensions of embeddings
dim = embeddings.shape[1]

# Build FAISS index
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print("FAISS index built with", index.ntotal, "entries")


# In[22]:


def search_query(query, top_k=3):
    query_emb = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_emb, top_k)

    results = []
    for idx in indices[0]:
        item = knowledge_base[idx]
        results.append(item["answer"])
    return results


# In[23]:


query = "What is Article 21?"
answers = search_query(query, top_k=2)

print("User:", query)
print("\nBot:", answers[0], "\n\nDisclaimer: This is general info. Please consult a lawyer for advice.")


# In[ ]:


while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Bot: Goodbye!")
        break
    answers = search_query(query, top_k=1)
    print("Bot:", answers[0], "\nDisclaimer: This is general info. Please consult a lawyer.")


# In[ ]:




