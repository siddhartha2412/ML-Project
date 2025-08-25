# chat_with_project.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL.Image")

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os

# 1. Path to your entire project folder
project_path = "C:/Users/HP/OneDrive/Desktop/ML_Project"

# 2. Load all files (recursive=True -> picks from subfolders too)
documents = SimpleDirectoryReader(project_path, recursive=True).load_data()

# 3. Use HuggingFace embeddings (free, no API key)
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# 4. Create/load index
try:
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    index = load_index_from_storage(storage_context, embed_model=embed_model)
except:
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir="./storage")

# 5. Query engine
query_engine = index.as_query_engine(similarity_top_k=5)

# 6. Chat loop
while True:
    user_input = input("Ask about your project (or type 'exit'): ")
    if user_input.lower() == "exit":
        break
    response = query_engine.query(user_input)
    print("\n", response, "\n")
