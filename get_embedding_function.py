#from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
# def get_embedding_function():
#     return OllamaEmbeddings(model="nomic-embed-text")
def get_embedding_function():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
