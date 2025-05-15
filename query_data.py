import argparse
import re
from langchain_chroma import Chroma  # ✅ Updated Import
from langchain.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # ✅ Updated Import for Ollama
from get_embedding_function import get_embedding_function
from collections import OrderedDict
CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

# ✅ Preload the Ollama model (prevents reloading on every query)
ollama_model = OllamaLLM(model="mistral")

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    response_text = query_rag(query_text)
    
    # Print the response
    print(response_text)

def query_rag(query_text: str):
    # ✅ Prepare the DB with correct embeddings
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    
    # ✅ Perform similarity search in the Chroma DB
    search_results = db.similarity_search_with_score(query_text, k=10)
    
    # Extract the text from the search results (assuming search_results is a list of tuples (document, score))
    context_text = "\n".join([result[0].page_content for result in search_results])
    
    # ✅ Create the prompt
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    # ✅ Use the preloaded Ollama model
    response_text = ollama_model.invoke(prompt)
    
    return response_text

if __name__ == "__main__":
    main()
