import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def preparar_base_conocimiento():
    # cargamos el manual de cencosud 
    loader = TextLoader("data/reglamento_cencosud.txt")
    documentos = loader.load()
    
    # dividimos en fragmentos de 300 caracteres con solapamiento de 50 caracteres para mejorar la recuperación de información
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(documentos)
    
    # creamos la base vectorial con embeddings de openai y guardamos localmente
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_cencosud")
    print("base de datos creada")

if __name__ == "__main__":
    preparar_base_conocimiento()