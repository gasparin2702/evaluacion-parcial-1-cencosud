import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def preparar_sistema_hibrido():
    # 1. carga de fuentes internas (todos los .txt en data/)
    # usamos directoryloader para no cargar uno por uno
    loader_interno = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
    docs_internos = loader_interno.load()
    
    # 2. carga de fuente externa (direccion del trabajo - seccion consultas frecuentes)
    # esta es la pagina que sirve para nutrirse de normativas legales chilenas
    url_dt = "https://www.dt.gob.cl/portal/1628/w3-article-60141.html" 
    loader_externo = WebBaseLoader(url_dt)
    docs_externos = loader_externo.load()
    
    # fusionamos todos los documentos
    todos_los_docs = docs_internos + docs_externos
    
    # 3. procesamiento: division de texto (chunking) 
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(todos_los_docs)
    
    # 4. vectorizacion y almacenamiento 
    embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=os.getenv("GITHUB_TOKEN"),
    base_url=os.getenv("GITHUB_BASE_URL")
)
    
    print(f"sistema nutrido con {len(docs_internos)} archivos internos y 1 fuente legal externa")

if __name__ == "__main__":
    preparar_sistema_hibrido()



    # despues de docs_externos = loader_externo.load()
print("verificando fuente externa:")
print(f"url cargada: {docs_externos[0].metadata['source']}")
print(f"titulo de la pagina: {docs_externos[0].metadata.get('title', 'no hay titulo')}")
print(f"primeros 200 caracteres: {docs_externos[0].page_content[:200]}")
