import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# configuracion para evitar advertencias en consola
os.environ["USER_AGENT"] = "asistente_cencosud_bot"

# cargar variables del archivo .env
load_dotenv()

def preparar_sistema_hibrido():
    # recuperar variables de entorno
    token = os.getenv("GITHUB_TOKEN")
    base_url = os.getenv("GITHUB_BASE_URL")
    
    if not token or not base_url:
        print("error: no se encuentran las llaves en el archivo .env")
        return

    print("iniciando proceso de ingesta...")

    # 1. carga de fuentes internas (.txt en carpeta data)
    try:
        loader_interno = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
        docs_internos = loader_interno.load()
    except Exception as e:
        print(f"error al cargar documentos internos: {e}")
        docs_internos = []
    
    # 2. carga de fuente externa (direccion del trabajo)
    try:
        url_dt = "https://www.dt.gob.cl/portal/1628/w3-article-60141.html" 
        loader_externo = WebBaseLoader(url_dt)
        docs_externos = loader_externo.load()
        
        # verificacion de carga externa
        if docs_externos:
            print("\nverificando fuente externa:")
            print(f"url cargada: {docs_externos[0].metadata['source']}")
            print(f"titulo: {docs_externos[0].metadata.get('title')}")
    except Exception as e:
        print(f"error al cargar fuente externa: {e}")
        docs_externos = []
    
    # fusion de documentos
    todos_los_docs = docs_internos + docs_externos
    
    if not todos_los_docs:
        print("error: no hay documentos para procesar.")
        return

    # 3. procesamiento de texto (chunking) 
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(todos_los_docs)
    
    # 4. vectorizacion y almacenamiento en faiss
    print("generando embeddings y base de datos vectorial...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=token,
        base_url=base_url
    )
    
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index_cencosud")
    
    print(f"\nsistema nutrido con exitos.")
    print(f"total de archivos internos: {len(docs_internos)}")
    print(f"total de fragmentos creados: {len(chunks)}")

if __name__ == "__main__":
    preparar_sistema_hibrido()
