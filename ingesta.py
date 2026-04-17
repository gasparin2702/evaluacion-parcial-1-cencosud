import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

os.environ["USER_AGENT"] = "bot_cencosud_ep1"
load_dotenv()

def preparar_sistema():
    token = os.getenv("GITHUB_TOKEN")
    base_url = os.getenv("GITHUB_BASE_URL")
    
    if not token or not base_url:
        print("error: no hay llaves en el .env. revisa el archivo.")
        return

    # cargando manuales de la carpeta data/
    print("cargando archivos internos...")
    loader_int = DirectoryLoader("data/", glob="*.txt", loader_cls=TextLoader)
    docs_int = loader_int.load()
    
    # cargando normativa externa
    print("conectando con direccion del trabajo...")
    url_dt = "https://www.dt.gob.cl/portal/1628/w3-article-60141.html"
    loader_ext = WebBaseLoader(url_dt)
    docs_ext = loader_ext.load()
    
    # procesamiento rag
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(docs_int + docs_ext)
    
    # embeddings con github models
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=token,
        base_url=base_url
    )
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local("faiss_index_cencosud")
    print("\n[exito] base de datos creada. lista para app.py")

if __name__ == "__main__":
    preparar_sistema()
