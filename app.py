import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Configuración inicial
os.environ["USER_AGENT"] = "asistente_cencosud"
load_dotenv()

def iniciar_chat():
    token = os.getenv("GITHUB_TOKEN")
    url = os.getenv("GITHUB_BASE_URL")
    
    if not token:
        print("Falta el GITHUB_TOKEN en el .env")
        return

    # Modelos y configuración
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        api_key=token,
        base_url=url
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=token,
        base_url=url
    )

    # Cargar la base de FAISS
    try:
        db = FAISS.load_local(
            "faiss_index_cencosud", 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 3})
    except Exception as e:
        print(f"Error cargando la DB (ejecuta ingesta.py primero): {e}")
        return

    # Definir el comportamiento del bot
    prompt_txt = """Eres el asistente de RRHH de Cencosud. 
    Usa solo la info del contexto para responder. Si no sale ahí, dile que hable con un humano.
    Si pregunta por lucas o sueldos, mándalo directo al portal 'Mi Cencosud'.

    Contexto: {context}
    Pregunta: {input}
    Respuesta:"""

    prompt = PromptTemplate.from_template(prompt_txt)

    # Armar la cadena RAG
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(retriever, doc_chain)

    print("\n--- Bot de RRHH Cencosud listo ---")
    print("(Escribe 'salir' para cerrar)\n")

    while True:
        user_input = input("Tú: ")
        
        if user_input.lower() in ['salir', 'exit']:
            break
        
        try:
            res = chain.invoke({"input": user_input})
            print(f"\nAsistente: {res['answer']}\n")
        except Exception as e:
            print(f"Error al responder: {e}")

if __name__ == "__main__":
    iniciar_chat()
