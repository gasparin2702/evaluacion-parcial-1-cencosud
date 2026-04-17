import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

def ejecutar_agente_cencosud():
    token = os.getenv("GITHUB_TOKEN")
    base_url = os.getenv("GITHUB_BASE_URL")

    # configuracion del modelo de lenguaje
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        base_url=base_url,
        api_key=token
    )

    # configuracion de embeddings (necesario para buscar en la base faiss)
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=token,
        base_url=base_url
    )
    
    # carga de la base de datos
    vector_store = FAISS.load_local(
        "faiss_index_cencosud", 
        embeddings, 
        allow_dangerous_deserialization=True
    )

    template = """eres el asistente virtual oficial de rr hh para colaboradores de cencosud retail. 
    responde solo con el contexto oficial. si preguntan por sueldos deriva al portal mi cencosud.

    contexto: {context}
    pregunta del trabajador: {question}
    respuesta:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    print("asistente cencosud listo. escribe 'salir' para terminar.")
    while True:
        p = input("\nconsulta: ")
        if p.lower() == 'salir': break
        res = chain.invoke(p)
        print(f"\nasistente cencosud: {res['result']}")

if __name__ == "__main__":
    ejecutar_agente_cencosud()
