import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

def iniciar_asistente():
    token = os.getenv("GITHUB_TOKEN")
    base_url = os.getenv("GITHUB_BASE_URL")

    # configuracion gpt-4o de github
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        base_url=base_url,
        api_key=token
    )

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=token,
        base_url=base_url
    )
    
    # carga de base de datos
    db = FAISS.load_local("faiss_index_cencosud", embeddings, allow_dangerous_deserialization=True)

    template = """eres el asistente de rr hh cencosud. usa solo el contexto para responder. 
    si no sabes, deriva a un humano. para sueldos usa 'mi cencosud'.
    contexto: {context}
    pregunta: {question}
    respuesta:"""

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )

    print("asistente activo. escribe 'salir' para cerrar.")
    while True:
        query = input("\nempleado: ")
        if query.lower() == 'salir': break
        res = qa_chain.invoke(query)
        print(f"asistente: {res['result']}")

if __name__ == "__main__":
    iniciar_asistente()
