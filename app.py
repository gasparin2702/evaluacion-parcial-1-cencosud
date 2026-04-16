import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

def ejecutar_agente_cencosud():
    # conectamos con github models api para usar gpt-4o y text-embedding-3-small
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1,
        base_url=os.getenv("GITHUB_BASE_URL"),
        api_key=os.getenv("GITHUB_TOKEN")
    )

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.load_local("faiss_index_cencosud", embeddings, allow_dangerous_deserialization=True)

    # aplicamos el prompt con rol y restricciones para que el asistente solo responda con información del reglamento y derive a mi cencosud para consultas de sueldos
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

    while True:
        p = input("\nconsulta (o 'salir'): ")
        if p.lower() == 'salir': break
        res = chain.invoke(p)
        print(f"asistente cencosud: {res['result']}")

if __name__ == "__main__":
    ejecutar_agente_cencosud()