Este proyecto corresponde a un asistente de RRHH basado en inteligencia artificial para Cencosud, utilizando un enfoque RAG (Retrieval-Augmented Generation). El sistema permite responder preguntas utilizando información interna de la empresa y normativa externa, combinando embeddings y un modelo de lenguaje.

Para poder ejecutar el proyecto correctamente, es importante seguir los pasos de instalación en orden.

Primero, se debe tener instalado Python en versión 3.10 o 3.11. Versiones más recientes como 3.13 pueden generar errores debido a incompatibilidades con algunas librerías.

Una vez instalado Python, se debe clonar el repositorio o descargar los archivos del proyecto en el computador.

Luego, abrir una terminal en la carpeta del proyecto y crear un entorno virtual con el siguiente comando:

python -m venv venv

Después, activar el entorno virtual. En Windows se hace con:

venv\Scripts\activate

Una vez activado, se deben instalar las dependencias necesarias ejecutando:

pip install -r requirements.txt

Es importante asegurarse de que todas las dependencias se instalen sin errores antes de continuar.

El proyecto utiliza variables de entorno para la conexión con el modelo, por lo que se debe crear un archivo llamado .env en la raíz del proyecto. Este archivo debe contener las siguientes variables:

GITHUB_TOKEN=tu_token_aqui
GITHUB_BASE_URL=tu_url_aqui

(estos valores dependen del servicio que se esté utilizando para acceder al modelo)

Antes de ejecutar el asistente, es necesario generar la base de datos vectorial. Para esto se ejecuta:

python ingesta.py

Este paso carga los documentos desde la carpeta data y desde una fuente web externa, los procesa y crea un índice FAISS que será utilizado por el sistema.

Si todo funciona correctamente, aparecerá un mensaje indicando que la base de datos fue creada con éxito.

Finalmente, para iniciar el asistente, se ejecuta:

python app.py

El sistema permitirá ingresar preguntas por consola y responderá utilizando la información disponible en el contexto. Para salir, se puede escribir “salir”.
