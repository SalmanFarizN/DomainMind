from langchain_community.chat_models import ChatOllama
from src.VectorDB import VectorDB

llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", streaming=True)

# Load the VectorDB and create a retriever
vectordb = VectorDB()
vectordb.load_vectorstore(persist_directory="data/doc_vectordb")
retriever = vectordb.create_retriever()
