"""
This submodule configures and initializes the retriever for vector database queries.
"""
from DomainMind.VectorDB import VectorDB

# Load the VectorDB and create a retriever
vectordb = VectorDB()
vectordb.load_vectorstore(persist_directory="data/doc_vectordb")
retriever = vectordb.create_retriever()