"""
Script to create a vector database from PDF documents.

This script loads PDF files from a specified directory, splits them into text chunks,
and creates a persistent vector database using the DomainMind modules.
It is used to preprocess documents for the RAG QA system.
"""

from pexpect import split_command_line
from DomainMind.DataLoad import PDFLoad
from DomainMind.VectorDB import VectorDB


pdf_loader = PDFLoad(dir="data/sample", chunk_size=1500, chunk_overlap=50)
split_docs = pdf_loader.load_split()

vector_db = VectorDB()
vector_db.create_vectorstore(split_docs, persist_directory="data/sampledoc_vectordb")
