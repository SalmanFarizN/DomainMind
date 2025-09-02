from pexpect import split_command_line
from src.DomainMind.DataLoad import PDFLoad
from src.DomainMind.VectorDB import VectorDB


pdf_loader = PDFLoad(dir="data/sample", chunk_size=1500, chunk_overlap=50)
split_docs = pdf_loader.load_split()

vector_db = VectorDB()
vector_db.create_vectorstore(split_docs, persist_directory="data/sampledoc_vectordb")
