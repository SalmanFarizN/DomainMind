from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os


class PDFLoad:
    """Class to load PDF files and split them into chunks"""

    def __init__(self, dir: str, chunk_size: int = 1500, chunk_overlap: int = 50):
        """
        Initialize the PDFLoad object.

        Args:
            dir (str): The directory from which to load PDF files.
            chunk_size (int, optional): The number of characters in each chunk. Defaults to 1500.
            chunk_overlap (int, optional): The number of overlapping characters between chunks. Defaults to 50.
        """
        self.dir = dir
        self.pdf_loader = PyMuPDFLoader
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_split(self) -> list:
        """
        Walks through the given directory, loads PDF files with PyMuPDFLoader, splits
        the text into chunks with RecursiveCharacterTextSplitter, and returns the list
        of chunks.

        Returns:
            list: A list of strings, where each string is a chunk of text from the PDFs.
        """
        split_docs = []
        file_count = 0

        for dirpath, dirnames, filenames in os.walk(self.dir):
            for filename in filenames:
                if filename.lower().endswith(".pdf"):
                    pdf_path = os.path.join(dirpath, filename)
                    loader = self.pdf_loader(pdf_path)
                    doc = loader.load()
                    chunks = self.text_splitter.split_documents(doc)
                    split_docs.extend(chunks)
                    file_count += 1

        print(f"Total PDF files found: {file_count}")

        return split_docs
