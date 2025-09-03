"""
This submodule provides the VectorDB class for creating, loading, and managing vector databases and retrievers.
"""
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
import torch
from typing import Optional


class VectorDB:
    """Class to create and load vector databases and retrievers."""

    def __init__(self):
        """
        Initialize the VectorDB object.

        This method sets the vector database and embedding models that will be used.
        The vectorstore and retriever are set to None, and should be set using the
        create_vectorstore or load_vectorstore methods, and the create_retriever method
        respectively.
        """
        self.vectordb = Chroma
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="intfloat/e5-base-v2",
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.vectorstore: Optional[VectorStore] = None
        self.retriever: Optional[VectorStoreRetriever] = None

    def create_vectorstore(
        self, split_docs: list, persist_directory: Optional[str] = None
    ):
        """
        Create a vectorstore from a list of split documents.

        Args:
            split_docs (list): List of split documents.
            persist_directory (Optional[str], optional): Directory to persist the
                vectorstore in. Defaults to None.

        Returns:
            VectorStore: The created vector store.
        """
        if persist_directory:
            self.vectorstore = self.vectordb.from_documents(
                split_docs,
                self.embedding_model,
                persist_directory=persist_directory,
            )
        else:
            self.vectorstore = self.vectordb.from_documents(
                split_docs, self.embedding_model
            )

        return self.vectorstore

    def load_vectorstore(self, persist_directory: str):
        """
        Load a vectorstore from a persisted directory.

        Args:
            persist_directory (str): Path to the persisted vectorstore.

        Returns:
            VectorStore: The loaded vector store.
        """
        self.vectorstore = self.vectordb(
            persist_directory=persist_directory, embedding_function=self.embedding_model
        )
        return self.vectorstore

    def create_retriever(self, k: int = 3):
        """
        Create a retriever from the vectorstore.

        Args:
            k (int, optional): Number of documents to retrieve. Defaults to 3.

        Returns:
            Retriever: The created retriever.
        """
        self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": k})
        return self.retriever
