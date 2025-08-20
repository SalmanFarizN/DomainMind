
# DomainMind: Local LLM with RAG on Domain-Specific Knowledgebase

DomainMind is a demonstration project showing how a local Large Language Model (LLM) can be enhanced with Retrieval-Augmented Generation (RAG) to answer questions using a custom, domain-specific knowledgebase. In this case, the knowledgebase consists of scientific articles and books from a PhD bibliography, enabling the LLM to provide informed responses grounded in specialized literature it was not originally trained on.

## Project Goals

- **Showcase RAG**: Demonstrate how RAG enables LLMs to access and utilize external, up-to-date, or proprietary knowledge.
- **Domain Adaptation**: Illustrate how LLMs can be tailored to specific fields (here, scientific research) without retraining the base model.
- **Local Operation**: Run everything locally for privacy, reproducibility, and independence from cloud APIs.

## How It Works

1. **Knowledgebase Construction**: Scientific articles and books are processed and stored in a vector database (ChromaDB).
2. **Document Retrieval**: When a user asks a question, relevant documents are retrieved from the knowledgebase using semantic search.
3. **LLM Augmentation**: Retrieved context is provided to the LLM, which generates answers grounded in the supplied material.

## Project Structure

- `main.py` — Entry point for running the demo.
- `src/` — Core modules:
	- `DataLoad.py`: Loads and processes documents.
	- `VectorDB.py`: Handles vector database operations.
	- `RAGChain.py`: Implements the RAG pipeline.
	- `LLMConfig.py`: LLM configuration and setup.
	- `Interface.py`: User interaction logic.
- `data/` — Contains the knowledgebase and raw files.
	- `raw/`: Source documents (articles, books).
	- `vector_db/`, `doc_vectordb/`: Vector database files.
- `notebooks/` — Jupyter notebooks for data processing and experimentation.
- `tests/` — Unit tests.

## Getting Started

### Prerequisites

- Python 3.11+
- [ChromaDB](https://www.trychroma.com/)
- [LangChain](https://python.langchain.com/)
- Local LLM (e.g., [Ollama](https://ollama.com/), [llama.cpp](https://github.com/ggerganov/llama.cpp), or similar)

Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the Demo

1. Place your domain-specific documents (PDFs, text files) in `data/raw/files/`.
2. Run the main script:

```bash
python main.py
```

3. Interact with the system by asking questions. The LLM will answer using information retrieved from your knowledgebase.

## Example Use Case

> **Q:** What are the main findings of Smith et al. (2020) regarding neural adaptation?
>
> **A:** [LLM responds with a summary based on the retrieved article from your bibliography.]

## Why Use RAG?

- **Extend LLM Knowledge**: Answer questions about topics not in the original training data.
- **Up-to-date Information**: Incorporate the latest research or proprietary data.
- **Transparency**: Provide sources for generated answers.

## Customization

- Swap in your own documents in `data/raw/files/`.
- Adjust retrieval and LLM settings in `src/LLMConfig.py` and `src/RAGChain.py`.

## References

- [Retrieval-Augmented Generation (RAG) Paper](https://arxiv.org/abs/2005.11401)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)

## License

This project is for demonstration and research purposes. See `LICENSE` for details.
