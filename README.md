# DomainMind: Local LLM with RAG on Domain-Specific Knowledgebase

<img alt="Documentation" src="https://img.shields.io/badge/docs-GitHub Pages-blue">

DomainMind is a demonstration project showing how a local Large Language Model (LLM) can be enhanced with Retrieval-Augmented Generation (RAG) to answer questions using a custom, domain-specific knowledgebase. In this case, the knowledgebase consists of scientific articles and books from a PhD bibliography, enabling the LLM to provide informed responses grounded in specialized literature it was not originally trained on.

## 🎯 Project Goals

- **Showcase RAG**: Demonstrate how RAG enables LLMs to access and utilize external, up-to-date, or proprietary knowledge.
- **Domain Adaptation**: Illustrate how LLMs can be tailored to specific fields (here, scientific research) without retraining the base model.
- **Local Operation**: Run everything locally for privacy, reproducibility, and independence from cloud APIs.

## 🧠 How It Works

1. **Knowledgebase Construction**: Scientific articles and books are processed and stored in a vector database (ChromaDB).
2. **Document Retrieval**: When a user asks a question, relevant documents are retrieved from the knowledgebase using semantic search.
3. **LLM Augmentation**: Retrieved context is provided to the LLM, which generates answers grounded in the supplied material.

## Project Structure

- `main.py` — Entry point for running the demo.
- `create_vectordb.py` — Script to create the vector database from a directory containing documents.
- `DomainMind/` — Core modules:
- `data/` — Contains the knowledgebase and raw files.
	- `sample/`: Sample PDF. 
	- `sampledoc_vectordb/`: Sample vector database for testing.
- `notebooks/` — Jupyter notebooks for data processing and experimentation.
- `prompts/` — Text files containing prompt templates for the LLM.


## 🚀 Getting Started

### Prerequisites

- Python 3.13.5+
- [ChromaDB](https://www.trychroma.com/)
- [LangChain](https://python.langchain.com/)
- Local LLM (e.g., [Ollama](https://ollama.com/), [llama.cpp](https://github.com/ggerganov/llama.cpp), or similar)

Install dependencies:

```bash
# Using pip
pip install -r requirements.txt

# Using uv
uv sync 
```

### Creating a Vector Database

1. Clone the repository and place your directory containing domain-specific documents (PDFs, text files) in the `data/` directory. Subfolders within this directory are also supported.
2. Update the paths in `create_vectordb.py` to point to your document folder.
3. Run the script to create your own ChromaDB vector database:

```bash
python create_vectordb.py
```

### Configuring the Retriever

1. Update the path to your newly created vector database in `DomainMind\RetrieverConfig.py`.
2. This will allow you to create a RAG chain using your custom VectorDB.

### Running the Demo

1. Run the main script:

```bash
python main.py
```

2. Interact with the system by asking questions. The LLM will answer using information retrieved from your knowledgebase.

## 📚 Example Use Case

In this example, we have created a VectorDB of the book "Understanding Deep Learning" by Dr. Simon J. D. Prince ([https://udlbook.github.io/udlbook/](https://udlbook.github.io/udlbook/)).


## Why Use RAG?

- **Extend LLM Knowledge**: Answer questions about topics not in the original training data.
- **Up-to-date Information**: Incorporate the latest research or proprietary data.
- **Transparency**: Provide sources for generated answers.

## Customization

- Swap in your own documents in `data/`.
- Adjust retrieval and LLM settings in `DomainMind/LLMConfig.py` and `DomainMind/RAGChain.py`.

## References

- [Retrieval-Augmented Generation (RAG) Paper](https://arxiv.org/abs/2005.11401)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)

## License

This project is for demonstration and research purposes. 