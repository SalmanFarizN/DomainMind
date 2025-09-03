"""
This submodule configures the local LLM for use in the DomainMind pipeline.
"""
from langchain_community.chat_models import ChatOllama

llm = ChatOllama(model="qwen3:8b", base_url="http://localhost:11434", streaming=True)
