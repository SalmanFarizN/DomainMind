"""
This submodule defines the multi-query retrieval chain for document retrieval and aggregation.
"""
from DomainMind.MultiQueryGenChain import generate_query_chain
from DomainMind.RetrieverConfig import retriever
from DomainMind.OutputProcess import get_unique_union


# Retrieval Chain
retrieval_chain = generate_query_chain | retriever.map() | get_unique_union
