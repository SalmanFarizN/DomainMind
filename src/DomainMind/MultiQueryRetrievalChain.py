from src.DomainMind.MultiQueryGenChain import generate_query_chain
from src.DomainMind.RetrieverConfig import retriever
from src.DomainMind.OutputProcess import get_unique_union


# Retrieval Chain
retrieval_chain = generate_query_chain | retriever.map() | get_unique_union
