from src.MultiQueryGenChain import generate_query_chain
from src.RetrieverConfig import retriever
from src.OutputProcess import get_unique_union


# Retrieval Chain
retrieval_chain = generate_query_chain | retriever.map() | get_unique_union
