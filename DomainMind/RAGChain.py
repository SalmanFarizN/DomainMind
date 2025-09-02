"""
This submodule defines the RAG chain, combining multi-query retrieval, history summarization,
and prompt orchestration for question answering.
"""
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from DomainMind.LLMConfig import llm
from DomainMind.RetrieverConfig import retriever
from DomainMind.SummaryChain import summarization_chain
from DomainMind.MultiQueryRetrievalChain import retrieval_chain
from DomainMind.DataLoad import load_prompt


# Main Prompt 
main_prompt = load_prompt("prompts/main_v1.txt")

messages = [
    SystemMessage(
        content=main_prompt
    ),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}\n\n History: {history}"
    ),
]
prompt = ChatPromptTemplate.from_messages(messages)


# RAG chain with multi-query retrieval and history summarization
rag_chain_multi_query = (
    {
        "history": RunnableLambda(
            lambda x: summarization_chain.invoke({"history": x["history"]})
        ),
        "context": RunnableLambda(
            lambda x: retrieval_chain.invoke(
                {
                    "question": x["question"],
                    "history": summarization_chain.invoke({"history": x["history"]}),
                }
            )
        ),
        "question": RunnableLambda(lambda x: (x["question"])),
    }
    | prompt
    | llm
    | StrOutputParser()
)
