from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.LLMConfig import llm
from src.RetrieverConfig import retriever
from src.SummaryChain import summarization_chain
from src.MultiQueryRetrievalChain import retrieval_chain


# Prompt template for RAG with multi-query and history summarization
messages = [
    SystemMessage(
        content="""
        You are a helpful scientific assistant. You need to answer the question in a scientifically sound manner, \\
        combining your own knowledge and the provided context and the conversation history. If you use information \\
        from the provided context, cite it. If the answer is based on your own knowledge, state so. The conversation \\
        history is provided above. You also have further context regarding the current question below from snippets \\
        from various scientific papers. 
        """
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
