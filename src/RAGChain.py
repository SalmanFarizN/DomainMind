from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from src.LLMConfig import llm
from src.RetrieverConfig import retriever

# Prompt for RAG chain with history and context.
messages = [
    MessagesPlaceholder(variable_name="history"),
    SystemMessage(
        content="You are a helpful scientific assistant. Answer using only the context provided."
    ),
    HumanMessagePromptTemplate.from_template(
        "Context:\n{context}\n\nQuestion: {question}"
    ),
]

prompt = ChatPromptTemplate.from_messages(messages)


# RAG Chain
rag_chain = (
    {
        "history": RunnableLambda(lambda x: (x["history"])),
        "context": RunnableLambda(
            lambda x: retriever.get_relevant_documents(x["question"])
        ),
        "question": RunnableLambda(lambda x: (x["question"])),
    }
    | prompt
    | llm
    | StrOutputParser()
)
