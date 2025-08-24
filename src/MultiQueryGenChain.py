import re
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.LLMConfig import llm
from src.OutputProcess import remove_think_blocks


# Prompt
template = """You are an AI scientific research assistant. Your task is to generate two 
different versions of the given user question and the conversation history so far to retrieve relevant documents from a vector 
database. The vector database consists of scientific papers, articles, books and other academic 
resources related to the field of Statistical Physics, Computational Physics, Statistics, Soft-Matter Physics
and other related fields. By generating multiple perspectives on the user question, your goal is to help
the user overcome some of the limitations of the distance-based similarity search. 
Provide these alternative questions separated by newlines. If the below question is unrelated or does not require
additional context, you can respond with "No relevant questions found."
Original question: {question}
Conversation history: {history}"""
multi_prompt = ChatPromptTemplate.from_template(template)

# Query generation cHAIN
generate_query_chain = (
    multi_prompt
    | llm
    | StrOutputParser()
    | remove_think_blocks
    | (lambda x: x.split("\n"))
)
