"""
This submodule defines the multi-query generation chain for alternative question generation,
using prompt templates and output processing.
"""
import re
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from DomainMind.LLMConfig import llm
from DomainMind.OutputProcess import remove_think_blocks
from DomainMind.DataLoad import load_prompt


# Prompt
template = load_prompt("prompts/multiquerygen_v1.txt")

multi_prompt = ChatPromptTemplate.from_template(template)

# Query generation cHAIN
generate_query_chain = (
    multi_prompt
    | llm
    | StrOutputParser()
    | remove_think_blocks
    | (lambda x: x.split("\n"))
)
