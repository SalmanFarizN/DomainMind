from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.DomainMind.LLMConfig import llm
from src.DomainMind.OutputProcess import remove_think_blocks
from src.DomainMind.DataLoad import load_prompt

# Prompt
template = load_prompt("prompts/summarygen_v1.txt")
summarization_prompt = ChatPromptTemplate.from_template(template)

# Summarization chain
summarization_chain = (
    summarization_prompt | llm | StrOutputParser() | remove_think_blocks
)
