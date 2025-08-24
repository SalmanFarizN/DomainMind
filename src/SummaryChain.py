from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.LLMConfig import llm
from src.OutputProcess import remove_think_blocks

# Prompt
template = """
You are an AI scientific research assistant. Summarize the following conversation between a user and an assistant.
- Focus on the key points, main ideas, and any important questions and answers.
- Exclude irrelevant, repetitive, or off-topic content.
- Use your own words; do not copy the conversation verbatim.
- Keep the summary concise and within 512-1024 tokens.
- Format the summary as a narration of how the conversation between the user and the assistant unfolded.

Original conversation:
{history}
"""
summarization_prompt = ChatPromptTemplate.from_template(template)

# Summarization chain
summarization_chain = (
    summarization_prompt | llm | StrOutputParser() | remove_think_blocks
)
