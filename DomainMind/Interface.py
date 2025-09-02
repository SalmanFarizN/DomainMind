"""
This submodule provides Gradio interface functions for the DomainMind RAG QA chatbot,
including streaming and non-streaming response handlers.
"""
# interface.py
import gradio as gr
from langchain.schema import HumanMessage, AIMessage
from DomainMind.RAGChain import rag_chain_multi_query
from DomainMind.OutputProcess import split_think_and_answer, get_thought
from DomainMind.OutputProcess import remove_details_blocks


def rag_qa_multi_query_final(message, history):
    """
    Non-streaming version of the RAG QA function.

    This function processes a user message and conversation history to generate a response
    using the RAG chain. It converts the Gradio history format to LangChain format,
    invokes the RAG chain, extracts thought and answer parts, and returns the final response.

    Args:
        message (str): The user's input message.
        history (list): List of previous conversation turns in Gradio format.

    Returns:
        dict: A dictionary with 'role' and 'content' keys for the assistant's response.
    """
    # Convert Gradio history format to LangChain message format
    history_langchain_format = []
    # Convert Gradio history (list of {"role": ..., "content": ...}) to LangChain format
    for turn in history:
        if turn["role"] == "user":
            history_langchain_format.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            cleaned_content = remove_details_blocks(turn["content"])
            history_langchain_format.append(AIMessage(content=cleaned_content))

    try:
        # Pass both current message and history to the chain
        response = rag_chain_multi_query.invoke(
            {"history": history_langchain_format, "question": message}
        )
        thought = get_thought(response)
        final_answer = split_think_and_answer(response)

        if thought:
            # Add collapsible section with the <think> content
            final_answer = f"<details><summary><b>🤔 Thinking</b></summary><pre>{thought}</pre></details>\n\n{final_answer}"
        return {"role": "assistant", "content": final_answer}
    except Exception as e:
        return {"role": "assistant", "content": f"❌ Error: {str(e)}"}


def rag_qa_multi_query_streaming(message, history):
    """
    Streaming version of the RAG QA function.

    This function processes a user message and conversation history to generate a streamed response
    using the RAG chain. It converts the Gradio history format to LangChain format,
    streams the response from the RAG chain, processes <think> tags, and yields chunks of the response.

    Args:
        message (str): The user's input message.
        history (list): List of previous conversation turns in Gradio format.

    Yields:
        dict: A dictionary with 'role' and 'content' keys for each chunk of the assistant's response.
    """
    # Convert Gradio history format to LangChain message format
    history_langchain_format = []
    # Convert Gradio history (list of {"role": ..., "content": ...}) to LangChain format
    for turn in history:
        if turn["role"] == "user":
            history_langchain_format.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            cleaned_content = remove_details_blocks(turn["content"])
            history_langchain_format.append(AIMessage(content=cleaned_content))
    try:
        raw_output = ""
        processed_full_output = ""

        # Stream the response per token and process <think> tags
        # for thinking models
        for chunk in rag_chain_multi_query.stream(
            {"history": history, "question": message}
        ):
            raw_output += chunk

            # Check for <think> start
            if "<think>" in chunk:
                chunk = chunk.replace("<think>", "Reasoning:")

            if "</think>" in chunk:
                chunk = chunk.replace("</think>", "End of Reasoning\n")

            processed_full_output += chunk

            yield {
                "role": "assistant",
                "content": processed_full_output,
            }

    except Exception as e:
        return {"role": "assistant", "content": f"❌ Error: {str(e)}"}


def launch_interface():
    """
    Launches the Gradio interface for the DomainMind RAG QA chatbot.

    This function creates a Gradio ChatInterface using the streaming response handler,
    sets up the title and description, and launches the web application.
    """
    demo = gr.ChatInterface(
        fn=rag_qa_multi_query_streaming,
        type="messages",
        title="DomainMind",
        description="Powered by RAG + Qwen3:8B",
    )
    demo.launch(share=False, inbrowser=True)
