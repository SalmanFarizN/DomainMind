# interface.py
import gradio as gr
import re
from langchain.schema import HumanMessage, AIMessage
from src.RAGChain import rag_chain


def split_think_and_answer(response):
    """
    Splits the response into the "thinking" part and the final answer.
    Args:
        response (str): The response string containing the thinking and answer parts.
    Returns:
        tuple: A tuple containing the thinking part and the final answer.
    """
    match = re.search(r"</think>\s*(.*)", response, re.DOTALL)
    return match.group(1).strip() if match else response.strip()


def get_thought(response):
    """
    Extracts the "thinking" part from the response.
    Args:
        response (str): The response string containing the thinking part.
    Returns:
        str: The extracted thinking part, or None if not found.
    """
    match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    return match.group(1).strip() if match else None


def rag_qa(message, history):
    history_langchain_format = []
    for turn in history:
        if turn["role"] == "user":
            history_langchain_format.append(HumanMessage(content=turn["content"]))
        elif turn["role"] == "assistant":
            history_langchain_format.append(AIMessage(content=turn["content"]))

    history_langchain_format.append(HumanMessage(content=message))

    try:
        response = rag_chain.invoke(
            {"history": history_langchain_format, "question": message}
        )
        thought = get_thought(response)
        final_answer = split_think_and_answer(response)
        if thought:
            final_answer = f"<details><summary><b>🤔 Thinking</b></summary><pre>{thought}</pre></details>\n\n{final_answer}"
        return {"role": "assistant", "content": final_answer}
    except Exception as e:
        return {"role": "assistant", "content": f"❌ Error: {str(e)}"}


def launch_interface():
    demo = gr.ChatInterface(
        fn=rag_qa,
        type="messages",
        title="📄 Scientific PDF Chatbot",
        description="Ask questions about your scientific PDFs. Powered by RAG + Qwen3:8B",
        examples=["What are colloidal particles?", "Tell me more about that"],
    )
    demo.launch(share=False, inbrowser=True)
