# interface.py
import gradio as gr
import re
from langchain.schema import HumanMessage, AIMessage
from src.RAGChain import rag_chain_multi_query
from src.OutputProcess import split_think_and_answer, get_thought
from src.OutputProcess import remove_details_blocks


def rag_qa_multi_query(message, history):
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


def launch_interface():
    demo = gr.ChatInterface(
        fn=rag_qa_multi_query,
        type="messages",
        title="DomainMind",
        description="Powered by RAG + Qwen3:8B",
    )
    demo.launch(share=False, inbrowser=True)
