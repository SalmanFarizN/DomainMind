"""
Main entry point for the DomainMind application.

This script initializes the environment and launches the Gradio interface for the RAG QA chatbot.
It loads environment variables for configuration and starts the interactive application.
"""

from DomainMind.Interface import launch_interface

# Enable LangSmith tracing
# Load environment variables from .env file to configure LangSmith 
# API keys and settings
from dotenv import load_dotenv
load_dotenv()


def main():
    """
    Main function to launch the DomainMind interface.

    This function calls the launch_interface function from the DomainMind.Interface module,
    which starts the Gradio web application for the RAG QA chatbot.
    """
    launch_interface()


if __name__ == "__main__":
    main()
