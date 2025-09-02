from src.DomainMind.Interface import launch_interface

# Enable LangSmith tracing
# Load environment variables from .env file to configure LangSmith 
# API keys and settings
from dotenv import load_dotenv
load_dotenv()


def main():
    launch_interface()


if __name__ == "__main__":
    main()
