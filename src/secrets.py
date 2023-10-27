import os

from dotenv import load_dotenv


def secrets() -> dict:
    load_dotenv()
    return {
        "openai_key": os.getenv('OPENAI_API_KEY'),
    }
