import os
from dotenv import load_dotenv

load_dotenv()

VECTOREDB_API_KEY = os.getenv("VECTOREDB_API_KEY")
VECTOREDB_ENDPOINT = os.getenv("VECTOREDB_ENDPOINT")
LLM_API_KEY = os.getenv("LLM_API_KEY")

    