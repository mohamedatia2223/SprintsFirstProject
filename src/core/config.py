import os
from dotenv import load_dotenv

load_dotenv()

VECTOREDB_API_KEY = os.getenv("VECTOREDB_API_KEY")
VECTOREDB_ENDPOINT = os.getenv("VECTOREDB_ENDPOINT")
LLM_API_KEY = os.getenv("LLM_API_KEY")

PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gemini-2.5-flash")
SECONDARY_MODEL = os.getenv("SECONDARY_MODEL", "gemini-3.5-flash")
TERTIARY_MODEL = os.getenv("TERTIARY_MODEL", "gemini-3.1-flash-lite")

    