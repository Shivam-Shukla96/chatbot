import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    CHROMA_DB_DIR: str = os.getenv("CHROMA_DB_DIR", "./chroma_db")

settings = Settings()
