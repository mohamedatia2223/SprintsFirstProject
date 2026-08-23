import os
from sqlalchemy import create_engine

os.makedirs("data", exist_ok=True)

TEACHER_DB_PATH = os.path.join("data", "ai_teacher_vec.db")

teacher_engine = create_engine(f"sqlite:///{TEACHER_DB_PATH}")

def get_teacher_engine():
    return teacher_engine
