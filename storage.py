from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

DB_PATH = "cardiodesk.db"

serde = JsonPlusSerializer(allowed_msgpack_modules=[("schemas", "MessageAnalysis")])

conn = sqlite3.connect(DB_PATH, check_same_thread=False)

checkpointer = SqliteSaver(conn = conn, serde=serde)

def config_for(case_id):
    return {"configurable": {"thread_id": f"case-{case_id}"}}
