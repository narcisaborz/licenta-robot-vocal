from agent_agent import _get_rag_db
db = _get_rag_db()
print(db._collection.metadata)