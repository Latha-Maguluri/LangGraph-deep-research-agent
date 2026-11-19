from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite import SqliteSaver


def create_sqlite_checkpointer(path: str) -> SqliteSaver:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path, check_same_thread=False)
    serializer = JsonPlusSerializer(
        allowed_msgpack_modules=[
            ("app.models.research", "CriticEvaluation"),
            ("app.models.research", "ClaimAudit"),
            ("app.models.research", "EvidenceRecord"),
            ("app.models.research", "ResearchPlan"),
            ("app.models.research", "ResearchTask"),
        ]
    )
    checkpointer = SqliteSaver(connection, serde=serializer)
    checkpointer.setup()
    return checkpointer
