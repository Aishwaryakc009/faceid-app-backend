import sqlite3
from pathlib import Path

DB_PATH = Path("facedb.sqlite")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS persons (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            label TEXT,
            photo_path TEXT,
            encoding BLOB NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS recognition_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            person_name TEXT,
            confidence REAL,
            source TEXT,
            timestamp TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (person_id) REFERENCES persons(id)
        );
    """)
    conn.commit()
    conn.close()