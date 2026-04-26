import sqlite3

conn = sqlite3.connect('facedb.sqlite')
conn.executescript("""
    CREATE TABLE IF NOT EXISTS person_encodings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id INTEGER NOT NULL,
        photo_path TEXT,
        encoding BLOB NOT NULL,
        created_at TEXT DEFAULT (datetime('now')),
        FOREIGN KEY (person_id) REFERENCES persons(id)
    );
""")
conn.commit()
conn.close()
print('Database updated!')