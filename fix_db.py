import sqlite3

conn = sqlite3.connect('facedb.sqlite')
conn.execute('''
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
''')

defaults = [
    ('access_control_enabled', 'false'),
    ('access_start_hour', '9'),
    ('access_end_hour', '18'),
    ('access_days', '1,2,3,4,5'),
]

for key, value in defaults:
    conn.execute(
        "INSERT OR IGNORE INTO settings (key, value) VALUES (?, ?)",
        (key, value)
    )

conn.commit()
conn.close()
print('Done!')