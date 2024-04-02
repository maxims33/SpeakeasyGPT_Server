import sqlite3

conn = sqlite3.connect("sql.db")
c = conn.cursor()

c.execute('''
CREATE TABLE IF NOT EXISTS AccountSettings
(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT, dateOfBirth TEXT, fullname TEXT, gender TEXT, orientation TEXT)
''')


c.execute("""
SELECT name FROM sqlite_master WHERE type='table'
""")

a = c.fetchall()
print(a)

conn.commit()
conn.close()
