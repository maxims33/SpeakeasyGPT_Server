import sqlite3

conn = sqlite3.connect("sql.db")
c = conn.cursor()

#c.execute('''
#DROP TABLE IF EXISTS test_table2
#''')

#c.execute('''
#CREATE TABLE IF NOT EXISTS AccountSettings
#(id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT, password TEXT, #dateOfBirth TEXT, fullname TEXT, gender TEXT, orientation TEXT)
#''')

#c.execute('''
#CREATE TABLE IF NOT EXISTS stocks
#(id INTEGER PRIMARY KEY AUTOINCREMENT, date text, trans text, symbol text, qty #real, price real)''')

#c.execute("INSERT INTO stocks (date, trans, symbol, qty, price) VALUES ('2006-01-05','BUY','RHAT',100,35.14)")
#c.execute("SELECT date, trans, symbol, qty, price from stocks")

c.execute("""SELECT * FROM AccountSettings""")

#c.execute("""
#SELECT name FROM sqlite_master WHERE type='table'
#""")

a = c.fetchall()
print(a)

conn.commit()
conn.close()

def getAccountSettings():
    conn = sqlite3.connect("sql.db")
    c = conn.cursor()
    c.execute(
        "SELECT username, password, fullname, dateOfBirth, orientation, gender from AccountSettings"
    )

    username, password, fullname, dateOfBirth, orientation, gender = c.fetchone(
    )
    respobj = AccountSettings(username=username,
                              password=password,
                              fullname=fullname,
                              dob=dateOfBirth,
                              orientation=orientation,
                              gender=gender)
    conn.close()
    return respobj

def setAccountSettings(request):
    conn = sqlite3.connect("sql.db")
    req = deserialize_AccountSettings(request)
    c = conn.cursor()
    c.execute(
        f"INSERT INTO AccountSettings (username, password, fullname, dateOfBirth, orientation, gender) VALUES ('{req.username}', '{req.password}', '{req.fullname}', '{req.dob}', '{req.orientation}', '{req.gender}')"
    )
    conn.commit()
    conn.close()