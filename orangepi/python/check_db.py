import sqlite3

def check_schema(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(Persons)")
    columns = cursor.fetchall()
    print(f"Schema của Persons trong {db_path}:")
    for col in columns:
        print(f"Column: {col[1]}, Type: {col[2]}")
    conn.close()

check_schema('temp.db')
check_schema('database.db')