import pandas as pd
import sqlite3 as sql
import psycopg2

def load_data(source: str, query: str):
    if source.endswith(".db"):
        conn = sql.connect(source)
    else:
        conn = psycopg2.connect(source)

    try:
        return pd.read_sql(query, conn)
    finally:
        conn.close()

conn = sql.connect("cleaned_lung_cancer.db")
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

print(tables)