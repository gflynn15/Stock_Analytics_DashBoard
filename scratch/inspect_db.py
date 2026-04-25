import sqlite3

def get_schema(db_path, table_name):
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name});")
        columns = cursor.fetchall()
        print(f"Columns in {table_name}:")
        col_names = [col[1] for col in columns]
        print(col_names)
        
        print("\nSearching for 'sector' in column names...")
        if 'sector' in col_names:
            print("Found 'sector' column!")
        else:
            print("'sector' column not found.")
            
        print("\nChecking first few rows for values...")
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 3;")
        rows = cursor.fetchall()
        for row in rows:
            # Print only first 10 elements of each row to avoid clutter
            print(row[:10])
            
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    get_schema("Notebooks/STOCK_DATA_WAREHOUSE.db", "FUNDAMENTAL_DATA")
