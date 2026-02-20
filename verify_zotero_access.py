import sqlite3
import os

def check_zotero_access():
    db_path = os.environ.get("ZOTERO_DB_PATH", os.path.expanduser("~/Zotero/zotero.sqlite"))
    if not os.path.exists(db_path):
        print("Zotero database not found.")
        return False
    try:
        conn = sqlite3.connect(db_path, timeout=5)
        conn.execute("SELECT 1 FROM items LIMIT 1")
        conn.close()
        print("Zotero database accessible.")
        return True
    except sqlite3.OperationalError as e:
        print(f"Zotero database locked or error: {e}")
        return False

if __name__ == "__main__":
    check_zotero_access()