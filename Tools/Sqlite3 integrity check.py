#moved
import sqlite3
import os
import shutil

def check_integrity(db_path):
    """Check the integrity of the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        results = cursor.fetchall()
        conn.close()
        return results
    except sqlite3.Error as e:
        return [(str(e),)]

def repair_database(db_path):
    """Attempt to repair the SQLite database."""
    try:
        # Create a backup of the original database
        backup_path = db_path + '.bak'
        shutil.copyfile(db_path, backup_path)
        print(f"Backup created at {backup_path}")

        # Use the recover mode to attempt to recover the database
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
        cursor.execute("VACUUM INTO 'recovered.db';")
        conn.close()

        # Replace the original database with the recovered one
        os.remove(db_path)
        shutil.move('recovered.db', db_path)
        print("Database has been repaired and replaced with the recovered version.")
    except sqlite3.Error as e:
        print(f"An error occurred while attempting to repair the database: {e}")

def main():
    db_path = 'skyline_agi.db'
    if not os.path.exists(db_path):
        print(f"The database file {db_path} does not exist.")
        return

    print("Checking the integrity of the database...")
    integrity_results = check_integrity(db_path)

    if all(result[0] == 'ok' for result in integrity_results):
        print("The database integrity check passed. No issues found.")
    else:
        print("Integrity check found issues:")
        for result in integrity_results:
            print(result[0])

        repair = input("Do you want to attempt to repair the database? (yes/no): ").strip().lower()
        if repair == 'yes':
            repair_database(db_path)
        else:
            print("No repair attempted. Please manually address the issues.")

if __name__ == "__main__":
    main()
