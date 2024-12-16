# database_manager.py

import sqlite3
import os
from typing import List, Tuple, Any, Optional
from agi_config import AGIConfiguration

class DatabaseManager:
    def __init__(self, config_path: str = 'config.json'):
        self.config = AGIConfiguration(config_path)
        self.database_path = self.config.get_database_path()
        print(f"Database Path: {self.database_path}")  # Debug print
        self.connection = self.connect_to_database()

    def connect_to_database(self) -> Optional[sqlite3.Connection]:
        """
        Connects to the SQLite database and returns the connection object.

        :return: SQLite connection object or None if connection fails.
        """
        if not os.path.isfile(self.database_path):
            print(f"Error: Database file not found - {self.database_path}")
            return None
        try:
            connection = sqlite3.connect(self.database_path)
            return connection
        except sqlite3.Error as e:
            print(f"Error connecting to database: {e}")
            return None

    def get_table_names(self) -> List[str]:
        """
        Retrieves a list of table names from the SQLite database.

        :return: List of table names.
        """
        if self.connection is None:
            return []
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            return [table[0] for table in tables]
        except sqlite3.Error as e:
            print(f"Error retrieving table names: {e}")
            return []

    def load_domain_dataset(self, table_name: str) -> List[Tuple[Any, ...]]:
        """
        Loads a dataset from the specified table in the SQLite database.

        :param table_name: The name of the table to load data from.
        :return: List of tuples containing the dataset rows.
        """
        if self.connection is None:
            return []
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"SELECT * FROM {table_name};")
            dataset = cursor.fetchall()
            return dataset
        except sqlite3.Error as e:
            print(f"Error loading dataset from table {table_name}: {e}")
            return []

    def update_domain_data(self, table_name: str, data: Tuple[Any, ...]):
        """
        Updates a dataset in the specified table in the SQLite database.

        :param table_name: The name of the table to update data in.
        :param data: Tuple of data to insert.
        """
        if self.connection is None:
            return
        try:
            # Get column names for the table
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = cursor.fetchall()
            column_names = [column[1] for column in columns if column[1] != 'id']  # Exclude 'id' if it exists

            # Create placeholders for the values
            placeholders = ', '.join(['?'] * len(data))
            column_list = ', '.join(column_names)

            # Construct the SQL query
            query = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders});"
            cursor.execute(query, data)
            self.connection.commit()
        except sqlite3.Error as e:
            print(f"Error updating data in table {table_name}: {e}")

    def get_recent_updates(self) -> List[Tuple[str, Tuple[Any, ...]]]:
        """
        Retrieves recent updates from the SQLite database.

        :return: List of tuples containing table names and the most recent row.
        """
        if self.connection is None:
            return []
        recent_updates = []
        try:
            cursor = self.connection.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            for table in tables:
                table_name = table[0]
                if table_name == 'sqlite_sequence':
                    continue  # Skip the sqlite_sequence table
                try:
                    cursor.execute(f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1;")
                    recent_updates.append((table_name, cursor.fetchone()))
                except sqlite3.Error as e:
                    print(f"Error retrieving recent updates from table {table_name}: {e}")
            return recent_updates
        except sqlite3.Error as e:
            print(f"Error retrieving recent updates: {e}")
            return []

    def close_connection(self):
        """
        Closes the database connection.
        """
        if self.connection:
            self.connection.close()

# Example Usage
if __name__ == "__main__":
    db_manager = DatabaseManager()
    table_names = db_manager.get_table_names()
    print("Table Names:", table_names)
    
    # Example: Load a dataset from a specific table
    domain_data = db_manager.load_domain_dataset('knowledge_base')
    print("Domain Data (knowledge_base):", domain_data)
    
    # Example: Update a dataset in a specific table
    # Ensure the number of values matches the number of columns in the table (excluding 'id')
    new_data = ('science', 'new_topic', 'New description', 1500.0, '{"source": "New Source", "tags": ["New Tag"]}', '2024-12-16 06:00:00', 'medium')
    db_manager.update_domain_data('knowledge_base', new_data)
    
    # Example: Retrieve recent updates
    recent_updates = db_manager.get_recent_updates()
    print("Recent Updates:", recent_updates)
    
    db_manager.close_connection()
