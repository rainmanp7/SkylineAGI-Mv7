import sqlite3
import os
from typing import List, Any, Optional
from agi_config import AGIConfiguration
import logging

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseManager:
    def __init__(self, config_path: str = 'config.json'):
        """
        Initializes the DatabaseManager with the configuration file path.

        :param config_path: Path to the configuration file (default: 'config.json').
        """
        self.config = AGIConfiguration(config_path)
        self.database_path = self.config.get_database_path()
        logging.info(f"Database Path: {self.database_path}")  # Debug log
        self.connection = self.connect_to_database()

    def connect_to_database(self) -> Optional[sqlite3.Connection]:
        """
        Connects to the SQLite database and returns the connection object.

        :return: SQLite connection object or None if connection fails.
        """
        if not os.path.isfile(self.database_path):
            logging.error(f"Error: Database file not found - {self.database_path}")
            return None
        try:
            connection = sqlite3.connect(self.database_path)
            logging.info("Successfully connected to the database.")
            return connection
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")
            return None

    def execute_query(self, query: str, params: tuple = ()) -> Optional[List[tuple]]:
        """
        Executes a query against the database and optionally fetches results.

        :param query: The SQL query to execute.
        :param params: Parameters to substitute into the query (default: empty tuple).
        :return: List of results for SELECT queries, or None for others.
        """
        if self.connection is None:
            logging.error("Database connection is not initialized.")
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            if query.strip().lower().startswith("select"):
                results = cursor.fetchall()
                logging.debug(f"Query Results: {results}")
                return results
            else:
                self.connection.commit()
                logging.info("Query executed successfully.")
                return None
        except sqlite3.Error as e:
            logging.error(f"Error executing query: {e}")
            return None

    def get_table_names(self) -> List[str]:
        """
        Retrieves a list of table names from the SQLite database.

        :return: List of table names.
        """
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        result = self.execute_query(query)
        return [table[0] for table in result] if result else []

    def load_domain_dataset(self, table_name: str) -> List[tuple]:
        """
        Loads a dataset from the specified table in the SQLite database.

        :param table_name: The name of the table to load data from.
        :return: List of tuples containing the dataset rows.
        """
        query = f"SELECT * FROM {table_name};"
        return self.execute_query(query) or []

    def update_domain_data(self, table_name: str, data: tuple):
        """
        Inserts a new row into the specified table in the SQLite database, excluding the 'id' column.

        :param table_name: The name of the table to insert data into.
        :param data: Tuple of data to insert, excluding the 'id' column.
        """
        if self.connection is None:
            logging.error("Database connection is not initialized.")
            return
        try:
            cursor = self.connection.cursor()
            # Retrieve column names for the table, excluding the 'id' column
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [column[1] for column in cursor.fetchall() if column[1] != 'id']

            # Ensure the data tuple matches the number of columns (excluding 'id')
            if len(data) != len(columns):
                logging.error(f"Data tuple length mismatch for table {table_name}. Expected {len(columns)} columns, got {len(data)}.")
                return

            # Create placeholders and construct the query
            placeholders = ', '.join(['?'] * len(data))
            column_list = ', '.join(columns)
            query = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders});"
            cursor.execute(query, data)
            self.connection.commit()
            logging.info(f"Data inserted into table {table_name}.")
        except sqlite3.Error as e:
            logging.error(f"Error inserting data into table {table_name}: {e}")

    def get_recent_updates(self) -> List[tuple]:
        """
        Retrieves recent updates from all tables in the SQLite database.

        :return: List of tuples containing table names and their most recent row.
        """
        recent_updates = []
        table_names = self.get_table_names()
        for table_name in table_names:
            if table_name == 'sqlite_sequence':  # Skip internal SQLite table
                continue
            # Check if the table has an 'id' column
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [column[1] for column in cursor.fetchall()]
            if 'id' in columns:
                query = f"SELECT * FROM {table_name} ORDER BY id DESC LIMIT 1;"
            else:
                query = f"SELECT * FROM {table_name} LIMIT 1;"
            recent_row = self.execute_query(query)
            if recent_row:
                recent_updates.append((table_name, recent_row[0]))
        return recent_updates

    def close_connection(self):
        """
        Closes the database connection.
        """
        if self.connection:
            self.connection.close()
            logging.info("Database connection closed.")

# Example Usage
if __name__ == "__main__":
    db_manager = DatabaseManager()
    # Log table names
    table_names = db_manager.get_table_names()
    logging.info("Table Names: %s", table_names)

    # Load a dataset from the "knowledge_base" table
    domain_data = db_manager.load_domain_dataset('knowledge_base')
    logging.info("Domain Data (knowledge_base): %s", domain_data)

    # Update a dataset in the "knowledge_base" table
    new_data = (
        'science', 'new_topic', 'New description', 1500.0, '{"source": "New Source", "tags": ["New Tag"]}',
        '2024-12-18 15:00:00', 'medium'
    )
    db_manager.update_domain_data('knowledge_base', new_data)

    # Retrieve recent updates
    recent_updates = db_manager.get_recent_updates()
    logging.info("Recent Updates: %s", recent_updates)

    # Close the connection
    db_manager.close_connection()
