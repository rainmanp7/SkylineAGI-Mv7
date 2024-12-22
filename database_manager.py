import sqlite3
import os
from typing import List, Any, Optional
from agi_config import AGIConfiguration
from logging_config import setup_logging

# Setting up logging
logger = setup_logging()

class DatabaseManager:
    def __init__(self, config_path: str = 'config.json'):
        self.config = AGIConfiguration(config_path)
        self.database_path = self.config.get_database_path()
        logger.info(f"Database Path: {self.database_path}")  # Debug log
        self.connection = self.connect_to_database()

    def connect_to_database(self) -> Optional[sqlite3.Connection]:
        if not os.path.isfile(self.database_path):
            logger.error(f"Error: Database file not found - {self.database_path}")
            return None
        try:
            connection = sqlite3.connect(self.database_path)
            logger.info("Successfully connected to the database.")
            return connection
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {e}")
            return None

    def execute_query(self, query: str, params: tuple = ()) -> Optional[List[tuple]]:
        if self.connection is None:
            logger.error("Database connection is not initialized.")
            return None
        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            if query.strip().lower().startswith("select"):
                results = cursor.fetchall()
                logger.debug(f"Query Results: {results}")
                return results
            else:
                self.connection.commit()
                logger.info("Query executed successfully.")
                return None
        except sqlite3.Error as e:
            logger.error(f"Error executing query: {e}")
            return None

    def get_table_names(self) -> List[str]:
        query = "SELECT name FROM sqlite_master WHERE type='table';"
        result = self.execute_query(query)
        return [table[0] for table in result] if result else []

    def load_domain_dataset(self, table_name: str) -> List[tuple]:
        query = f"SELECT * FROM {table_name};"
        return self.execute_query(query) or []

    def update_domain_data(self, table_name: str, data: tuple):
        if self.connection is None:
            logger.error("Database connection is not initialized.")
            return
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [column[1] for column in cursor.fetchall() if column[1] != 'id']
            if len(data) != len(columns):
                logger.error(f"Data tuple length mismatch for table {table_name}. Expected {len(columns)} columns, got {len(data)}.")
                return
            placeholders = ', '.join(['?'] * len(data))
            column_list = ', '.join(columns)
            query = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders});"
            cursor.execute(query, data)
            self.connection.commit()
            logger.info(f"Data inserted into table {table_name}.")
        except sqlite3.Error as e:
            logger.error(f"Error inserting data into table {table_name}: {e}")

    def get_recent_updates(self) -> List[tuple]:
        recent_updates = []
        table_names = self.get_table_names()
        for table_name in table_names:
            if table_name == 'sqlite_sequence':
                continue
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
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed.")

if __name__ == "__main__":
    db_manager = DatabaseManager()
    table_names = db_manager.get_table_names()
    logger.info("Table Names: %s", table_names)
    domain_data = db_manager.load_domain_dataset('knowledge_base')
    logger.info("Domain Data (knowledge_base): %s", domain_data)
    new_data = (
        'science', 'new_topic', 'New description', 1500.0, '{"source": "New Source", "tags": ["New Tag"]}',
        '2024-12-18 15:00:00', 'medium'
    )
    db_manager.update_domain_data('knowledge_base', new_data)
    recent_updates = db_manager.get_recent_updates()
    logger.info("Recent Updates: %s", recent_updates)
    db_manager.close_connection()
