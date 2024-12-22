# cross_domain_evaluation.py
# Updated Dec 22

import sqlite3
import os
import json
from datetime import datetime
from typing import List, Any, Optional
import logging
from agi_config import AGIConfiguration
from complexity import ComplexityRange
from complexity import ComplexityMetrics
from logging_config import setup_logging

# Set up logging
logger = setup_logging(script_name="cross_domain_evaluation")

class DatabaseManager:
    def __init__(self, config_path: str = 'config.json'):
        """
        Initializes the DatabaseManager with the configuration file path.

        :param config_path: Path to the configuration file (default: 'config.json').
        """
        self.config = AGIConfiguration(config_path)
        self.database_path = self.config.get_database_path()
        logger.info(f"Database Path: {self.database_path}")  # Debug log
        self.connection = self.connect_to_database()

    def connect_to_database(self) -> Optional[sqlite3.Connection]:
        """
        Connects to the SQLite database and returns the connection object.

        :return: SQLite connection object or None if connection fails.
        """
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
        """
        Executes a query against the database and optionally fetches results.

        :param query: The SQL query to execute.
        :param params: Parameters to substitute into the query (default: empty tuple).
        :return: List of results for SELECT queries, or None for others.
        """
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
            logger.error("Database connection is not initialized.")
            return
        try:
            cursor = self.connection.cursor()
            # Retrieve column names for the table, excluding the 'id' column
            cursor.execute(f"PRAGMA table_info({table_name});")
            columns = [column[1] for column in cursor.fetchall() if column[1] != 'id']

            # Ensure the data tuple matches the number of columns (excluding 'id')
            if len(data) != len(columns):
                logger.error(f"Data tuple length mismatch for table {table_name}. Expected {len(columns)} columns, got {len(data)}.")
                return

            # Create placeholders and construct the query
            placeholders = ', '.join(['?'] * len(data))
            column_list = ', '.join(columns)
            query = f"INSERT INTO {table_name} ({column_list}) VALUES ({placeholders});"
            cursor.execute(query, data)
            self.connection.commit()
            logger.info(f"Data inserted into table {table_name}.")
        except sqlite3.Error as e:
            logger.error(f"Error inserting data into table {table_name}: {e}")

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
            logger.info("Database connection closed.")


class CrossDomainGeneralization:
    def __init__(self, db_manager: DatabaseManager):
        if db_manager.connection is None:
            logger.error("Database connection is not initialized for CrossDomainGeneralization.")
            raise ValueError("Database connection is not initialized.")
        self.db = db_manager

    def fetch_knowledge(self, domain: str = None, complexity_range: str = None) -> Optional[List[tuple]]:
        """
        Retrieve knowledge entries based on domain and complexity range.

        :param domain: Domain to filter by (default: None, meaning no filter).
        :param complexity_range: Complexity range to filter by (default: None, meaning no filter).
        :return: List of knowledge entries or None if the query fails.
        """
        query = "SELECT * FROM knowledge_base WHERE 1=1"
        params = []
        if domain:
            query += " AND category = ?"
            params.append(domain)
        if complexity_range:
            query += " AND complexity_range = ?"
            params.append(complexity_range)

        return self.db.execute_query(query, tuple(params))

    def classify_complexity(self, score: float) -> str:
        """Classify a complexity score into its range."""
        return ComplexityRange.normalize_to_range(score)

    def assimilate_data(self, category: str, key: str, value: str, complexity_score: float, metadata: dict = None):
        """
        Add or update knowledge in the database.

        :param category: Category of the knowledge.
        :param key: Key of the knowledge.
        :param value: Value of the knowledge.
        :param complexity_score: Complexity score of the knowledge.
        :param metadata: Additional metadata (default: None).
        """
        complexity_range = self.classify_complexity(complexity_score)
        metadata_json = json.dumps(metadata) if metadata else None
        query = """
            INSERT INTO knowledge_base (category, key, value, complexity_score, metadata, complexity_range, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """
        params = (category, key, value, complexity_score, metadata_json, complexity_range, datetime.now())
        self.db.execute_query(query, params)

    def cross_domain_reasoning(self, query_key: str, target_domain: str) -> List[str]:
        """
        Analyze a query from one domain and apply knowledge to another.

        :param query_key: Key to query in the source domain.
        :param target_domain: Target domain to apply the knowledge to.
        :return: List of reasoning results.
        """
        source_data = self.db.execute_query("SELECT * FROM knowledge_base WHERE key = ?", (query_key,))
        if not source_data:
            logger.warning(f"No data found for key: {query_key}")
            return []

        results = []
        for item in source_data:
            # Hypothetical logic to transfer insights
            results.append(f"Transferring {item[1]} to {target_domain}")
        return results


# Example Usage
if __name__ == "__main__":
    db_manager = DatabaseManager()
    if db_manager.connection is None:
        logger.error("Failed to initialize database connection. Exiting.")
        exit(1)

    generalizer = CrossDomainGeneralization(db_manager)

    # Test fetching knowledge
    knowledge = generalizer.fetch_knowledge(domain="science")
    logger.info(f"Knowledge fetched: {knowledge}")
    print(f"Knowledge fetched: {knowledge}")

    # Assimilate new data
    generalizer.assimilate_data(
        category="finance",
        key="economic_cycles",
        value="Economic cycles affect global trade patterns.",
        complexity_score=2300,
        metadata={"source": "Economic Journal", "tags": ["economics", "trade"]}
    )

    # Test cross-domain reasoning
    reasoning_result = generalizer.cross_domain_reasoning("economic_cycles", "science")
    logger.info(f"Reasoning result: {reasoning_result}")
    print(f"Reasoning result: {reasoning_result}")

    # Close the connection
    db_manager.close_connection()
