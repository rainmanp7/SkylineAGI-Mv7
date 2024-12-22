#moved
import sqlite3
import os
import logging
from typing import Optional

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseSchemaExporter:
    def __init__(self, database_path: str):
        """
        Initializes the DatabaseSchemaExporter with the path to the database.

        :param database_path: Path to the SQLite database file.
        """
        self.database_path = database_path
        logging.info(f"Database Path: {self.database_path}")

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

    def export_schema(self) -> Optional[str]:
        """
        Exports the schema of the SQLite database.

        :return: The schema of the database as a string or None if an error occurs.
        """
        connection = self.connect_to_database()
        if connection is None:
            return None

        try:
            cursor = connection.cursor()
            cursor.execute("SELECT sql FROM sqlite_master WHERE type='table';")
            schema = cursor.fetchall()
            schema_str = "\n".join([table[0] for table in schema if table[0] is not None])
            logging.info("Schema exported successfully.")
            return schema_str
        except sqlite3.Error as e:
            logging.error(f"Error exporting schema: {e}")
            return None
        finally:
            connection.close()
            logging.info("Database connection closed.")

    def save_schema_to_file(self, schema: str, output_file: str):
        """
        Saves the schema to a file.

        :param schema: The schema of the database as a string.
        :param output_file: The path to the output file.
        """
        try:
            with open(output_file, 'w') as file:
                file.write(schema)
            logging.info(f"Schema saved to {output_file}.")
        except IOError as e:
            logging.error(f"Error saving schema to file: {e}")

# Example Usage
if __name__ == "__main__":
    db_path = 'skyline_agi.db'
    schema_exporter = DatabaseSchemaExporter(db_path)
    schema = schema_exporter.export_schema()
    if schema:
        schema_exporter.save_schema_to_file(schema, 'skyline_agi_schema.sql')
