import sqlite3
import os
import logging
from typing import Optional

# Setting up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseDumper:
    def __init__(self, database_path: str, output_file: str):
        """
        Initializes the DatabaseDumper with the path to the database and the output file.

        :param database_path: Path to the SQLite database file.
        :param output_file: Path to the output SQL dump file.
        """
        self.database_path = database_path
        self.output_file = output_file
        logging.info(f"Database Path: {self.database_path}")
        logging.info(f"Output File: {self.output_file}")

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

    def dump_database(self) -> bool:
        """
        Dumps the entire database to an SQL file.

        :return: True if the dump is successful, False otherwise.
        """
        connection = self.connect_to_database()
        if connection is None:
            return False

        try:
            with open(self.output_file, 'w') as file:
                for line in connection.iterdump():
                    file.write(f"{line}\n")
            logging.info(f"Database dumped successfully to {self.output_file}.")
            return True
        except IOError as e:
            logging.error(f"Error writing to file {self.output_file}: {e}")
            return False
        except sqlite3.Error as e:
            logging.error(f"Error during database dump: {e}")
            return False
        finally:
            connection.close()
            logging.info("Database connection closed.")

# Example Usage
if __name__ == "__main__":
    db_path = 'skyline_agi.db'
    output_file = 'skyline_agi_dump.sql'
    db_dumper = DatabaseDumper(db_path, output_file)
    if db_dumper.dump_database():
        logging.info("Database dump completed successfully.")
    else:
        logging.error("Database dump failed.")