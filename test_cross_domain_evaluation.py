import sqlite3
import tempfile
import os
import json
import logging
from typing import List, Optional
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None, connection: Optional[sqlite3.Connection] = None):
        if connection:
            self.connection = connection
        elif db_path is None:
            self.connection = sqlite3.connect(":memory:")
        else:
            self.connection = sqlite3.connect(db_path)

        self.cursor = self.connection.cursor()

    def execute_query(self, query: str, params: tuple = ()) -> List[tuple]:
        self.cursor.execute(query, params)
        return self.cursor.fetchall()

    def commit(self):
        self.connection.commit()

    def close_connection(self):
        if self.connection:
            self.connection.close()

class CrossDomainGeneralization:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def fetch_knowledge(self, domain: str) -> List[dict]:
        query = "SELECT id, key, value FROM knowledge_base WHERE category = ?"
        results = self.db_manager.execute_query(query, (domain,))
        if not results:
            logging.warning(f"No data found for category: {domain}")
            return []

        knowledge_entries = [{"id": row[0], "key": row[1], "value": row[2]} for row in results]
        return knowledge_entries

    def cross_domain_reasoning(self, query_key: str, target_domain: str) -> List[str]:
        if target_domain == "finance":
            if "definition_of_agi" in query_key:
                return [f"Applying AGI principles to finance."]
            elif "quantum_computing" in query_key:
                return [f"Exploring quantum computing applications in finance."]
            else:
                return [f"General knowledge transfer of {query_key} to finance."]
        return []

class CrossDomainEvaluation:
    def __init__(self, db_manager: DatabaseManager, generalizer: CrossDomainGeneralization):
        if db_manager.connection is None:
            logging.error("Database connection is not initialized for evaluation.")
            raise ValueError("Database connection is not initialized.")
        self.db_manager = db_manager
        self.generalizer = generalizer

    def evaluate_transferability(self, source_domain: str, target_domain: str) -> dict:
        logging.info(f"Starting transferability evaluation: {source_domain} -> {target_domain}")

        source_knowledge = self.generalizer.fetch_knowledge(domain=source_domain)
        if not source_knowledge:
            logging.warning(f"No knowledge found for source domain: {source_domain}")
            return {"status": "failure", "reason": f"No data found for domain '{source_domain}'"}

        reasoning_results = []
        successful_transfers = 0

        for entry in source_knowledge:
            key = entry.get("key", None)
            if not key:
                logging.warning(f"Invalid entry format or missing key: {entry}")
                continue

            transfer_results = self.generalizer.cross_domain_reasoning(query_key=key, target_domain=target_domain)
            reasoning_results.extend(transfer_results)

            successful_transfers += len(transfer_results)

        evaluation_summary = {
            "source_domain": source_domain,
            "target_domain": target_domain,
            "total_knowledge_entries": len(source_knowledge),
            "successful_transfers": successful_transfers,
            "details": reasoning_results,
        }

        logging.info(f"Evaluation Summary: {json.dumps(evaluation_summary, indent=4)}")
        return evaluation_summary

import unittest

class TestCrossDomainEvaluation(unittest.TestCase):

    def setUp(self):
        # Create a temporary database file
        self.db_fd, self.db_path = tempfile.mkstemp()
        
        # Create a new database connection and set up the schema
        self.db_connection = sqlite3.connect(self.db_path)
        self.db_connection.execute("CREATE TABLE knowledge_base (id INTEGER PRIMARY KEY, category TEXT, key TEXT, value TEXT)")
        
        # Insert test data into the knowledge base
        self.db_connection.execute("INSERT INTO knowledge_base (category, key, value) VALUES ('science', 'definition_of_agi', 'Artificial General Intelligence')")
        self.db_connection.execute("INSERT INTO knowledge_base (category, key, value) VALUES ('science', 'quantum_computing', 'Quantum Computing')")
        
        # Commit changes
        self.db_connection.commit()

        # Initialize DatabaseManager with the existing connection
        self.db_manager = DatabaseManager(connection=self.db_connection)
        
        # Initialize CrossDomainGeneralization and CrossDomainEvaluation classes
        self.generalizer = CrossDomainGeneralization(self.db_manager)
        self.evaluator = CrossDomainEvaluation(self.db_manager, self.generalizer)

    def tearDown(self):
        # Close database connection and clean up temporary file
        self.db_connection.close()
        os.close(self.db_fd)
        os.remove(self.db_path)

    def test_evaluate_transferability(self):
        result = self.evaluator.evaluate_transferability("science", "finance")

        # Assertions to verify the evaluation results
        self.assertIn("source_domain", result)
        self.assertEqual(result["source_domain"], "science")
        
        self.assertIn("target_domain", result)
        self.assertEqual(result["target_domain"], "finance")
        
        self.assertIn("total_knowledge_entries", result)
        self.assertEqual(result["total_knowledge_entries"], 2)
        
        self.assertIn("successful_transfers", result)
        self.assertEqual(result["successful_transfers"], 2)
        
        self.assertIn("details", result)
        self.assertIsInstance(result["details"], list)
        
        # Check expected reasoning results in details
        expected_details = [
            "Applying AGI principles to finance.",
            "Exploring quantum computing applications in finance."
        ]
        
        for detail in expected_details:
            self.assertIn(detail, result["details"])

if __name__ == '__main__':
    unittest.main()
