import sqlite3
import os
import json
import logging
from typing import List, Optional
from datetime import datetime
from agi_config import AGIConfiguration
from complexity import ComplexityRange
from database_manager import DatabaseManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CrossDomainGeneralization:
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager

    def fetch_knowledge(self, domain: str) -> List[dict]:
        """
        Fetch knowledge entries for a specific category from the knowledge_base table.
        :param domain: The category to fetch knowledge for.
        :return: List of dictionaries representing the knowledge.
        """
        query = "SELECT id, key, value FROM knowledge_base WHERE category = ?"
        results = self.db_manager.execute_query(query, (domain,))
        if not results:
            logging.warning(f"No data found for category: {domain}")
            return []

        # Convert tuples to dictionaries
        knowledge_entries = [{"id": row[0], "key": row[1], "value": row[2]} for row in results]
        return knowledge_entries

    def cross_domain_reasoning(self, query_key: str, target_domain: str) -> List[str]:
        """
        Perform reasoning to transfer knowledge from one domain to another.
        :param query_key: The key of the knowledge entry.
        :param target_domain: The target domain for knowledge transfer.
        :return: List of transfer results.
        """
        # Here you can implement actual reasoning logic
        # For instance, you could look up relevant transformations, mappings, or domain-specific adjustments
        # This is a placeholder for actual reasoning logic; replace with your implementation
        # You might want to retrieve some contextual information from the database or perform some logic
        results = []

        # Example logic: simple transformation based on keywords
        if target_domain == "finance":
            if "definition_of_agi" in query_key:
                results.append(f"Applying AGI principles to finance.")
            elif "quantum_computing" in query_key:
                results.append(f"Exploring quantum computing applications in finance.")
            else:
                results.append(f"General knowledge transfer of {query_key} to finance.")

        return results

class CrossDomainEvaluation:
    def __init__(self, db_manager: DatabaseManager, generalizer: CrossDomainGeneralization):
        """
        Initializes CrossDomainEvaluation with the database manager and generalization object.

        :param db_manager: DatabaseManager instance.
        :param generalizer: CrossDomainGeneralization instance.
        """
        if db_manager.connection is None:
            logging.error("Database connection is not initialized for evaluation.")
            raise ValueError("Database connection is not initialized.")
        self.db_manager = db_manager
        self.generalizer = generalizer

    def evaluate_transferability(self, source_domain: str, target_domain: str) -> dict:
        """
        Evaluate the transferability of knowledge between domains.

        :param source_domain: Domain to retrieve knowledge from.
        :param target_domain: Domain to transfer knowledge to.
        :return: Dictionary with evaluation summary.
        """
        logging.info(f"Starting transferability evaluation: {source_domain} -> {target_domain}")

        # Fetch knowledge from the source domain
        source_knowledge = self.generalizer.fetch_knowledge(domain=source_domain)
        if not source_knowledge:
            logging.warning(f"No knowledge found for source domain: {source_domain}")
            return {"status": "failure", "reason": f"No data found for domain '{source_domain}'"}

        reasoning_results = []
        successful_transfers = 0

        for entry in source_knowledge:
            key = entry.get("key", None)  # Ensure the entry contains the "key" field
            if not key:
                logging.warning(f"Invalid entry format or missing key: {entry}")
                continue

            transfer_results = self.generalizer.cross_domain_reasoning(query_key=key, target_domain=target_domain)
            reasoning_results.extend(transfer_results)

            # Count successful transfers based on custom logic
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

# Example Usage
if __name__ == "__main__":
    # Initialize database and generalization components
    db_manager = DatabaseManager()
    if db_manager.connection is None:
        logging.error("Failed to initialize database connection. Exiting.")
        exit(1)

    generalizer = CrossDomainGeneralization(db_manager)
    evaluator = CrossDomainEvaluation(db_manager, generalizer)

    # Perform cross-domain evaluation
    evaluation_result = evaluator.evaluate_transferability("science", "finance")
    print("Evaluation Result:", json.dumps(evaluation_result, indent=4))

    # Close the database connection
    db_manager.close_connection()
