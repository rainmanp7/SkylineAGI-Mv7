# agi_config.py
# updated Dec22 2024

import json
import logging
from typing import Dict, Any, Optional
from logging_config import setup_logging

# Configure logging using setup_logging
setup_logging()

_config_executed = False  # Flag to prevent duplicate execution

class AGIConfiguration:
    def __init__(self, config_path: str = 'config.json'):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads the configuration from a JSON file.

        :param config_path: Path to the configuration file.
        :return: Configuration dictionary.
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found - {config_path}")
            return {}
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from configuration file - {config_path}: {e}")
            return {}
        except UnicodeDecodeError as e:
            logging.error(f"Error decoding configuration file (UTF-8 expected) - {config_path}: {e}")
            return {}
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading the configuration file - {config_path}: {e}")
            return {}

    def get_dynamic_setting(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a dynamic setting from the configuration.

        :param key: The key of the setting to retrieve.
        :param default: The default value to return if the key is not found.
        :return: The value of the setting or the default value.
        """
        return self.config.get(key, default)

    def get_database_settings(self) -> Dict[str, Any]:
        """
        Retrieves the database settings from the configuration.

        :return: Dictionary containing database settings.
        """
        return self.config.get('database', {})

    def get_database_path(self) -> str:
        """
        Retrieves the database path from the configuration.

        :return: The path to the database file.
        """
        return self.get_database_settings().get('database_path', '')

    def get_knowledge_base_path(self) -> str:
        """
        Retrieves the knowledge base path from the configuration.

        :return: The path to the knowledge base directory.
        """
        return self.get_database_settings().get('knowledge_base_path', '')

    def get_domain_dataset_config(self) -> str:
        """
        Retrieves the domain dataset configuration path from the configuration.

        :return: The path to the domain dataset configuration file.
        """
        return self.get_database_settings().get('domain_dataset_config', '')


# Proof of functionality when executed directly
if __name__ == "__main__" and not _config_executed:
    _config_executed = True  # Set the flag to True
    logging.info("AGI Configuration File Triggered")
    
    # Initialize the configuration
    agi_config = AGIConfiguration()
    
    # Fetch and print configuration details
    database_settings = agi_config.get_database_settings()
    database_path = agi_config.get_database_path()
    knowledge_base_path = agi_config.get_knowledge_base_path()
    domain_dataset_config = agi_config.get_domain_dataset_config()
    
    logging.info("Database Settings: %s", database_settings)
    logging.info("Database Path: %s", database_path)
    logging.info("Knowledge Base Path: %s", knowledge_base_path)
    logging.info("Domain Dataset Config Path: %s", domain_dataset_config)
    logging.info("Proof of functionality: AGI Configuration loaded and operational.")
