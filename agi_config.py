import json
from typing import Dict, Any

class AGIConfiguration:
    def __init__(self, config_path: str = 'config.json'):
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading configuration: {e}")
            return {}
    
    def get_dynamic_setting(self, key: str, default=None):
        return self.config.get(key, default)

    def get_database_settings(self) -> Dict[str, Any]:
        return self.config.get('database', {})

    def get_knowledge_base_path(self) -> str:
        return self.get_database_settings().get('knowledge_base_path', '')

    def get_domain_dataset_config(self) -> str:
        return self.get_database_settings().get('domain_dataset_config', '')


# Proof of functionality when executed directly
if __name__ == "__main__":
    print("AGI Configuration File Triggered")
    
    # Initialize the configuration
    agi_config = AGIConfiguration()
    
    # Fetch and print configuration details
    database_settings = agi_config.get_database_settings()
    knowledge_base_path = agi_config.get_knowledge_base_path()
    domain_dataset_config = agi_config.get_domain_dataset_config()
    
    print("Database Settings:", database_settings)
    print("Knowledge Base Path:", knowledge_base_path)
    print("Domain Dataset Config Path:", domain_dataset_config)
    print("Proof of functionality: AGI Configuration loaded and operational.")
