# begin database_manager.py
import json
import os
from agi_config import AGIConfiguration

class DatabaseManager:
    def __init__(self, knowledge_base_path=None, domain_dataset_config='domain_dataset.json'):
        self.config = AGIConfiguration()
        if knowledge_base_path is None:
            knowledge_base_path = self.config.get_dynamic_setting('knowledge_base_path')
        self.knowledge_base_path = knowledge_base_path
        self.domain_dataset_config = domain_dataset_config
        self.domain_datasets = self.load_domain_datasets()

    def load_domain_datasets(self):
        with open(self.domain_dataset_config, 'r') as f:
            return json.load(f)

    def load_domain_dataset(self, domain, index, params=None):
        domain_dataset = self.domain_datasets.get(domain)
        if domain_dataset:
            file_path = domain_dataset
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    # Assuming CSV format, you may need to adjust based on your file format
                    import csv
                    dataset = list(csv.reader(f))
                    if index < len(dataset):
                        return dataset[index]
                    else:
                        print(f"Error: Index out of range - {index}")
                        return []
            else:
                print(f"Error: File not found - {file_path}")
                return []
        else:
            print(f"Error: Domain not found - {domain}")
            return []

    def update_domain_data(self, domain, data):
        domain_dataset = self.domain_datasets.get(domain)
        if domain_dataset:
            file_path = domain_dataset
            with open(file_path, 'a', newline='') as f:
                # Assuming CSV format, you may need to adjust based on your file format
                import csv
                writer = csv.writer(f)
                writer.writerow(data)
        else:
            print(f"Error: Domain not found - {domain}")

    def get_recent_updates(self):
        # Implementation to retrieve recent updates
        recent_updates = []
        for domain, file_path in self.domain_datasets.items():
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    # Assuming CSV format, you may need to adjust based on your file format
                    import csv
                    dataset = list(csv.reader(f))
                    if dataset:
                        recent_updates.append((domain, dataset[-1]))
            else:
                print(f"Error: File not found - {file_path}")
        return recent_updates

# Example Usage
if __name__ == "__main__":
    db_manager = DatabaseManager()
    print(db_manager.load_domain_dataset('Math_D1', 0))  # Load a domain dataset by index
    db_manager.update_domain_data('Math_D1', ['new', 'data', 'row'])  # Update a domain dataset
    print(db_manager.get_recent_updates())  # Retrieve recent updates

# End database_manager.py