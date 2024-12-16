
# domain_knowleadgebase
from knowledge_base import TieredKnowledgeBase  # Direct import from existing file
import json
import os
import csv

class DomainKnowledgeBase:
    def __init__(self, 
                 dataset_config_path='domain_dataset.json', 
                 knowledge_base_path='knowledge_base/', 
                 knowledge_base=None):
        # If no knowledge_base provided, create a new TieredKnowledgeBase
        self.tiered_knowledge_base = knowledge_base or TieredKnowledgeBase()
        
        # Load domain datasets from the dataset configuration file
        self.dataset_config_path = dataset_config_path
        self.knowledge_base_path = knowledge_base_path
        self.domain_datasets = self.load_domain_datasets()
        
        # Initialize the knowledge base with the loaded domain datasets
        self.initialize_knowledge_base()

    def load_domain_datasets(self):
        with open(self.dataset_config_path, 'r') as f:
            return json.load(f)

    def initialize_knowledge_base(self):
        for domain, file_path in self.domain_datasets.items():
            full_file_path = os.path.join(self.knowledge_base_path, file_path)
            if os.path.isfile(full_file_path):
                with open(full_file_path, 'r') as f:
                    # Assuming CSV format, you may need to adjust based on your file format
                    dataset = list(csv.reader(f))
                    self.tiered_knowledge_base.add_domain_knowledge(domain, dataset)
            else:
                print(f"Error: File not found - {full_file_path}")

    def get_domain_knowledge(self, domain):
        return self.tiered_knowledge_base.get_domain_knowledge(domain)

    def update_domain_knowledge(self, domain, new_knowledge):
        self.tiered_knowledge_base.update_domain_knowledge(domain, new_knowledge)

    def remove_domain_knowledge(self, domain):
        self.tiered_knowledge_base.remove_domain_knowledge(domain)

# Example Usage
# if __name__ == "__main__":
#   domain_kb = DomainKnowledgeBase()
#  
#print(domain_kb.get_domain_knowledge('Math_D1'))
    # Now you can access the domain knowledge 
# base with the properly loaded datasets

