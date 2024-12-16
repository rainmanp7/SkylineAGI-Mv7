# main_knowledgebase.py
# updated Dec5 2024
import json
import os

class MainKnowledgeBase:
    def __init__(self, domain_dataset_config='domain_dataset.json'):
        self.knowledge_bases = {}  # Dictionary to store different knowledge bases
        self.domain_dataset_config = domain_dataset_config
        self.load_domain_datasets()

    def load_domain_datasets(self):
        with open(self.domain_dataset_config, 'r') as f:
            domain_datasets = json.load(f)
            for name, file_path in domain_datasets.items():
                knowledge_base = self.load_dataset(file_path)
                self.add_knowledge_base(name, knowledge_base)

    def load_dataset(self, file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r') as f:
                # Assuming CSV format, you may need to adjust based on your file format
                import csv
                knowledge_base = list(csv.reader(f))
                return knowledge_base
        else:
            print(f"Error: File not found - {file_path}")
            return []

    def add_knowledge_base(self, name, knowledge_base):
        self.knowledge_bases[name] = knowledge_base

    def get_knowledge_base(self, name):
        return self.knowledge_bases.get(name)

    def remove_knowledge_base(self, name):
        if name in self.knowledge_bases:
            del self.knowledge_bases[name]

    # Additional methods to manage knowledge across bases

# Example Usage
if __name__ == "__main__":
    main_kb = MainKnowledgeBase()
    print(main_kb.get_knowledge_base('Math_D1'))  # Access a knowledge base by name
