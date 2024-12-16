import numpy as np
import json
import os
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

class SimpleNN(nn.Module):
    def __init__(self, input_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

class CrossDomainGeneralization:
    """
    A class for cross-domain generalization, enabling knowledge transfer between domains.
    
    Attributes:
    - knowledge_base: The knowledge base instance.
    - model: The model instance.
    - domain_dataset_config: The path to the domain dataset configuration file.
    - domain_datasets: A dictionary of domain datasets.
    """

    def __init__(self, knowledge_base, model, domain_dataset_config='domain_dataset.json'):
        """
        Initialize the CrossDomainGeneralization class.
        
        Args:
        - knowledge_base: The knowledge base instance.
        - model: The model instance.
        - domain_dataset_config: The path to the domain dataset configuration file.
        """
        self.knowledge_base = knowledge_base
        self.model = model
        self.domain_dataset_config = domain_dataset_config
        self.domain_datasets = self.load_domain_datasets()

    def load_domain_datasets(self):
        """
        Load domain datasets from the configuration file.
        
        Returns:
        - A dictionary of domain datasets.
        """
        try:
            with open(self.domain_dataset_config, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading domain dataset configuration: {str(e)}")
            return {}

    def load_and_preprocess_data(self, domain, level):
        """
        Load and preprocess data from the given domain and level.
        
        Args:
        - domain: The domain name.
        - level: The level name.
        
        Returns:
        - Preprocessed data (features and labels).
        """
        try:
            domain_dataset = self.domain_datasets.get(domain)
            if domain_dataset:
                datasets = domain_dataset.get('datasets')
                if datasets:
                    file_path = datasets.get(level)
                    if file_path:
                        # Resolve the full path relative to the script's execution directory
                        script_dir = os.path.dirname(__file__)
                        full_path = os.path.join(script_dir, file_path)
                        logging.info(f"Expected file path: {full_path}")

                        if os.path.isfile(full_path):
                            logging.info(f"File found: {full_path}")
                            # Load the data using numpy
                            data = np.loadtxt(full_path, delimiter=',', skiprows=1, dtype=str)
                            headers = data[0]
                            feature_data = data[1:, :]
                            target_column = 'Compx'
                            
                            # Log the headers for debugging
                            logging.info(f"Headers found in the file: {headers}")
                            
                            # Identify feature columns
                            features = []
                            target_index = -1
                            for i, header in enumerate(headers):
                                if header not in ['Domain', 'Compx', 'Range', 'Range_Start', 'Range_End', 'info', 'CIter']:
                                    features.append(i)
                                if header == target_column:
                                    target_index = i
                            
                            if target_index == -1:
                                logging.error(f"Target column '{target_column}' not found for domain '{domain}' and level '{level}'.")
                                return None, None, None, None

                            # Convert feature data to numeric, ignoring errors
                            feature_data_numeric = []
                            labels = []
                            for row in feature_data:
                                try:
                                    numeric_row = [float(row[i]) for i in features]
                                    label = float(row[target_index])
                                    feature_data_numeric.append(numeric_row)
                                    labels.append(label)
                                except ValueError:
                                    logging.warning(f"Skipping row with non-numeric data: {row}")

                            if not feature_data_numeric or not labels:
                                logging.error(f"No valid numeric data found for domain '{domain}' and level '{level}'.")
                                return None, None, None, None

                            feature_data_numeric = np.array(feature_data_numeric)
                            labels = np.array(labels)

                            scaler = StandardScaler()
                            features_scaled = scaler.fit_transform(feature_data_numeric)

                            X_train, X_val, y_train, y_val = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

                            # Convert to PyTorch tensors
                            X_train = torch.tensor(X_train, dtype=torch.float32)
                            y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
                            X_val = torch.tensor(X_val, dtype=torch.float32)
                            y_val = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)

                            return X_train, y_train, X_val, y_val

                        else:
                            logging.error(f"File not found for domain '{domain}' and level '{level}': {full_path}")
                            return None, None, None, None

                    else:
                        logging.error(f"No file path found for domain '{domain}' and level '{level}'.")
                        return None, None, None, None

                else:
                    logging.error(f"No datasets found for domain '{domain}'.")
                    return None, None, None, None

            else:
                logging.error(f"No dataset found for domain '{domain}'.")
                return None, None, None, None

        except Exception as e:
            logging.error(f"Error loading data for domain '{domain}' and level '{level}': {str(e)}")
            return None, None, None, None

    def transfer_knowledge(self, source_domain, target_domain):
        """
        Transfer knowledge from the source domain to the target domain.
        
        Args:
        - source_domain: The source domain name.
        - target_domain: The target domain name.
        """
        source_knowledge = self.knowledge_base.query(source_domain)

        if not source_knowledge:
            logging.error(f"No knowledge found for source domain '{source_domain}'.")
            return

        # Placeholder for knowledge transfer logic
        logging.info(f"Knowledge transferred from {source_domain} to {target_domain}.")

    def fine_tune_model(self, domain, level):
        """
        Fine-tune the model for the given domain and level.
        
        Args:
        - domain: The domain name.
        - level: The level name.
        """
        X_train, y_train, X_val, y_val = self.load_and_preprocess_data(domain, level)

        if X_train is None:
            logging.error("Training data could not be loaded. Fine-tuning aborted.")
            return

        input_size = X_train.shape[1]
        self.model = SimpleNN(input_size)  # Reinitialize the model with the correct input size

        criterion = nn.MSELoss()  # Use MSE for regression since 'Compx' is numeric
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            predictions = self.model(X_val)
            predictions_classes = (predictions > 0.5).float()

        # Since 'Compx' is numeric, we can't use accuracy, precision, recall, f1_score directly
        # Instead, we can use metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE)
        mse = criterion(predictions, y_val).item()
        
        logging.info(f"Model fine-tuned on '{domain}' level '{level}' with MSE: {mse:.2f}")

    def evaluate_cross_domain_performance(self, domains_levels):
        """
        Evaluate the model's performance across multiple domains and levels.
        
        Args:
        - domains_levels: A list of tuples (domain, level).
        
        Returns:
        - A dictionary with performance metrics for each domain and level.
        """
        results = {}
        
        for domain, level in domains_levels:
            result = self.evaluate_domain(domain, level)
            if domain not in results:
                results[domain] = {}
            results[domain][level] = result
        
        return results

    def evaluate_domain(self, domain, level):
        """
        Evaluate the model's performance on a single domain and level.
        
        Args:
        - domain: The domain name.
        - level: The level name.
        
        Returns:
        - A dictionary with performance metrics.
        """
        X_train, y_train, X_val, y_val = self.load_and_preprocess_data(domain, level)
        
        if X_train is not None:
            input_size = X_train.shape[1]
            model = SimpleNN(input_size)  # Reinitialize the model with the correct input size

            criterion = nn.MSELoss()  # Use MSE for regression since 'Compx' is numeric
            optimizer = optim.Adam(model.parameters(), lr=0.001)

            for epoch in range(10):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                predictions = model(X_val)
                predictions_classes = (predictions > 0.5).float()

            # Since 'Compx' is numeric, we can't use accuracy, precision, recall, f1_score directly
            # Instead, we can use metrics like Mean Squared Error (MSE) or Mean Absolute Error (MAE)
            mse = criterion(predictions, y_val).item()

            return {
                'mse': mse,
            }
        
        return None

if __name__ == "__main__":
    # Configure logging to write to a file
    log_file = 'cross_domain_generalization.log'
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Get the number of available CPUs
    cpu_count = os.cpu_count()
    logging.info(f"Available CPU count: {cpu_count}")

    # Initialize the model
    # The input size will be dynamically determined in fine_tune_model and evaluate_domain
    model = SimpleNN(input_size=1)  # Initial input size, will be updated

    # Initialize the CrossDomainGeneralization class
    cdg = CrossDomainGeneralization(None, model, domain_dataset_config='domain_dataset.json')

    # Simulate performance data for multiple domains and levels
    domains_levels = [('Math', 'level_1'), ('Math', 'level_2'), ('Science', 'level_1')]

    # Evaluate cross-domain performance
    results = cdg.evaluate_cross_domain_performance(domains_levels)
    logging.info(f"Cross-domain performance results: {results}")

    # Display the number of CPUs used
    logging.info(f"Used CPU count: {cpu_count}")