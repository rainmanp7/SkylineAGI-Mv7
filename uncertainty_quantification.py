# Created on Nov 14, 2024, 9:40 pm
# Uncertainty Code implementation.
# modified 10:43am Dec5
# real-world data and database Dec16

import numpy as np
import scipy.stats as stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import RandomForestClassifier
import logging
from database_manager import DatabaseManager

class UncertaintyQuantification:
    def __init__(self, config: dict = None):
        """
        Initialize UncertaintyQuantification with a given config.

        Args:
        - config (dict): Configuration dictionary. Defaults to None.
        """
        self.config = config or {}
        self.epistemic_uncertainty: float = 0.0
        self.aleatoric_uncertainty: float = 0.0
        self.confidence_level: float = 0.0
        self.uncertainty_threshold: float = self.config.get('uncertainty_threshold', 0.5)
        self.calibration_method: str = self.config.get('calibration_method', 'histogram')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.db_manager = DatabaseManager()

    def estimate_epistemic(self, ensemble_predictions: np.ndarray) -> float:
        """
        Estimate epistemic uncertainty from ensemble predictions.

        Args:
        - ensemble_predictions (np.ndarray): Ensemble prediction outputs.

        Returns:
        - float: Estimated epistemic uncertainty.
        """
        try:
            epistemic_var = np.var(ensemble_predictions, axis=0)
            self.epistemic_uncertainty = np.mean(epistemic_var)
            return self.epistemic_uncertainty
        except Exception as e:
            self.logger.error(f"Error in epistemic uncertainty estimation: {e}")
            return None

    def handle_aleatoric(self, data_variance: float, method: str = 'std_dev') -> float:
        """
        Handle aleatoric uncertainty based on the given method.

        Args:
        - data_variance (float): Data variance.
        - method (str): Method to handle aleatoric uncertainty. Defaults to 'std_dev'.

        Returns:
        - float: Handled aleatoric uncertainty.
        """
        try:
            if method == 'std_dev':
                self.aleatoric_uncertainty = np.sqrt(data_variance)
            elif method == 'variance':
                self.aleatoric_uncertainty = data_variance
            else:
                self.aleatoric_uncertainty = method
            return self.aleatoric_uncertainty
        except Exception as e:
            self.logger.error(f"Error in aleatoric uncertainty handling: {e}")
            return None

    def calibrate_confidence(self, predictions: np.ndarray, true_labels: np.ndarray, method: str = None) -> tuple:
        """
        Calibrate confidence level based on the given method.

        Args:
        - predictions (np.ndarray): Model predictions.
        - true_labels (np.ndarray): True labels.
        - method (str): Calibration method. Defaults to None, using config's calibration_method.

        Returns:
        - tuple: (confidence_level, calibrated_predictions)
        """
        try:
            method = method or self.calibration_method
            if method == 'histogram':
                # Directly use predicted probabilities for histogram calibration
                prob_true, prob_pred = calibration_curve(true_labels, predictions[:, 1], n_bins=10)
                self.confidence_level = 1 - np.mean(np.abs(prob_true - prob_pred))
                return self.confidence_level, None
            else:
                # Log a warning and default to histogram if other methods are requested without feature data
                self.logger.warning("Original feature data not available for calibration method {}. Defaulting to 'histogram'.".format(method))
                return self.calibrate_confidence(predictions, true_labels, method='histogram')
        except Exception as e:
            self.logger.error(f"Error in confidence calibration: {e}")
            return None, None

    def make_decision_with_uncertainty(self, predictions: np.ndarray, uncertainty_threshold: float = None) -> dict:
        """
        Make a decision based on uncertainty.

        Args:
        - predictions (np.ndarray): Model predictions.
        - uncertainty_threshold (float): Threshold for uncertainty. Defaults to None, using self.uncertainty_threshold.

        Returns:
        - dict: Decision outcome with confidence and total uncertainty.
        """
        try:
            uncertainty_threshold = uncertainty_threshold or self.uncertainty_threshold
            total_uncertainty = self.epistemic_uncertainty + self.aleatoric_uncertainty
            if isinstance(total_uncertainty, np.ndarray):
                total_uncertainty = np.mean(total_uncertainty)
            if total_uncertainty < uncertainty_threshold:
                decision = np.mean(predictions)
                confidence = "High"
            else:
                decision = None
                confidence = "Low"
            return {"decision": decision, "confidence": confidence, "total_uncertainty": total_uncertainty}
        except Exception as e:
            self.logger.error(f"Error in decision-making with uncertainty: {e}")
            return None

    def log_uncertainty_metrics(self, log_level: str = 'INFO') -> dict:
        """
        Log uncertainty metrics.

        Args:
        - log_level (str): Logging level. Defaults to 'INFO'.

        Returns:
        - dict: Uncertainty metrics.
        """
        uncertainty_log = {
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "confidence_level": self.confidence_level
        }
        self.logger.log(getattr(logging, log_level), f"Uncertainty Metrics: {uncertainty_log}")
        return uncertainty_log

    def fetch_and_correct_data(self) -> tuple:
        """
        Fetch data from the database and correct it if necessary.

        Returns:
        - tuple: (model_predictions, ensemble_predictions, data_variance, predictions_probabilities, true_labels)
        """
        try:
            query = "SELECT model_predictions, ensemble_predictions, data_variance, predictions_probabilities, true_labels FROM predictions LIMIT 100;"
            cursor = self.db_manager.connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                raise ValueError("No data found in the database.")
            model_predictions = np.array([float(row[0]) for row in rows])
            ensemble_predictions = np.array([eval(row[1]) for row in rows])
            data_variance = np.array([float(row[2]) for row in rows])
            predictions_probabilities = np.array([eval(row[3]) for row in rows])
            true_labels = np.array([int(row[4]) for row in rows])
            if predictions_probabilities.ndim == 1:
                predictions_probabilities = np.array([predictions_probabilities])
            elif predictions_probabilities.ndim == 3:
                predictions_probabilities = predictions_probabilities.reshape(-1, predictions_probabilities.shape[-1])
            self.logger.info(f"Initial shapes - predictions_probabilities: {predictions_probabilities.shape}, true_labels: {true_labels.shape}")
            if predictions_probabilities.shape[0] != true_labels.shape[0]:
                self.logger.warning("Number of samples in predictions_probabilities and true_labels do not match. Fetching additional samples.")
                predictions_probabilities, true_labels = self.fetch_additional_samples(predictions_probabilities, true_labels)
            self.logger.info(f"Corrected shapes - predictions_probabilities: {predictions_probabilities.shape}, true_labels: {true_labels.shape}")
            return model_predictions, ensemble_predictions, data_variance, predictions_probabilities, true_labels
        except Exception as e:
            self.logger.error(f"Error fetching data from the database: {e}")
            model_predictions = np.array([0.7, 0.3])
            ensemble_predictions = np.array([[0.6, 0.4], [0.8, 0.2], [0.7, 0.3]])
            data_variance = 0.1
            predictions_probabilities = np.array([[0.4, 0.6], [0.3, 0.7]])
            true_labels = np.array([1, 0])
            return model_predictions, ensemble_predictions, data_variance, predictions_probabilities, true_labels

    def fetch_additional_samples(self, predictions_probabilities: np.ndarray, true_labels: np.ndarray) -> tuple:
        """
        Fetch additional samples if the number of predictions and labels do not match.

        Args:
        - predictions_probabilities (np.ndarray): Predictions probabilities.
        - true_labels (np.ndarray): True labels.

        Returns:
        - tuple: (predictions_probabilities, true_labels)
        """
        try:
            n_additional_samples = len(predictions_probabilities) - len(true_labels)
            if n_additional_samples <= 0:
                return predictions_probabilities, true_labels
            query = f"SELECT predictions_probabilities, true_labels FROM predictions LIMIT {n_additional_samples};"
            cursor = self.db_manager.connection.cursor()
            cursor.execute(query)
            additional_rows = cursor.fetchall()
            if not additional_rows:
                self.logger.warning("No additional samples found in the database.")
                return predictions_probabilities, true_labels
            additional_predictions_probabilities = np.array([eval(row[0]) for row in additional_rows])
            additional_true_labels = np.array([int(row[1]) for row in additional_rows])
            if additional_predictions_probabilities.ndim == 1:
                additional_predictions_probabilities = np.array([additional_predictions_probabilities])
            elif additional_predictions_probabilities.ndim == 3:
                additional_predictions_probabilities = additional_predictions_probabilities.reshape(-1, additional_predictions_probabilities.shape[-1])
            predictions_probabilities = np.concatenate((predictions_probabilities, additional_predictions_probabilities), axis=0)
            true_labels = np.concatenate((true_labels, additional_true_labels), axis=0)
            self.update_database(predictions_probabilities, true_labels)
            return predictions_probabilities, true_labels
        except Exception as e:
            self.logger.error(f"Error fetching additional samples from the database: {e}")
            return predictions_probabilities, true_labels

    def update_database(self, predictions_probabilities: np.ndarray, true_labels: np.ndarray) -> None:
        """
        Update the database with new predictions and labels.

        Args:
        - predictions_probabilities (np.ndarray): Predictions probabilities.
        - true_labels (np.ndarray): True labels.
        """
        try:
            query = "SELECT id, predictions_probabilities, true_labels FROM predictions LIMIT 100;"
            cursor = self.db_manager.connection.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            if not rows:
                self.logger.warning("No rows found in the database to update.")
                return
            for i, row in enumerate(rows):
                new_predictions_probabilities = predictions_probabilities[i].tolist()
                new_true_label = true_labels[i]
                update_query = f"UPDATE predictions SET predictions_probabilities = '{new_predictions_probabilities}', true_labels = {new_true_label} WHERE id = {row[0]};"
                cursor.execute(update_query)
            self.db_manager.connection.commit()
            self.logger.info("Database updated with new data.")
        except Exception as e:
            self.logger.error(f"Error updating the database: {e}")

# Example Usage
if __name__ == "__main__":
    config = {'uncertainty_threshold': 0.3, 'calibration_method': 'histogram'}
    uq = UncertaintyQuantification(config)
    
    while True:
        model_predictions, ensemble_predictions, data_variance, predictions_probabilities, true_labels = uq.fetch_and_correct_data()
        if predictions_probabilities.ndim == 1:
            predictions_probabilities = np.array([predictions_probabilities])
        elif predictions_probabilities.ndim == 3:
            predictions_probabilities = predictions_probabilities.reshape(-1, predictions_probabilities.shape[-1])
        
        # Ensure true_labels is a 1D array
        if true_labels.ndim != 1:
            true_labels = true_labels.flatten()
        
        # Estimate uncertainties
        epistemic_uncertainty = uq.estimate_epistemic(ensemble_predictions)
        aleatoric_uncertainty = uq.handle_aleatoric(data_variance)
        
        # Calibrate confidence
        confidence_level, _ = uq.calibrate_confidence(predictions_probabilities, true_labels)
        
        if confidence_level is not None:
            break
        else:
            uq.logger.warning("Calibration failed. Fetching additional samples and retrying.")
            predictions_probabilities, true_labels = uq.fetch_additional_samples(predictions_probabilities, true_labels)
    
    # Make decision with uncertainty
    decision_outcome = uq.make_decision_with_uncertainty(model_predictions)
    
    # Log uncertainty metrics
    uq.log_uncertainty_metrics()
    
    # Print results
    print("Epistemic Uncertainty:", epistemic_uncertainty)
    print("Aleatoric Uncertainty:", aleatoric_uncertainty)
    print("Confidence Level:", confidence_level)
    print("Decision Outcome:", decision_outcome)
    
    # Close the database connection
    uq.db_manager.close_connection()
