# Created on Nov 14, 2024, 9:40 pm
# Uncertainty Code implementation.
# modified 10:43am Dec5

import numpy as np
import scipy.stats as stats
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import logging

class UncertaintyQuantification:
    def __init__(self, config=None):
        """
        Initialize Uncertainty Quantification module.
        
        :param config: Configuration settings for uncertainty handling.
            - Optional keys:
                - `uncertainty_threshold` (float): Default threshold for decision-making.
                - `calibration_method` (str): Method for confidence calibration (e.g., 'histogram', 'isotonic', 'sigmoid').
        """
        self.config = config or {}
        self.epistemic_uncertainty = 0.0
        self.aleatoric_uncertainty = 0.0
        self.confidence_level = 0.0
        self.uncertainty_threshold = self.config.get('uncertainty_threshold', 0.5)
        self.calibration_method = self.config.get('calibration_method', 'histogram')
        
        # Initialize logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def estimate_epistemic(self, model_predictions, ensemble_predictions):
        """
        Estimate epistemic uncertainty using variance across ensemble predictions.
        
        :param model_predictions: Predictions from a single model (for context, not used in calculation).
        :param ensemble_predictions: Predictions from multiple models (2D array, models x outputs).
        :return: Epistemic uncertainty score (float).
        """
        try:
            epistemic_var = np.var(ensemble_predictions, axis=0)  # Variance across models for each output
            self.epistemic_uncertainty = np.mean(epistemic_var)  # Average variance across all outputs
            return self.epistemic_uncertainty
        except Exception as e:
            self.logger.error(f"Error in epistemic uncertainty estimation: {e}")
            return None

    def handle_aleatoric(self, data_variance, method='std_dev'):
        """
        Estimate aleatoric uncertainty based on data variance.
        
        :param data_variance: Variance in the input data.
        :param method: Method to estimate aleatoric uncertainty (std_dev, variance, or a custom float value).
        :return: Aleatoric uncertainty score.
        """
        try:
            if method == 'std_dev':
                self.aleatoric_uncertainty = np.sqrt(data_variance)
            elif method == 'variance':
                self.aleatoric_uncertainty = data_variance
            else:
                self.aleatoric_uncertainty = method  # Assuming a custom value is provided
            return self.aleatoric_uncertainty
        except Exception as e:
            self.logger.error(f"Error in aleatoric uncertainty handling: {e}")
            return None

    def calibrate_confidence(self, predictions, true_labels, method=None):
        """
        Calibrate model confidence using prediction probabilities.
        
        :param predictions: Model prediction probabilities.
        :param true_labels: Actual labels.
        :param method: Calibration method (overrides config if provided). Options: 'histogram', 'isotonic', 'sigmoid'.
        :return: Confidence calibration metric (float) and optionally, calibrated predictions.
        """
        try:
            method = method or self.calibration_method
            if method == 'histogram':
                prob_true, prob_pred = calibration_curve(true_labels, predictions, n_bins=10)
                self.confidence_level = 1 - np.mean(np.abs(prob_true - prob_pred))
                return self.confidence_level, None
            elif method in ['isotonic', 'sigmoid']:
                from sklearn.calibration import CalibratedClassifierCV
                from sklearn.ensemble import RandomForestClassifier
                base_estimator = RandomForestClassifier(n_estimators=100)
                calibrated_clf = CalibratedClassifierCV(base_estimator, method=method, cv=3)
                calibrated_clf.fit(predictions, true_labels)
                calibrated_predictions = calibrated_clf.predict_proba(predictions)
                self.confidence_level = 1 - brier_score_loss(true_labels, calibrated_predictions, pos_label=true_labels.max())
                return self.confidence_level, calibrated_predictions
            else:
                self.logger.warning("Unsupported calibration method. Defaulting to histogram calibration.")
                return self.calibrate_confidence(predictions, true_labels, method='histogram')
        except Exception as e:
            self.logger.error(f"Error in confidence calibration: {e}")
            return None, None

    def make_decision_with_uncertainty(self, predictions, uncertainty_threshold=None):
        """
        Make decisions while considering uncertainty.
        
        :param predictions: Model predictions.
        :param uncertainty_threshold: Optional threshold override (defaults to config value if not provided).
        :return: Decision, uncertainty status, and total uncertainty.
        """
        try:
            uncertainty_threshold = uncertainty_threshold or self.uncertainty_threshold
            total_uncertainty = self.epistemic_uncertainty + self.aleatoric_uncertainty
            
            if total_uncertainty < uncertainty_threshold:
                decision = np.mean(predictions)
                confidence = "High"
            else:
                decision = None  # Defer decision
                confidence = "Low"
            
            return {
                "decision": decision,
                "confidence": confidence,
                "total_uncertainty": total_uncertainty
            }
        except Exception as e:
            self.logger.error(f"Error in decision-making with uncertainty: {e}")
            return None

    def log_uncertainty_metrics(self, log_level='INFO'):
        """
        Log uncertainty metrics for tracking and analysis.
        
        :param log_level: Level at which to log the metrics (e.g., DEBUG, INFO, WARNING).
        """
        uncertainty_log = {
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty,
            "confidence_level": self.confidence_level
        }
        self.logger.log(getattr(logging, log_level), f"Uncertainty Metrics: {uncertainty_log}")
        return uncertainty_log

# Example Usage
if __name__ == "__main__":
    # Configuration
    config = {
        'uncertainty_threshold': 0.3,
        'calibration_method': 'isotonic'
    }
    
    # Initialize Uncertainty Quantification
    uq = UncertaintyQuantification(config)
    
    # Example Predictions and Data
    model_predictions = np.array([0.7, 0.3])  # Single model prediction
    ensemble_predictions = np.array([[0.6, 0.4], [0.8, 0.2], [0.7, 0.3]])  # Ensemble predictions
    data_variance = 0.1  # Example data variance
    predictions_probabilities = np.array([[0.4, 0.6], [0.3, 0.7]])  # Prediction probabilities for calibration
    true_labels = np.array([1, 0])  # Actual labels for calibration
    
    # Estimate Uncertainties
    epistemic_uncertainty = uq.estimate_epistemic(model_predictions, ensemble_predictions)
    aleatoric_uncertainty = uq.handle_aleatoric(data_variance)
    
    # Calibrate Confidence
    confidence_level, _ = uq.calibrate_confidence(predictions_probabilities, true_labels)
    
    # Make Decision with Uncertainty
    decision_outcome = uq.make_decision_with_uncertainty(model_predictions)
    
    # Log Uncertainty Metrics
    uq.log_uncertainty_metrics()
    
    print("Epistemic Uncertainty:", epistemic_uncertainty)
    print("Aleatoric Uncertainty:", aleatoric_uncertainty)
    print("Confidence Level:", confidence_level)
    print("Decision Outcome:", decision_outcome)