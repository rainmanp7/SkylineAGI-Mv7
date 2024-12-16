# complexity.py updated 12/14/2024
# imports corrected.
from enum import Enum
from typing import Any, Callable, List
import numpy as np
import scipy.stats as stats
import logging
import json

class ComplexityRange(Enum):
    """Detailed complexity ranges for nuanced classification"""
    EASY = (1111, 1389)
    SIMP = (1390, 1668)
    NORM = (1669, 1947)
    MODS = (1948, 2226)
    HARD = (2227, 2505)
    PARA = (2506, 2784)
    VICE = (2785, 3063)
    ZETA = (3064, 3342)
    TETR = (3343, 3621)
    EAFV = (3622, 3900)
    SIPO = (3901, 4179)
    NXXM = (4180, 4458)
    MIDS = (4459, 4737)
    HAOD = (4738, 5016)
    PARZ = (5017, 5295)
    VIFF = (5296, 5574)
    ZEXA = (5575, 5853)
    SIP8 = (5854, 6132)
    NXVM = (6133, 6411)
    VIDS = (6412, 6690)
    HA3D = (6691, 6969)
    PFGZ = (6970, 7248)
    VPFF = (7249, 7527)
    Z9XA = (7528, 7806)
    TIPO = (7807, 8085)
    NXNM = (8086, 8364)
    MPD7 = (8365, 9918)

    @staticmethod
    def normalize_to_range(value: float) -> int:
        """Map a raw complexity score to a specific number within the defined ranges."""
        for level in ComplexityRange:
            min_val, max_val = level.value
            if min_val <= value <= max_val:
                return int(round(value))
        # Default to the minimum of the first range
        return ComplexityRange.EASY.value[0]

class ComplexityMetrics:
    """Advanced complexity analysis with multiple computational strategies"""
    
    @staticmethod
    def calculate_entropy(text: str) -> float:
        """Calculate Shannon entropy to measure text complexity and randomness."""
        try:
            ascii_values = [ord(char) for char in text]
            unique, counts = np.unique(ascii_values, return_counts=True)
            probabilities = counts / len(ascii_values)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            return entropy
        except Exception as e:
            logging.error(f"Entropy calculation error: {e}")
            return 0

    @staticmethod
    def calculate_variance_complexity(text: str) -> float:
        """Calculate complexity based on variance of ASCII values."""
        try:
            ascii_values = np.array([ord(char) for char in text])
            variance = np.var(ascii_values)
            skewness = stats.skew(ascii_values)
            kurtosis = stats.kurtosis(ascii_values)
            
            # Combine multiple statistical measures
            complexity_score = (
                variance * 
                (1 + abs(skewness)) * 
                (1 + abs(kurtosis) / 2)
            )
            return complexity_score
        except Exception as e:
            logging.error(f"Variance complexity calculation error: {e}")
            return 0

class AdvancedComplexityFactor:
    """Comprehensive complexity analysis with multiple metrics."""
    
    def __init__(self, custom_complexity_func: Callable[[str], int] = None):
        """Initialize with optional custom complexity calculation."""
        self.custom_complexity_func = custom_complexity_func
        self.metrics_log = []

    def calculate(self, text: str) -> int:
        """Comprehensive complexity calculation."""
        try:
            # Calculate multiple complexity metrics from the text
            metrics = {
                'entropy': ComplexityMetrics.calculate_entropy(text),
                'variance': ComplexityMetrics.calculate_variance_complexity(text),
            }

            # Custom complexity function if provided
            if self.custom_complexity_func:
                custom_complexity = self.custom_complexity_func(text)
                metrics['custom'] = custom_complexity

            # Aggregate complexity metrics
            total_complexity = sum(metrics.values())

            # Scale raw complexity to the defined range
            raw_value = 1111 + ((total_complexity % 1) * (9918 - 1111))

            # Map the normalized value to the appropriate range
            exact_value = ComplexityRange.normalize_to_range(raw_value)

            # Log complexity analysis
            complexity_analysis = {
                'metrics': metrics,
                'total_complexity': total_complexity,
                'raw_value': raw_value,
                'exact_value': exact_value,
            }
            
            self.metrics_log.append(complexity_analysis)

            return exact_value

        except Exception as e:
            logging.error(f"Comprehensive complexity calculation error: {e}")
            return ComplexityRange.EASY.value[0]  # Return default range's minimum value

    def export_complexity_log(self, filename: str = 'complexity_log.json'):
        """Export complexity metrics log to a JSON file."""
        try:
            with open(filename, 'w') as f:
                json.dump(self.metrics_log, f, indent=4)
                
            logging.info(f"Complexity log exported to {filename}")
            
        except Exception as e:
            logging.error(f"Complexity log export error: {e}")

# Example custom complexity function (optional)
def custom_text_complexity(text: str) -> float:
    """Example of a custom complexity calculation based on specific criteria."""
    return len(set(text)) / len(text) if len(text) > 0 else 0.0

# Example usage
def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s: %(message)s'
    )

    # Example input text (you can replace this with any text you want to analyze)
    input_text = "This is an example sentence to analyze the complexity of the text."

    # Initialize the complexity factor with an optional custom function.
    complexity_analyzer = AdvancedComplexityFactor(
        custom_complexity_func=custom_text_complexity
    )

    # Calculate the data complexity level.
    complexity = complexity_analyzer.calculate(input_text)
    
    print(f"Data Complexity Exact Value: {complexity}")

    # Export the complexity log to a JSON file.
    complexity_analyzer.export_complexity_log()

if __name__ == "__main__":
    main()
