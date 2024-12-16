# parallel bayesian
import warnings
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import logging

# Suppress specific warnings from scikit-learn
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

class HyperparameterOptimization:
    def __init__(self, max_iterations=10, tolerance=0.05):
        self.cache = {}
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def parallel_bayesian_optimization(self, X_train, y_train, X_val, y_val, hyperparameters):
        best_result = None
        previous_score = float('-inf')
        iterations = 0
        updated_hyperparameters = hyperparameters.copy()

        with ThreadPoolExecutor() as executor:
            while iterations < self.max_iterations:
                futures = [
                    executor.submit(
                        self._execute_bayesian_optimization,
                        X_train, y_train, X_val, y_val,
                        self._perturb_hyperparameters(updated_hyperparameters)
                    ) for _ in range(3)
                ]

                current_results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result:
                            current_results.append(result)
                    except Exception as e:
                        self.logger.error(f"Optimization iteration failed: {e}")

                if not current_results:
                    self.logger.warning("No valid results found in this iteration. Stopping.")
                    break

                best_current_result = max(current_results, key=lambda x: x[1])
                improvement = abs(best_current_result[1] - previous_score)

                if improvement < self.tolerance:
                    self.logger.info(f"Early stopping at iteration {iterations + 1}")
                    break

                previous_score = best_current_result[1]
                best_result = best_current_result
                updated_hyperparameters = best_current_result[0]

                iterations += 1
                self.logger.info(f"Iteration {iterations}: Best Score = {previous_score}")

        if best_result is None:
            self.logger.warning("No valid results after all iterations.")
            return {"status": "No valid results"}
        
        return best_result

    def _execute_bayesian_optimization(self, X_train, y_train, X_val, y_val, hyperparameters):
        kernel_constant = hyperparameters.get("kernel_constant", 1.0)
        kernel_length_scale = max(hyperparameters.get("kernel_length_scale", 1.0), 1e-5)

        kernel = C(kernel_constant) * RBF(kernel_length_scale)
        gp_model = GaussianProcessRegressor(kernel=kernel)

        try:
            gp_model.fit(X_train, y_train)
            score = gp_model.score(X_val, y_val)
            quality_score = self._compute_quality_score(gp_model, X_val, y_val)

            return (
                {"kernel_constant": kernel_constant, "kernel_length_scale": kernel_length_scale},
                score,
                quality_score
            )
        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return None

    def _perturb_hyperparameters(self, hyperparameters):
        perturbed = hyperparameters.copy()
        perturbed['kernel_constant'] *= np.random.uniform(0.9, 1.1)
        perturbed['kernel_length_scale'] *= np.random.uniform(0.9, 1.1)
        return perturbed

    def _compute_quality_score(self, model, X_val, y_val):
        try:
            score = model.score(X_val, y_val)
            predictions = model.predict(X_val)
            mse = np.mean((predictions - y_val) ** 2)
            quality_score = 0.7 * score - 0.3 * mse
            return quality_score
        except Exception as e:
            self.logger.error(f"Quality score computation error: {e}")
            return -np.inf

# Example usage
def main():
    np.random.seed(42)
    X_train = np.random.rand(100, 5)
    y_train = np.random.rand(100)
    X_val = np.random.rand(20, 5)
    y_val = np.random.rand(20)

    hyperparameters = {
        "kernel_constant": 1.0,
        "kernel_length_scale": 1.0
    }

    optimizer = HyperparameterOptimization(max_iterations=10, tolerance=0.01)
    best_result = optimizer.parallel_bayesian_optimization(X_train, y_train, X_val, y_val, hyperparameters)
    
    print("Best Result:", best_result)

# Directly calling the main function
main()
