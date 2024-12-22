# logging_config.py
# Updated Dec 22

import logging
from logging.handlers import RotatingFileHandler
import os
import sys

def setup_logging(log_level=logging.INFO, script_name=None):
    """Set up logging configuration and return a logger."""
    # Determine the script's filename without the .py extension
    if script_name is None:
        script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    log_file = f"{script_name}.log"
    log_dir = os.path.join(os.getcwd(), "logs")  # Use a subdirectory 'logs' in the current working directory

    # Check if log directory is writable
    if not os.path.exists(log_dir):
        try:
            os.makedirs(log_dir, exist_ok=True)
            print(f"Created log directory: {log_dir}")
        except PermissionError:
            print(f"No write access to create log directory: {log_dir}. Falling back to console logging.")
            logger = logging.getLogger()
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level)
            return logger
        except Exception as e:
            print(f"Unexpected error while creating log directory: {e}. Falling back to console logging.")
            logger = logging.getLogger()
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(log_level)
            return logger

    log_file = os.path.join(log_dir, log_file)

    # Get the root logger
    logger = logging.getLogger()
    
    # Avoid duplicate handlers
    if not logger.hasHandlers():
        try:
            # Create a rotating file handler
            file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)  # 5 MB per file
            print(f"Created file handler for log file: {log_file}")
        except PermissionError:
            print(f"No write access to log file: {log_file}. Falling back to console logging.")
            file_handler = logging.StreamHandler()
        except Exception as e:
            print(f"Unexpected error while setting up file handler: {e}. Falling back to console logging.")
            file_handler = logging.StreamHandler()

        # Custom formatter with day, date, and time
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Configure console handler with the same formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def main():
    """Main function to run the application."""
    # Call the setup_logging function before logging any messages
    logger = setup_logging()

    # Log application startup
    logger.info("Application started.")

    # Simulate an actual event
    logger.info("Event: Processing operation started.")
    # Insert your operational logic here

if __name__ == "__main__":
    main()
