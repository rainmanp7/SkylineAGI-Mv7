# logging_config.py start 

import logging
from logging.handlers import RotatingFileHandler
import os
from threading import Event  # For async_ops_completed simulation

# Create a global event for asynchronous operation completion simulation
async_ops_completed = Event()

def setup_logging(log_level=logging.INFO):
    """Set up logging configuration."""
    log_file = "app.log"
    log_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else os.getcwd()

    # Check if log directory is writable
    if not os.access(log_dir, os.W_OK):
        log_dir = os.path.expanduser("~/logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "app.log")
        print(f"Falling back to log directory: {log_dir}")

    # Get the root logger
    logger = logging.getLogger()
    
    # Avoid duplicate handlers
    if not logger.hasHandlers():
        try:
            # Create a rotating file handler
            file_handler = RotatingFileHandler(log_file, maxBytes=5 * 1024 * 1024, backupCount=5)  # 5 MB per file
        except PermissionError:
            print(f"No write access to log file: {log_file}. Falling back to console logging.")
            file_handler = logging.StreamHandler()
        except Exception as e:
            print(f"Unexpected error while setting up file handler: {e}. Falling back to console logging.")
            file_handler = logging.StreamHandler()

        # Custom formatter with day, date, and time
        formatter = logging.Formatter('%(message)s. %(asctime)s', datefmt='%a %b %d %H:%M:%S %Y')
        file_handler.setFormatter(formatter)

        # Configure console handler with the same formatter
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        # Configure logger
        logger.setLevel(log_level)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

def main():
    """Main function to run the application."""
    # Call the setup_logging function before logging any messages
    setup_logging()

    # Get the logger
    logger = logging.getLogger(__name__)

    logger.info("Application started.")  # Log application startup

    # Simulate an actual event
    logger.info("Event: Processing operation started.")
    # Insert your operational logic here

if __name__ == "__main__":
    main()

# end of logging config.
