import os
import logging
from logging.handlers import RotatingFileHandler
import datetime

if not os.path.exists('logs'): # if logs directory does not exist, create it
    os.makedirs('logs')

def setup_logger(name='SAMGUI', level=logging.INFO, 
                max_file_size=10*1024*1024, backup_count=2):
    """
    Setup and return a logger instance.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        max_file_size: Max size of log file before rotating in bytes (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if logger.handlers:
        logger.handlers.clear()
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create file handler with rotating logs
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join('logs', f'{name}_{timestamp}.log')
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=max_file_size, 
        backupCount=backup_count
    )
    file_handler.setFormatter(formatter)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

if __name__ == '__main__':
    logger = setup_logger()
    logger.info('Test log message')