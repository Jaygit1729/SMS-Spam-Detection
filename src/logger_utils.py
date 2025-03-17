import logging
import os

def setup_logger(log_file):
    """
    Sets up a logger that writes logs to a file and prints them to the console.

    """
    logger = logging.getLogger(log_file)  
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers(): 
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        stream_handler = logging.StreamHandler()

        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger


"""
In a nutshell,

Create a Logger → A unique logger is created for each module.

Set Log Level → It is configured to record messages starting from INFO level.

Check for Existing Handlers → To avoid duplicate logs, the code ensures handlers are added only once.

Create Necessary Folders → If the log file's directory doesn’t exist, it is created automatically.

Add File Handler → Logs are written to a file for later reference.

Add Console Handler → Logs are also displayed in the console for real-time monitoring.

Format the Logs → A specific format (timestamp - log level - message) is applied for better readability.

Return the Logger → The fully configured logger is returned, ready to be used for logging messages.

"""
