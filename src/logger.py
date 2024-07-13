import logging
import os
from datetime import datetime

timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')# simple yeh time stamp format ha

log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)# simple yeh logs folder create karega


log_filename = f"{timestamp}.log"
log_file_path = os.path.join(log_dir, log_filename)

# Configure the logging settings
logging.basicConfig(
    filename=log_file_path,
    format="%(asctime)s - %(name)s - [Line: %(lineno)d] - %(levelname)s - %(message)s",
    level=logging.INFO
)
