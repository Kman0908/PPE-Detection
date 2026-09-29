import os
import logging
from datetime import datetime

log_file = f'{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log'

log_folder = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_folder, exist_ok = True)

LOG = os.path.join(log_folder, log_file)

logging.basicConfig(
    filename = LOG,
    format = '[%(asctime)s] %(levelname)s [%(filename)s:%(lineno)d] %(message)s',
    level = logging.INFO
)

logging.getLogger(__name__)