import logging
import os


logs_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'logs'
)

os.makedirs(logs_folder, exist_ok=True)


# Creating logger
logger = logging.getLogger('main')
logger.setLevel(logging.DEBUG)

# Creating file handler
fh = logging.FileHandler(os.path.join(logs_folder, 'logs.txt'))
fh.setLevel(logging.DEBUG)

# Creating formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    '%Y-%m-%d %H:%M:%S'
)

# Adding formatter to file handler
fh.setFormatter(formatter)

# Adding file handler to logger
logger.addHandler(fh)
