import artm
import logging
import os


# change log style
lc = artm.messages.ConfigureLoggingArgs()
lc.minloglevel = 3
lib = artm.wrapper.LibArtm(logging_config=lc)


logs_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'logs',
)

os.makedirs(logs_folder, exist_ok=True)


# Creating logger
logger = logging.getLogger()
logger.setLevel(logging.WARNING)

# Creating file handler
fh = logging.FileHandler(os.path.join(logs_folder, 'logs.txt'))
fh.setLevel(logging.WARNING)

# Creating formatter
formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s',
    '%Y-%m-%d %H:%M:%S'
)

# Adding formatter to file handler
fh.setFormatter(formatter)

# Adding file handler to logger
logger.addHandler(fh)
