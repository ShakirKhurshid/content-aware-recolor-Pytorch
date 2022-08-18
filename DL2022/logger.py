import logging 
import sys


def log_to_stdout(logger_name="User logging", level=logging.INFO):
    '''Set up fuction for logging'''
    logging.basicConfig(filename='logname.txt',
                            filemode='a',
                            format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                            datefmt='%H:%M:%S')
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    if not any([isinstance(h, logging.StreamHandler) for h in logger.handlers]):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(handler)

    return logger

