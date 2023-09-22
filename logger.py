import logging
import sys
def init_logger(logger_path):
    logging.basicConfig(filename=logger_path, 
                    format='%(asctime)s [%(levelname)s] %(message)s' ,
                    datefmt='%Y-%m-%d %H:%M:%S',
                    force=True,   
                    level=logging.DEBUG)

    # Send log to stdout in addition to the log file:
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
    return logging
