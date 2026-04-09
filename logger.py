import os
import logging


def createLog(verb_level, toFile=False, path="output"):
    """ """
    logger = logging.getLogger(__name__)

    if verb_level == 0:
        logger.setLevel(logging.INFO)
    elif verb_level == 1:
        logger.setLevel(logging.DEBUG)
    else:
        raise ValueError("Invalid verbosity level. Use 0 for INFO, \
                          1 for DEBUG.")

    formatter = logging.Formatter('[%(levelname)s] - %(message)s')

    if toFile:
        fpath = os.path.join(path, "shells.log")
        file_handler = logging.FileHandler(fpath)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger

