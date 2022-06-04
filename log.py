import os
import logging

class Logger(object):
    """docstring for Logger"""
    def __init__(self, log_dir, exp_name):
        super(Logger, self).__init__()
        self.log_dir = log_dir
        self.exp_name = exp_name
        self.train_logger = self.get_logger(os.path.join(self.log_dir,f'{self.exp_name}_train.log'), name="train_logger")
        self.val_logger = self.get_logger(os.path.join(self.log_dir,f'{self.exp_name}_val.log'), name="validation_logger")
        self.test_logger = self.get_logger(os.path.join(self.log_dir,f'{self.exp_name}_test.log'), name="test_logger")

    def get_logger(self, filename, verbosity=1, name=None):
        # logger
        level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

        formatter = logging.Formatter(
            "[%(asctime)s] %(message)s")

        logger = logging.getLogger(name)
        logger.setLevel(level_dict[verbosity])

        if not logger.handlers:
            fh = logging.FileHandler(filename, "a")
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger

    def record_train_log(self, content):
        self.train_logger.info(content)

    def record_val_log(self, content):
        self.val_logger.info(content)

    def record_test_log(self, content):
        self.test_logger.info(content)    