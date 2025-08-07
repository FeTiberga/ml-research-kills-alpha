import os
import logging
import datetime


class Logger(object):
    def __init__(self, log_name="ml-research-kills-alpha"):
        self.log_filename = self.__generate_log_filename(log_name)
        self.log_dir = 'logs'
        os.makedirs(self.log_dir, exist_ok=True)
        self.azure_container = "ml-research-kills-alpha-logs"
        self.azure_key_prefix = "pipeline-logs"

        try:
            self.__configure_logging()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Logging configuration failed: {str(e)}")

    @staticmethod
    def __generate_log_filename(log_name):
        """
        Generate the log filename as
        :return: the log filename
        """
        current_datetime = datetime.datetime.now()
        formatted_date_time = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")
        return "{}_{}.log".format(log_name, formatted_date_time)

    def __configure_logging(self):
        """
        Configure logging
        """
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, self.log_filename)),
                logging.StreamHandler(),
            ],
        )

    @staticmethod
    def info(msg):
        logging.info(msg)

    @staticmethod
    def warn(msg):
        logging.warning(msg)

    @staticmethod
    def error(msg):
        logging.error(msg)

    @staticmethod
    def debug(msg):
        logging.debug(msg)

    @staticmethod
    def small_banner(msg):
        message_length = len(msg)
        num_asterisks = 40
        asterisks_on_each_side = (num_asterisks - message_length - 2) // 2

        banner = f"{'*' * asterisks_on_each_side} {msg.upper()} {'*' * asterisks_on_each_side}"

        logging.info(banner)


logger = Logger()
