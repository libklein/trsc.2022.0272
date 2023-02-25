# coding=utf-8
import logging
import logging.config

from pathlib import Path
from typing import Optional

from yaml import safe_load as load_yaml
from os import getenv as get_environment_variable
from sys import stderr

def create_file_handler(name: str, is_basename=False):
    LOG_DIR = Path('.')
    if is_basename:
        file_handler = logging.FileHandler(LOG_DIR / name, mode='w', delay=True)
    else:
        file_handler = logging.FileHandler(str(LOG_DIR / name) + '.log', mode='w', delay=True)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt='[%(asctime)s][%(levelname)s]: %(message)s', datefmt='%d.%m.%Y %H:%M:%S')
    file_handler.setFormatter(formatter)

    return file_handler


def setup_default_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def configure(logging_config_file: Optional[Path] = Path('./logging.yaml'), environment_var: Optional[str] = 'EVSPNL_LOGGER_CONFIG'):
    # Read configuration
    if logging_config_file is None or not logging_config_file.exists():
        logging_config_file = Path(get_environment_variable(environment_var), './logging.yaml')

    if logging_config_file.exists():
        try:
            with open(logging_config_file) as logging_config_file_handle:
                logging_config_dict = load_yaml(logging_config_file_handle.read())
                logging.config.dictConfig(logging_config_dict)
                return
        except Exception as e:
            print(e, file=stderr)
            print('Coult not read log! Falling back to default logging config.', file=stderr)

    logging.basicConfig()
    logging.warning(f'Could not read logging config. Falling back to default.')