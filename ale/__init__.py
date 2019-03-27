from . import drivers
from .drivers import load, loads

# init logger
import logging

LOG_FORMAT = '%(name)s||%(asctime)-15s||%(levelname)s||%(message)s'
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger('ALE')
logger.setLevel(logging.DEBUG)
