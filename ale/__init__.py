from . import drivers
from .drivers import load, loads
<<<<<<< HEAD

# init logger
import logging

LOG_FORMAT = '%(name)s||%(asctime)-15s||%(levelname)s||%(message)s'
logging.basicConfig(format=LOG_FORMAT)
logger = logging.getLogger('ALE')
logger.setLevel(logging.DEBUG)
=======
>>>>>>> 895190557c1f11db2a32456a904f249c9aafb422
