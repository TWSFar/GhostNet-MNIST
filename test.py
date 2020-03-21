import logging

logging.basicConfig(
            format='[%(asctime)s %(levelname)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG)
f_handler = logging.FileHandler('error.log', mode='w')
logger = logging.getLogger(__name__)
logger.addHandler(f_handler)

logger.info('werewrewrwe')
logger.info('wasdgdsrewrwe')
logger.info('wgasgwrewrwe')
logger.info('gasd')