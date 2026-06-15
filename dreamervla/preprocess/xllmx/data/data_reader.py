import logging
import time
from io import BytesIO

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
logger = logging.getLogger(__name__)


def read_general(path) -> str | BytesIO:
    if "s3://" in path:
        init_ceph_client_if_needed()
        file_bytes = BytesIO(client.get(path))
        return file_bytes
    else:
        return path


def init_ceph_client_if_needed():
    global client
    if client is None:
        logger.info("initializing ceph client ...")
        st = time.time()
        from petrel_client.client import Client  # noqa

        client = Client("/path/to/petreloss.conf")
        ed = time.time()
        logger.info(f"initialize client cost {ed - st:.2f} s")


client = None
