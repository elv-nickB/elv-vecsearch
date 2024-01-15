import logging
import time
from typing import Tuple
import subprocess

logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

class timeit:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        logger.info(f'{self.message}')
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logger.info(f'Finished {self.message}...')
        logger.info(f"Elapsed time: {self.interval:.4f} seconds")

# Run a command and return it's stdout
# Throws error if command fails
def run_command(cmd: str) -> Tuple[str, str]:
    logger.debug(f"Running command\n{cmd}\n")
    res = subprocess.run(cmd.split(' '), capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Failed to run command\n{cmd}\nstderr=...\n{res.stderr}\nstdout=...\n {res.stdout}")
    return res.stdout, res.stderr