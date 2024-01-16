import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class timeit:
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        logging.info(f'{self.message}')
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        logging.info(f'Finished {self.message}...')
        logging.info(f"Elapsed time: {self.interval:.4f} seconds")