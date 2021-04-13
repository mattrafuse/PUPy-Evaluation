import time
import math
from os import path

PARSED_PATH = 'datasets/mdc/parsed'
BT_USER_PATH = f'{PARSED_PATH}/bt_by_user'

class Ticker():
    def __init__(self, size, accuracy=3):
        super().__init__()

        self.size = size
        self.index = 0
        self.pct = 0
        self.start_time = time.time()
        self.accuracy = accuracy

    def tick(self, preamble='', postamble=''):
        self.index += 1
        if int((self.index / self.size) * 100 * (10 ** self.accuracy)) > self.pct:
            self.pct = int((self.index / self.size) * 100 * (10 ** self.accuracy))
            val = self.index / self.size

            estimate = (self.size - self.index) * ((time.time() - self.start_time) / self.index)
            estimate = math.ceil(estimate / 60)

            if self.accuracy >= 3:
                print(f'{preamble}{val:.3%} (~{estimate}m remaining) {postamble}                   \r', end="")
            elif self.accuracy >= 2:
                print(f'{preamble}{val:.2%} (~{estimate}m remaining) {postamble}                   \r', end="")
            else:
                print(f'{preamble}{val:.1%} (~{estimate}m remaining) {postamble}                   \r', end="")


def get_file_size(filename):
        return path.getsize(path.join(BT_USER_PATH, filename))