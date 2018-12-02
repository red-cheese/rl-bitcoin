

from strats import base
import random
random.seed(1)


class RandomTrader(base.BaseTrader):

    @property
    def name(self):
        return 'RandomUniform ( min=-2, max=2, min_pos={}, max_pos={} )'.format(
            self._MIN_POSITION, self._MAX_POSITION)

    def _forecast(self, timestamp, price):
        return random.uniform(-2, 2)
