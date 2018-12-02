

import numpy as np
from strats import base


class MomentumTrader(base.BaseTrader):
    """ROC momentum from http://www.quantsportal.com/momentum-strategies/."""

    _WINDOW = 8
    # _WINDOW = 30  # For MIN.

    def __init__(self, window=_WINDOW):
        super(MomentumTrader, self).__init__()

        # Will only issue trades when we have sufficient history of ticks.
        self.__tick = 0

        self.__window = window
        self.__prices = np.zeros(shape=(self.__window,))

    @property
    def name(self):
        return 'ROCMomentum ( window={}, min_pos={}, max_pos={} )'.format(
            self._WINDOW, self._MIN_POSITION, self._MAX_POSITION)

    def _forecast(self, timestamp, price):
        self.__prices[0:-1] = self.__prices[1:]
        self.__prices[-1] = price

        self.__tick += 1
        if self.__tick < self._WINDOW:
            return 0.

        roc = self.__prices[-1] / self.__prices[0] - 1
        return roc if np.isfinite(roc) else 0.
