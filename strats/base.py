

class BaseTrader:

    _MIN_POSITION = -20  # In Bitcoin.
    _MAX_POSITION = 20  # In Bitcoin.

    def __init__(self):
        self._position = 0.
        self._profit = 0.

    @property
    def description(self):
        raise NotImplementedError

    @property
    def profit(self):
        return round(self._profit, 2)

    def _forecast(self, timestamp, price):
        """Decide whether to buy, sell, or stay where we are -
        for the next time slice.

        Returns: < 0 (sell/short), 0 (stay), or > 0 (buy)."""

        raise NotImplementedError

    def trade(self, timestamp, price):
        """Issue a trade for the next time slice, after observing
        the current time slice.

        Assumes infinite borrow (i.e. we always can buy/sell),
        but applies position limits."""

        print("{} TRADE".format(timestamp))

        forecast = self._forecast(timestamp, price)
        print("{} FORECAST: {}".format(timestamp, forecast))
        print("{} CURRENT POSITION: {}".format(timestamp, self._position))

        if forecast > 0:
            if self._position < self._MAX_POSITION:
                self._position += forecast
                self._profit -= (forecast * price)
            else:
                print("{} WARN: Can't long from MAX_POSITION".format(timestamp))

        elif forecast < 0:
            if self._position > self._MIN_POSITION:
                self._position += forecast
                self._profit -= (forecast * price)  # forecast < 0.
            else:
                print("{} WARN: Can't short from MIN_POSITION".format(timestamp))

        print("{} NEW POSITION: {}".format(timestamp, self._position))
        print()

    def close(self, timestamp, price):
        print("{} CLOSE: position {} at price {}".format(timestamp, self._position, price))
        self._profit += self._position * price
