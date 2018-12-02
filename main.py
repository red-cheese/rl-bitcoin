

import argparse
import data_utils
import sys
from strats import momentum, rand


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', help='Frequency of data: MIN (minute) or H (hour)')
    args = parser.parse_args(sys.argv[1:])

    frequency = args.frequency if args.frequency else 'H'
    data = data_utils.load_aggregates(frequency)
    train_data, test_data = data, data  # TODO Split for strategies when we need training

    # 0 - Random trader (to check that its profit is around 0).
    rand_trader = rand.RandomTrader()

    # 1 - Momentum.
    momentum_trader = momentum.MomentumTrader()

    # Fit models when needed.

    # Do the first trade.
    start_date, price = test_data[0]
    momentum_trader.trade(start_date, price)
    rand_trader.trade(start_date, price)

    for timestamp, price in test_data[1:-1]:
        momentum_trader.trade(timestamp, price)
        rand_trader.trade(timestamp, price)

    # Close the position.
    end_date, price = test_data[-1]
    momentum_trader.close(end_date, price)
    rand_trader.close(end_date, price)

    print()
    print('=============')
    print('Start date:', start_date)
    print('End date:', end_date)
    print('Frequency:', frequency)
    print('Profits:')
    print('\t{}'.format(momentum_trader.name), momentum_trader.profit)
    print('\t{}'.format(rand_trader.name), rand_trader.profit)


if __name__ == '__main__':
    main()
