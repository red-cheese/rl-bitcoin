

import argparse
import data_utils
import sys

import strats.momentum
import strats.rand
import strats.rl.toy_q_learning


def momentum(data):
    momentum_trader = strats.momentum.MomentumTrader()

    for timestamp, price in data[:-1]:
        momentum_trader.trade(timestamp, price)

    # Close the position.
    end_date, price = data[-1]
    momentum_trader.close(end_date, price)

    print()
    print('=============')
    print('Profits:')
    print('\t{}'.format(momentum_trader.name), momentum_trader.profit)


def rand(data):
    rand_trader = strats.rand.RandomTrader()

    for timestamp, price in data[:-1]:
        rand_trader.trade(timestamp, price)

    # Close the position.
    end_date, price = data[-1]
    rand_trader.close(end_date, price)

    print()
    print('=============')
    print('Profits:')
    print('\t{}'.format(rand_trader.name), rand_trader.profit)


def toy_q(data):
    strats.rl.toy_q_learning.run(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', help='Frequency of data: MIN (minute) or H (hour)')
    parser.add_argument('--alg', help='Trading algorithm')
    args = parser.parse_args(sys.argv[1:])

    frequency = args.frequency if args.frequency else 'H'
    alg = args.alg if args.alg else 'toy_q'

    data = data_utils.load_aggregates(frequency)
    start_date, end_date = data[0][0], data[-1][0]

    if alg == 'momentum':
        momentum(data)
    elif alg == 'random':
        rand(data)
    elif alg == 'toy_q':
        toy_q(data)
    else:
        raise ValueError("Unknown algorithm: '{}'".format(alg))

    print()
    print('=============')
    print('Start date:', start_date)
    print('End date:', end_date)
    print('Frequency:', frequency)
    print('Strategy:', alg)


if __name__ == '__main__':
    main()
