

import argparse
import data_utils
import plot_utils
import sys

import strats.momentum
import strats.rand
import strats.rl.baseline
import strats.rl.mc
import strats.rl.q
import strats.rl.setup


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', help='Frequency of data: MIN (minute) or H (hour)')
    args = parser.parse_args(sys.argv[1:])
    frequency = args.frequency if args.frequency else 'H'

    data = data_utils.load_aggregates(frequency)
    start_date, end_date = data[0][0], data[-1][0]

    env = strats.rl.setup.Environment(alpha=0.5, simple_returns=False)

    mc_1st_model_name, mc_1st_episode_rewards, mc_1st_episode_profits = strats.rl.mc.run(env, data, mode='First_Visit')
    mc_every_model_name, mc_every_episode_rewards, mc_every_episode_profits = strats.rl.mc.run(env, data,
                                                                                               mode='Every_Visit')
    q_model_name, q_episode_rewards, q_episode_profits = strats.rl.q.run(env, data)
    baseline_model_name, baseline_episode_rewards, baseline_episode_profits = strats.rl.baseline.run(env, data)

    if env.simple_returns:
        # Plot all episode rewards to compare.
        plot_utils.plot_episode_results(
            env.name,
            [q_model_name, baseline_model_name, mc_1st_model_name, mc_every_model_name],
            [q_episode_rewards, baseline_episode_rewards, mc_1st_episode_rewards, mc_every_episode_rewards],
            ylabel='Reward')

    # Plot all episode profits to compare.
    plot_utils.plot_episode_results(
        env.name,
        [q_model_name, baseline_model_name, mc_1st_model_name, mc_every_model_name],
        [q_episode_profits, baseline_episode_profits, mc_1st_episode_profits, mc_every_episode_profits])

    print()
    print('=============')
    print('Start date:', start_date)
    print('End date:', end_date)
    print('Frequency:', frequency)


if __name__ == '__main__':
    main()
