

import matplotlib.pyplot as plt
import numpy as np
import os


PLOTS_DIR = './plots/'  # Path from main.py.


def _create_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)


def plot_episode_profits(model_name, pretty_model_name, episode_profits, optimal_profit, ylabel='Profit$'):
    """Total profit wrt episode (or other metric, as per ylabel)."""

    dir_path = os.path.join(PLOTS_DIR, model_name)
    _create_dir(dir_path)

    num_episodes = len(episode_profits)
    plt.title('{} per episode'.format(ylabel))
    plt.xlabel('Episode')
    plt.ylabel(ylabel)
    plt.plot([optimal_profit] * num_episodes, color='r', label='Optimal')
    plt.plot(episode_profits, label=pretty_model_name)
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':')
    plt.grid(which='major', linestyle=':')
    plt.savefig(os.path.join(dir_path, 'episode_{}.png'.format(ylabel.lower())))
    # plt.show()
    plt.gcf().clear()


def plot_step_trades(model_name, pretty_model_name, episode_idx, all_prices, all_trades):
    """BTC prices with buy/sell/stay decisions."""

    assert len(all_prices) == len(all_trades)
    all_prices = np.asarray(all_prices)
    all_trades = np.asarray(all_trades)

    dir_path = os.path.join(PLOTS_DIR, model_name)
    _create_dir(dir_path)

    # Plot the last 100 prices/trades.
    num = 100
    prices = all_prices[-num:]
    trades = all_trades[-num:]

    plt.title('Last {} prices and actions'.format(num))
    plt.xlabel('Step')
    plt.ylabel('BTC price$')
    plt.plot(prices)
    plt.scatter(np.arange(num)[trades == 1], prices[trades == 1], color='g', label='Buy')
    plt.scatter(np.arange(num)[trades == -1], prices[trades == -1], color='r', label='Sell')
    plt.scatter(np.arange(num)[trades == 0], prices[trades == 0], color='orange', label='Do nothing')
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':')
    plt.grid(which='major', linestyle=':')
    plt.savefig(os.path.join(dir_path, 'episode{}_trades.png'.format(episode_idx)))
    # plt.show()
    plt.gcf().clear()
