

import matplotlib.pyplot as plt
import numpy as np
import os


PLOTS_DIR = './plots/'  # Path from main.py.


def plot_episode_profits(model_name, episode_profits, optimal_profit):
    """Total profit wrt episode."""

    num_episodes = len(episode_profits)
    plt.title('Total profit')
    plt.xlabel('Episode')
    plt.ylabel('Profit$')
    plt.plot([optimal_profit] * num_episodes, color='r', label='Optimal')
    plt.plot(episode_profits, label='Q-learning')
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':')
    plt.grid(which='major', linestyle=':')
    plt.savefig(os.path.join(PLOTS_DIR, model_name, 'episode_profits.png'))
    # plt.show()


def plot_step_trades(model_name, episode_idx, all_prices, all_trades):
    """BTC prices with buy/sell/stay decisions."""

    assert len(all_prices) == len(all_trades) + 1
    all_prices = np.asarray(all_prices)
    all_trades = np.asarray(all_trades)

    # Plot the last 500 prices/trades.
    num = 500
    prices = all_prices[-(num + 1):-1]
    trades = all_trades[-num:]

    plt.title('Last {} prices and actions'.format(num))
    plt.xlabel('Step')
    plt.ylabel('BTC price$')
    plt.plot(prices)
    plt.scatter(np.arange(num)[trades == 1], prices[trades == 1], color='g', label='Buy')
    plt.scatter(np.arange(num)[trades == -1], prices[trades == -1], color='r', label='Sell')
    plt.legend(loc='lower right')
    plt.minorticks_on()
    plt.grid(which='minor', linestyle=':')
    plt.grid(which='major', linestyle=':')
    plt.savefig(os.path.join(PLOTS_DIR, model_name, 'episode{}_trades.png'.format(episode_idx)))
    # plt.show()
