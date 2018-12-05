

import enum
import numpy as np

import data_utils


NUM_EPISODES = 10
ALPHA = 0.1  # "Learning rate".
GAMMA = 1.  # Discount factor - no discount.


class A(enum.Enum):
    """
    Can have min -1 or max + 1 BTC.

    #A = 5
    """

    STAY = 0
    BUY_1 = 1
    BUY_2 = 2
    SELL_1 = 3
    SELL_2 = 4


class State:
    """
    #S = 9 (3 possible positions x 3 possible return directions: -1, 0, +1)
    """

    def __init__(self, position, next_return_direction):
        self.position = position
        self.next_return_direction = next_return_direction  # TODO Rename to future_return


def main():
    data = data_utils.load_aggregates('H')

    # Preprocess to have returns -1, 0, 1.
    returns = np.zeros(shape=(len(data) - 1,), dtype=np.int32)
    prices = np.array([price for timestamp, price in data])
    returns[:] = np.where(prices[1:] > prices[:-1], 1, returns)
    returns[:] = np.where(prices[1:] < prices[:-1], -1, returns)

    num_steps = returns.shape[0]

    # Init Q(s, a) arbitrarily.
    # Pairs of (s, a), where s = (position, next return).
    # Q_table = np.random.normal(loc=0, scale=0.001, size=45).reshape((3, 3, 5))
    Q_table = np.zeros(shape=(3, 3, 5))

    for i in range(NUM_EPISODES):
        print('Play episode', i)
        s = State(position=0, next_return_direction=returns[0])
        cash = 0
        balance = 0  # Current balance = cash + position * current price.

        # Repeat for each state of the episode.
        for j in range(num_steps - 1):  # TODO num_steps or (num_steps - 1) ?
            # Choose the best "a" from "s" using current Q.
            possible_as_from_s = Q_table[s.position + 1, s.next_return_direction + 1, :]
            a = np.where(possible_as_from_s == possible_as_from_s.max())
            a = np.random.choice(a[0])

            # Take action a. Observe s'.
            if a == A.STAY.value:
                new_position = s.position
                new_cash = cash

            elif a == A.BUY_1.value:
                actual_buy = min(1, 1 - s.position)
                new_position = s.position + actual_buy
                new_cash = cash - actual_buy * prices[j]

            elif a == A.BUY_2.value:
                actual_buy = min(2, 1 - s.position)
                new_position = s.position + actual_buy
                new_cash = cash - actual_buy * prices[j]

            elif a == A.SELL_1.value:
                actual_sell = min(1, s.position + 1)
                new_position = s.position - actual_sell
                new_cash = cash + actual_sell * prices[j]

            elif a == A.SELL_2.value:
                actual_sell = min(2, s.position + 1)
                new_position = s.position - actual_sell
                new_cash = cash + actual_sell * prices[j]

            else:
                raise RuntimeError('Unknown a: {}'.format(a))

            assert -1 <= new_position <= 1
            assert -1 <= s.position <= 1

            # Observe r and construct s_prime.
            new_balance = new_cash + new_position * prices[j + 1]  # We are already at the new state -> reward.
            r = new_balance - balance
            s_prime = State(position=new_position, next_return_direction=returns[j + 1])

            balance = new_balance
            cash = new_cash

            # Update Q table.
            sa = (s.position + 1, s.next_return_direction + 1, a)
            max_next_Q = Q_table[s_prime.position + 1, s_prime.next_return_direction + 1, :].max()
            Q_table[sa] += ALPHA * (r + GAMMA * max_next_Q - Q_table[sa])

            s = s_prime

        print('End episode', i)
        # Print total profit.
        print('Total profit:', cash + s.position * prices[-1])
        print('Q:')
        print()
        for x in range(Q_table.shape[0]):  # position
            for y in range(Q_table.shape[1]):  # next return
                for z in range(Q_table.shape[2]):  # action
                    print('pos {} / future return sign {} / action {}'.format(x - 1, y - 1, z), Q_table[x, y, z])
            print()
        print('Optimal profit:', (np.fabs(prices[1:] - prices[:-1])).sum())
        print()
        print()


if __name__ == '__main__':
    main()
