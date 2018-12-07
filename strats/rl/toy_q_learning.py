

import matplotlib.pyplot as plt
import numpy as np

import data_utils


EPSILON = 0.01  # E-greedy Q-learning.
NUM_EPISODES = 10
ALPHA = 1.0  # "Learning rate".
GAMMA = 0.8  # Discount factor - no discount.


MIN_POSITION = -1
MAX_POSITION = +1


class A:
    """
    Can have min -1 or max + 1 BTC.

    #A = 5
    """

    SELL_2 = -2
    SELL_1 = -1
    STAY = 0
    BUY_1 = 1
    BUY_2 = 2

    ALL_ACTIONS = [SELL_2, SELL_1, STAY, BUY_1, BUY_2]


class State:
    """
    #S = 9 (3 possible positions x 3 possible return directions: -1, 0, +1)
    """

    def __init__(self, position, future_return):
        self.position = position
        self.future_return = future_return


def is_allowed(s, a):
    return MIN_POSITION <= s.position + a <= MAX_POSITION


def get_allowed_actions(s):
    allowed_actions = np.asarray([a for a in A.ALL_ACTIONS if is_allowed(s, a)], dtype=np.int32)
    assert (allowed_actions == np.sort(allowed_actions)).all()
    return allowed_actions


def main():
    data = data_utils.load_aggregates('H')

    # Preprocess to have returns: -1, 0, 1.
    prices = np.array([price for timestamp, price in data])
    returns = np.sign(prices[1:] - prices[:-1]).astype(np.int32)  # returns[i] = sign(prices[i + 1] - prices[i])
    num_steps = returns.shape[0]
    optimal_profit = (np.fabs(prices[1:] - prices[:-1])).sum()  # Target profit.

    # Init Q(s, a) with zeros.
    # Pairs of (s: State, a: A), where s = (position, future return).
    Q_table = np.zeros(shape=(3, 3, 5))

    for i in range(NUM_EPISODES):
        print('Play episode', i)
        s = State(position=0, future_return=returns[0])
        cash = 0
        balance = 0  # Current balance = cash + position * current price.

        # Repeat for each state of the episode.
        for j in range(num_steps):
            # Choose the best "a" from "s" using current Q.
            possible_as_from_s = get_allowed_actions(s)
            Q_possible_as_from_s = Q_table[s.position + 1, s.future_return + 1, possible_as_from_s + 2]
            if np.random.random() >= EPSILON / (i + 1) or i == NUM_EPISODES - 1:  # Never explore in the last episode.
                a = possible_as_from_s[Q_possible_as_from_s == Q_possible_as_from_s.max()]
                a = np.random.choice(a)
                chose_random_action = False
            else:
                a = np.random.choice(possible_as_from_s)
                chose_random_action = True

            # Take action a. Observe s'.
            new_position = s.position + a
            new_cash = cash - a * prices[j]
            assert -1 <= new_position <= 1

            # We just made a trade => observe reward r based on the next price.
            new_balance = new_cash + new_position * prices[j + 1]
            r = np.sign(new_balance - balance)
            # Observe our new state.
            s_prime = State(position=new_position, future_return=returns[j + 1] if j < num_steps - 1 else None)

            if j < num_steps - 1:
                # Update Q table.
                s_a_idx = (s.position + 1, s.future_return + 1, a + 2)
                max_next_Q = Q_table[s_prime.position + 1, s_prime.future_return + 1, :].max()
                delta_Q = ALPHA * (r + GAMMA * max_next_Q - Q_table[s_a_idx])
                if i != NUM_EPISODES - 1:  # TODO ???
                    Q_table[s_a_idx] += delta_Q
            else:
                delta_Q = None  # No updates to Q in the terminal state.

            # Some debug logging.
            if not chose_random_action and s.future_return == -1 and NUM_EPISODES == 1:
                print('Step', j, '/ s.position', s.position, '/ s.future_return', s.future_return,
                      '/ a', a, '/ s_prime.position', s_prime.position,
                      '/ reward', r, '/ delta Q', delta_Q, '/ old balance', balance, '/ new balance', new_balance)

            # Check Q table. TODO?

            balance = new_balance
            cash = new_cash
            s = s_prime

        print('End episode', i)
        assert balance == cash + s.position * prices[-1]
        print('Total episode profit:', balance)
        print('Target profit:', optimal_profit)
        print('Q table:')
        for x in range(Q_table.shape[0]):  # position
            for y in range(Q_table.shape[1]):  # future return
                for z in range(Q_table.shape[2]):  # action
                    print('\t\tpos {} / future return {} / action {}'
                          .format(x - 1, y - 1, z), Q_table[x, y, z])
                print()
            print()
        print('================================')
        print()

        # Plot total profit wrt episode
        # Plot BTC price with Up/Down decisions

        # Print warning when we make a wrong action: Compare optimal matrix and final/wrt to step Q-value matrix


if __name__ == '__main__':
    main()
