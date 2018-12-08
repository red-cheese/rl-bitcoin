import collections
import numpy as np
import plot_utils


START_EPSILON = 0.1
NUM_EPISODES = 10
GAMMA = 0.5

MIN_POSITION = -1
MAX_POSITION = +1

NUM_FUTURE_RETURNS = 1


class A:
    """
    Actions from SELL K to BUY K.
    #A = 2K + 1.
    """

    K = 2
    ALL_ACTIONS = [i for i in range(-K, K + 1)]
    SIZE = 2 * K + 1  # 5 atm


class State:
    """
    #S = (MAX_POSITION - MIN_POSITION + 1) terminal states
            + (3 ^ num_future_returns) x (MAX_POSITION - MIN_POSITION + 1 positions) non-terminal states
    """

    SIZE = 4 * (MAX_POSITION - MIN_POSITION + 1)  # 12 atm

    def __init__(self, position, future_returns, is_terminal=False):
        if len(future_returns) != NUM_FUTURE_RETURNS:
            print(len(future_returns))
            raise AssertionError

        future_returns = np.asarray(future_returns)
        assert (-1 <= future_returns <= 1).all()
        if is_terminal:
            assert (future_returns == 0).all()

        self.position = position
        self.future_returns = future_returns
        self.is_terminal = is_terminal


MODEL_NAME = 'vi'


def is_allowed(s, a):
    assert not s.is_terminal
    return MIN_POSITION <= s.position + a <= MAX_POSITION


def get_allowed_actions(s):
    if s.is_terminal:
        allowed_actions = np.array([-s.position], dtype=np.int32)
    else:
        allowed_actions = np.asarray([a for a in A.ALL_ACTIONS if is_allowed(s, a)], dtype=np.int32)
        assert (allowed_actions == np.sort(allowed_actions)).all()
    return allowed_actions


def position_to_idx(position):
    assert MIN_POSITION <= position <= MAX_POSITION
    return position - MIN_POSITION


def is_terminal_to_idx(is_terminal):
    return 0 if not is_terminal else 1


def future_returns_to_idx(future_returns):
    if len(future_returns) != NUM_FUTURE_RETURNS:
        print(len(future_returns))
        raise AssertionError

    future_returns = np.asarray(future_returns)
    assert (-1 <= future_returns <= 1).all()
    return future_returns + 1


def action_to_idx(a):
    return A.ALL_ACTIONS.index(a)


def state_to_idx(s):
    idx1 = position_to_idx(s.position)
    idx2 = tuple(future_returns_to_idx(s.future_returns))
    idx3 = is_terminal_to_idx(s.is_terminal)
    return (idx1, *idx2, idx3)


def state_action_to_idx(s, a):
    idx1 = tuple(state_to_idx(s))
    idx2 = action_to_idx(a)
    return (*idx1, idx2)


def policy(Q_table, s, epsilon):
    """Returns action from my action space."""

    allowed_actions = get_allowed_actions(s)

    if np.random.random() < epsilon:
        return np.random.choice(allowed_actions)

    else:
        s_idx = state_to_idx(s)
        actions_indices = np.asarray([action_to_idx(a) for a in allowed_actions])
        Q_allowed_actions = Q_table[(*s_idx, actions_indices)]
        max_Q_actions = allowed_actions[Q_allowed_actions == Q_allowed_actions.max()]
        return np.random.choice(max_Q_actions)


def compute_epsilon(episode_idx):
    return START_EPSILON / (episode_idx + 1) if episode_idx < NUM_EPISODES - 1 else 0.


def get_next_state(s, a, s_idx, all_returns):
    if s.is_terminal:
        return None

    is_terminal = (s_idx + 1) == len(all_returns) - NUM_FUTURE_RETURNS  # If s_prime is terminal.
    s_prime = State(position=s.position + a,
                    future_returns=all_returns[(s_idx + 1):(s_idx + 1 + NUM_FUTURE_RETURNS)],
                    is_terminal=is_terminal)
    assert -1 <= s_prime.position <= 1
    return s_prime


def compute_reward(s, a):
    return (s.position + a) * s.future_returns[0]


def compute_cumul_reward(realised_rewards, step_idx):
    cumul_reward = 0.
    for i, r in enumerate(realised_rewards[step_idx:]):
        cumul_reward += r * np.power(GAMMA, i)
    return cumul_reward


def run(data):
    # Preprocess to have returns: -1, 0, 1.
    prices = np.array([price for timestamp, price in data])
    returns = np.zeros(shape=(prices.shape[0] + NUM_FUTURE_RETURNS - 1,), dtype=np.int32)
    # returns[i] = sign(prices[i + 1] - prices[i]); returns[-NUM_FUTURE_RETURNS:] = 0
    returns[:-NUM_FUTURE_RETURNS] = np.sign(prices[1:] - prices[:-1])
    num_steps = prices.shape[0]
    # optimal_profit = (np.fabs(prices[1:] - prices[:-1])).sum()  # Target profit.  # TODO
    optimal_profit = (returns != 0).sum()

    # Init Q(s, a) with zeros.
    # Pairs of (s: State, a: A), where s = (position, future returns, is_terminal).
    shape = [MAX_POSITION - MIN_POSITION + 1]  # Positions.
    shape.extend([3] * NUM_FUTURE_RETURNS)  # Sequences of future returns.
    shape.append(2)  # is_terminal.
    shape.append(A.SIZE)  # Action.
    Q_table = np.zeros(shape=tuple(shape))

    episode_profits = []
    state_action_idx_to_1st_cumul_reward_list = collections.defaultdict(list)  ###
    for episode_idx in range(NUM_EPISODES):
        print('Play episode', episode_idx)
        s = State(position=0, future_returns=returns[:NUM_FUTURE_RETURNS], is_terminal=False)

        realised_actions = []
        realised_states = []
        realised_rewards = []

        for step_idx in range(num_steps):
            # Choose the best "a" from "s" according to the policy based on Q.
            epsilon = compute_epsilon(episode_idx)
            a = policy(Q_table, s, epsilon)
            s_prime = get_next_state(s, a, step_idx, returns)
            r = compute_reward(s, a)

            realised_states.append(s)
            realised_actions.append(a)
            realised_rewards.append(r)

            # Some debug logging.
            # if NUM_EPISODES == 1:
            #     print('Step', step_idx, '/ s.position', s.position, '/ s.future_returns[0]', s.future_returns[0],
            #           '/ a', a, '/ s_prime.position', s_prime.position if s_prime is not None else None,
            #           '/ reward', r)

            # TODO Print warning when we make a wrong action: Compare optimal matrix and final/wrt to step Q-value mx

            # Transition.
            s = s_prime

        # End of episode => update Q.
        state_action_idx_to_1st_cumul_reward = {}
        for step_idx, (s, a) in enumerate(zip(realised_states, realised_actions)):
            state_action_idx = state_action_to_idx(s, a)
            if state_action_idx not in state_action_idx_to_1st_cumul_reward:
                cumul_reward = compute_cumul_reward(realised_rewards, step_idx)
                state_action_idx_to_1st_cumul_reward[state_action_idx] = cumul_reward
            state_action_idx_to_1st_cumul_reward_list[state_action_idx].append(state_action_idx_to_1st_cumul_reward[state_action_idx])
        # Update Q.
        for state_action_idx, cumul_reward_list in state_action_idx_to_1st_cumul_reward_list.items():
            # print('position:', state_action_idx[0] - 1,
            #       '\treturn:', state_action_idx[1] - 1,
            #       '\taction:', state_action_idx[-1] - 2,
            #       '\t mean:', np.mean(cumul_reward_list))
            # print()
            Q_table[state_action_idx] = np.mean(cumul_reward_list)

        # Plot episode trades.
        # TODO plot_utils.plot_step_trades(MODEL_NAME, episode_idx, prices, np.sign(realised_actions))

        print('End episode', episode_idx)
        profit = np.sum(realised_rewards)
        print('Total episode profit:', profit)
        episode_profits.append(profit)
        print('Target profit:', optimal_profit)
        print('================================')
        print()

    # Plot total profit wrt episode.
    plot_utils.plot_episode_profits(MODEL_NAME, episode_profits, optimal_profit)
