import collections
import numpy as np
import plot_utils


START_EPSILON = 0.1
NUM_EPISODES = 10
GAMMA = 0.9
MODES = ['first_visit', 'every_visit']


class A:
    """
    Actions from SELL K to BUY K.
    #A = 2K + 1.
    """

    K = 2
    ALL_ACTIONS = [i for i in range(-K, K + 1)]
    SIZE = 2 * K + 1


class State:

    MIN_POSITION = -1
    MAX_POSITION = +1
    NUM_FUTURE_RETURNS = 1  # Possible return values: -1, 0, +1.

    def __init__(self, position, future_returns, is_terminal=False):
        assert len(future_returns) == State.NUM_FUTURE_RETURNS

        future_returns = np.asarray(future_returns)
        assert (-1 <= future_returns <= 1).all()
        if is_terminal:
            assert (future_returns == 0).all()

        self.position = position
        self.future_returns = future_returns
        self.is_terminal = is_terminal


def is_allowed(s, a):
    assert not s.is_terminal
    return State.MIN_POSITION <= s.position + a <= State.MAX_POSITION


def get_allowed_actions(s):
    if s.is_terminal:
        allowed_actions = np.array([-s.position], dtype=np.int32)
    else:
        allowed_actions = np.asarray([a for a in A.ALL_ACTIONS if is_allowed(s, a)], dtype=np.int32)
        assert (allowed_actions == np.sort(allowed_actions)).all()
    return allowed_actions


def position_to_idx(position):
    assert State.MIN_POSITION <= position <= State.MAX_POSITION
    return position - State.MIN_POSITION


def is_terminal_to_idx(is_terminal):
    return 0 if not is_terminal else 1


def future_returns_to_idx(future_returns):
    assert len(future_returns) == State.NUM_FUTURE_RETURNS
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
    return START_EPSILON / np.sqrt(episode_idx + 1) if episode_idx < NUM_EPISODES - 1 else 0.


def get_next_state(s, a, s_idx, all_returns):
    if s.is_terminal:
        return None

    # Check if s_prime will be terminal.
    is_terminal = (s_idx + 1) == len(all_returns) - State.NUM_FUTURE_RETURNS
    s_prime = State(position=s.position + a,
                    future_returns=all_returns[(s_idx + 1):(s_idx + 1 + State.NUM_FUTURE_RETURNS)],
                    is_terminal=is_terminal)
    assert -1 <= s_prime.position <= 1
    return s_prime


def compute_reward(s, a):
    return (s.position + a) * s.future_returns[0]


def compute_cumul_reward(realised_rewards, step_idx):
    cumul_reward = 0.
    for i, r in enumerate(realised_rewards[step_idx:]):
        cumul_reward += r * np.power(GAMMA, i)
        if i > 100:  # GAMMA ^ 100 is very small and can be neglected.
            break
    return cumul_reward


def run(data, mode='every_visit'):
    """Based on https://www.doc.ic.ac.uk/~mpd37/talks/2013-09-17-rl.pdf."""

    assert mode in MODES

    # Name for dir.
    model_name = 'mc_{mode}_episodes{episodes}_gamma{gamma}_AK{AK}_min{min}_max{max}_returns{returns}'.format(
        mode=mode, episodes=NUM_EPISODES, gamma=GAMMA, AK=A.K, min=State.MIN_POSITION, max=State.MAX_POSITION,
        returns=State.NUM_FUTURE_RETURNS)
    # Name for plots.
    model_pretty_name = 'MC {mode} (gamma = {gamma})'.format(mode=mode.replace('_', ' '), gamma=GAMMA)

    # Preprocess to have returns: -1, 0, 1.
    # Returns are padded so that the final state has NUM_FUTURE_RETURNS all equal to 0.
    prices = np.array([price for timestamp, price in data])
    returns = np.zeros(shape=(prices.shape[0] + State.NUM_FUTURE_RETURNS - 1,), dtype=np.int32)
    returns[:-State.NUM_FUTURE_RETURNS] = np.sign(prices[1:] - prices[:-1])
    num_steps = prices.shape[0]

    optimal_reward = (returns != 0).sum()  # Optimal reward. Not optimal profit in $!
    optimal_profit = np.fabs(prices[1:] - prices[:-1]).sum()  # Optimal profit in $.

    # Init Q(s, a) with zeros.
    # Pairs of (s: State, a: A), where s = (position, future returns, is_terminal).
    shape = [State.MAX_POSITION - State.MIN_POSITION + 1]  # Positions.
    shape.extend([3] * State.NUM_FUTURE_RETURNS)  # Sequences of future returns.
    shape.append(2)  # is_terminal.
    shape.append(A.SIZE)  # Action.
    Q_table = np.zeros(shape=tuple(shape))

    episode_rewards = []  # Total episode rewards.
    episode_profits = []  # Total episode profits in $.
    state_action_idx_to_cumul_reward_list = collections.defaultdict(list)

    for episode_idx in range(NUM_EPISODES):
        print('Play episode', episode_idx)
        s = State(position=0, future_returns=returns[:State.NUM_FUTURE_RETURNS], is_terminal=False)

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

            # Transition.
            s = s_prime

        # End of episode => update Q.
        # First compute discounted sums of rewards for each (state, action) pair.
        # Hack: note that State and Action are not hashable, but their index is hashable.
        state_action_idx_to_cumul_reward = {}
        for step_idx, (s, a) in enumerate(zip(realised_states, realised_actions)):
            state_action_idx = state_action_to_idx(s, a)
            if state_action_idx not in state_action_idx_to_cumul_reward:
                cumul_reward = compute_cumul_reward(realised_rewards, step_idx)
                state_action_idx_to_cumul_reward[state_action_idx] = cumul_reward
            state_action_idx_to_cumul_reward_list[state_action_idx].append(
                state_action_idx_to_cumul_reward[state_action_idx])

            # In Every-Visit MC, we want to compute discounted sums of reward for each step,
            # i.e. for each encounter of the (state, action) pair.
            # Hence don't cache the cumul reward of the first encounter.
            if mode == 'every_visit':
                del state_action_idx_to_cumul_reward[state_action_idx]

        # Now can actually update Q.
        for state_action_idx, cumul_reward_list in state_action_idx_to_cumul_reward_list.items():
            Q_table[state_action_idx] = np.mean(cumul_reward_list)

        # Plot episode trades.
        plot_utils.plot_step_trades(model_name, model_pretty_name, episode_idx, prices, np.sign(realised_actions))

        print('End episode', episode_idx)
        total_episode_reward = np.sum(realised_rewards)
        total_episode_profit = np.sum(realised_rewards[:-1] * np.fabs(prices[1:] - prices[:-1]))
        print('Total episode reward:', total_episode_reward)
        print('Target reward:', optimal_reward)
        print('Total episode profit:', total_episode_profit)
        print('Target profit:', optimal_profit)
        episode_rewards.append(total_episode_reward)
        episode_profits.append(total_episode_profit)
        print('================================')
        print()

    # Plot total reward and profit wrt episode.
    plot_utils.plot_episode_profits(model_name, model_pretty_name, episode_rewards, optimal_reward, ylabel='Reward')
    plot_utils.plot_episode_profits(model_name, model_pretty_name, episode_profits, optimal_profit, ylabel='Profit$')
