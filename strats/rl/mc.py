import collections
import numpy as np
import plot_utils
import strats.rl.setup as setup


MODES = ['First_Visit', 'Every_Visit']


def policy(env, Q_table, s, epsilon):
    """Returns action from my action space."""

    allowed_actions = env.get_allowed_actions(s)

    if np.random.random() < epsilon:
        return np.random.choice(allowed_actions)

    else:
        state_idx = env.state_to_idx(s)
        actions_indices = np.asarray([env.action_to_idx(a) for a in allowed_actions])
        Q_allowed_actions = Q_table[(*state_idx, actions_indices)]
        max_Q_actions = allowed_actions[Q_allowed_actions == Q_allowed_actions.max()]
        return np.random.choice(max_Q_actions)


def run(env: setup.Environment, data, mode='Every_Visit'):
    """Based on https://www.doc.ic.ac.uk/~mpd37/talks/2013-09-17-rl.pdf."""

    assert mode in MODES

    model_name = 'MC {mode}'.format(mode=mode.replace('_', ' '))

    # Preprocess to have returns: -1, 0, 1.
    # Returns are padded so that the final state has NUM_FUTURE_RETURNS all equal to 0.
    prices = np.array([price for timestamp, price in data])
    returns = np.zeros(shape=(prices.shape[0] + env.num_future_returns - 1,), dtype=np.int32)
    returns[:-env.num_future_returns] = np.sign(prices[1:] - prices[:-1])
    num_steps = prices.shape[0]

    # Init Q(s, a) with zeros.
    # Pairs of (s: State, a: A), where s = (position, future returns, is_terminal).
    shape = [env.max_position - env.min_position + 1]  # Positions.
    shape.extend([3] * env.num_future_returns)  # Sequences of future returns.
    shape.append(2)  # is_terminal.
    shape.append(len(env.all_actions))  # Action.
    Q_table = np.zeros(shape=tuple(shape))

    episode_rewards = []  # Total episode rewards.
    episode_profits = []  # Total episode profits in $.
    state_action_idx_to_cumul_reward_list = collections.defaultdict(list)

    for episode_idx in range(setup.NUM_EPISODES):
        print(model_name, '/ Play episode', episode_idx)
        s = env.State(position=0, future_returns=returns[:env.num_future_returns], is_terminal=False)

        realised_actions = []
        realised_states = []
        realised_rewards = []

        for step_idx in range(num_steps):
            # Choose the best "a" from "s" according to the policy based on Q.
            epsilon = setup.compute_epsilon(episode_idx)
            a = policy(env, Q_table, s, epsilon)
            s_prime = env.NextState(s, a, step_idx, returns)
            r = env.compute_reward(s, a)

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
            state_action_idx = env.state_action_to_idx(s, a)
            if state_action_idx not in state_action_idx_to_cumul_reward:
                cumul_reward = env.compute_cumul_reward(realised_rewards, step_idx)
                state_action_idx_to_cumul_reward[state_action_idx] = cumul_reward
            state_action_idx_to_cumul_reward_list[state_action_idx].append(
                state_action_idx_to_cumul_reward[state_action_idx])

            # In Every-Visit MC, we want to compute discounted sums of reward for each step,
            # i.e. for each encounter of the (state, action) pair.
            # Hence don't cache the cumul reward of the first encounter.
            if mode == 'Every_Visit':
                del state_action_idx_to_cumul_reward[state_action_idx]

        # Now can actually update Q.
        for state_action_idx, cumul_reward_list in state_action_idx_to_cumul_reward_list.items():
            Q_table[state_action_idx] = np.mean(cumul_reward_list)

        # Plot episode trades.
        plot_utils.plot_step_trades(env.name, model_name, episode_idx, prices, np.sign(realised_actions))

        print(model_name, '/ End episode', episode_idx)
        total_episode_reward = np.sum(realised_rewards)
        total_episode_profit = env.compute_total_profit(realised_states, realised_actions, prices)
        print(model_name, '/ Total episode reward:', total_episode_reward)
        print(model_name, '/ Total episode profit:', total_episode_profit)
        episode_rewards.append(total_episode_reward)
        episode_profits.append(total_episode_profit)
        print(model_name, '/ ================================')
        print()

    return model_name, episode_rewards, episode_profits
