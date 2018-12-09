import numpy as np
import plot_utils
import strats.rl.setup as setup


MODEL_NAME = 'Q-learning'


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


def run(env: setup.Environment, data):
    """Based on https://www.doc.ic.ac.uk/~mpd37/talks/2013-09-17-rl.pdf."""

    # Note that returns are padded so that the final state has env.num_future_returns all equal to 0.
    prices = np.array([price for timestamp, price in data])
    num_steps = prices.shape[0]
    returns = env.preprocess_returns(prices)

    # Init Q(s, a) with zeros.
    # Pairs of (s: State, a: A), where s = (position, future returns, is_terminal).
    shape = [env.max_position - env.min_position + 1]  # Positions.
    shape.extend([env.returns_bins] * env.num_future_returns)  # Sequences of future returns.
    shape.append(2)  # is_terminal.
    shape.append(len(env.all_actions))  # Action.
    Q_table = np.zeros(shape=tuple(shape))

    episode_rewards = []  # Total episode rewards.
    episode_profits = []  # Total episode profits in $.

    for episode_idx in range(setup.NUM_EPISODES):
        print(MODEL_NAME, '/ Play episode', episode_idx)
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

            # Q is updated on each step (except for the terminal state).
            if s_prime is not None:
                state_action_idx = env.state_action_to_idx(s, a)
                s_prime_idx = env.state_to_idx(s_prime)
                allowed_actions_from_s_prime = np.asarray([env.action_to_idx(aa)
                                                           for aa in env.get_allowed_actions(s_prime)])
                max_Q_s_prime = Q_table[(*s_prime_idx, allowed_actions_from_s_prime)].max()
                Q_table[state_action_idx] += env.alpha * (r + env.gamma * max_Q_s_prime - Q_table[state_action_idx])

            # Transition.
            s = s_prime

        # Plot episode realised actions.
        plot_utils.plot_step_trades(env.name, MODEL_NAME, episode_idx, prices, np.sign(realised_actions))

        print(MODEL_NAME, '/ End episode', episode_idx)
        # Note that total_episode_reward won't make sense when returns are not simple (i.e. not -1, 0, 1).
        total_episode_reward = np.sum(realised_rewards)
        total_episode_profit = env.compute_total_profit(realised_states, realised_actions, prices)
        print(MODEL_NAME, '/ Total episode reward:', total_episode_reward)
        print(MODEL_NAME, '/ Total episode profit:', total_episode_profit)
        episode_rewards.append(total_episode_reward)
        episode_profits.append(total_episode_profit)
        print(MODEL_NAME, '/ ================================')
        print()

    return MODEL_NAME, episode_rewards, episode_profits
