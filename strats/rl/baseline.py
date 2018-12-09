import numpy as np
import plot_utils
import strats.rl.setup as setup


MODEL_NAME = 'Baseline'


def baseline_policy(self, s):
    """Just looks at the first next return and buys/sells as much as it can."""

    allowed_actions = self.get_allowed_actions(s)

    if s.is_terminal:
        assert len(allowed_actions) == 1
        return allowed_actions[0]

    allowed_actions_sign = np.sign(allowed_actions)
    good_actions = allowed_actions[(allowed_actions_sign == np.sign(s.future_returns[0])) | (allowed_actions == 0)]
    return good_actions[np.argmax(np.abs(good_actions))]


def run(env: setup.Environment, data):
    """Based on https://www.doc.ic.ac.uk/~mpd37/talks/2013-09-17-rl.pdf."""

    # Note that returns are padded so that the final state has env.num_future_returns all equal to 0.
    prices = np.array([price for timestamp, price in data])
    num_steps = prices.shape[0]
    returns = env.preprocess_returns(prices)

    episode_rewards = []  # Total episode rewards.
    episode_profits = []  # Total episode profits in $.

    for episode_idx in range(setup.NUM_EPISODES):
        print(MODEL_NAME, '/ Play episode', episode_idx)
        s = env.State(position=0, future_returns=returns[:env.num_future_returns], is_terminal=False)

        realised_actions = []
        realised_states = []
        realised_rewards = []

        for step_idx in range(num_steps):
            a = baseline_policy(env, s)
            s_prime = env.NextState(s, a, step_idx, returns)
            r = env.compute_reward(s, a)

            realised_states.append(s)
            realised_actions.append(a)
            realised_rewards.append(r)

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
