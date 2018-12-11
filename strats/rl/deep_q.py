import collections
import numpy as np
import plot_utils
import random
import strats.rl.setup as setup

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


MODEL_NAME = 'Deep Q Learning with replay'


def build_model(env):
    model = Sequential()
    # Make it 35 when simple_returns=True in main.py.
    model.add(Dense(250, input_dim=env.feature_vector_size, activation='relu'))
    model.add(Dense(len(env.all_actions), activation='linear'))
    model.compile(loss='mse',
                  optimizer=Adam(lr=0.01))
    return model


def greedy_policy(env, Q_network, s):
    allowed_actions = env.get_allowed_actions(s)
    s_feature_vector = env.state_to_feature_vector(s)
    Q_actions = Q_network.predict(s_feature_vector.reshape((1, -1)), batch_size=1)[0]
    allowed_actions_indices = np.asarray([env.action_to_idx(a) for a in allowed_actions])
    Q_allowed_actions = Q_actions[allowed_actions_indices]
    max_Q_actions = allowed_actions[Q_allowed_actions == Q_allowed_actions.max()]
    return np.random.choice(max_Q_actions), Q_allowed_actions.max()


def policy(env, Q_network, s, epsilon):
    """Returns action from my action space."""

    if np.random.random() < epsilon:
        allowed_actions = env.get_allowed_actions(s)
        return np.random.choice(allowed_actions)

    else:
        return greedy_policy(env, Q_network, s)[0]


def run(env: setup.Environment, data):
    """Based on https://www.doc.ic.ac.uk/~mpd37/talks/2013-09-17-rl.pdf."""

    # Note that returns are padded so that the final state has env.num_future_returns all equal to 0.
    prices = np.array([price for timestamp, price in data])
    num_steps = prices.shape[0]
    returns = env.preprocess_returns(prices)

    episode_rewards = []  # Total episode rewards.
    episode_profits = []  # Total episode profits in $.

    mem = collections.deque(maxlen=500)
    batch_size = 64

    Q_network = build_model(env)

    for episode_idx in range(setup.NUM_EPISODES):
        print(MODEL_NAME, '/ Play episode', episode_idx)
        s = env.State(position=0, future_returns=returns[:env.num_future_returns], is_terminal=False)

        realised_actions = []
        realised_states = []
        realised_rewards = []

        for step_idx in range(num_steps):
            # Choose the best "a" from "s" according to the policy based on Q.
            epsilon = setup.compute_epsilon(episode_idx)
            a = policy(env, Q_network, s, epsilon)
            s_prime = env.NextState(s, a, step_idx, returns)
            r = env.compute_reward(s, a)

            realised_states.append(s)
            realised_actions.append(a)
            realised_rewards.append(r)

            # Q is updated on each step (except for the terminal state).
            if s_prime is not None:
                mem.append((s, a, r, s_prime))

                if len(mem) >= batch_size and step_idx % 10 == 0:  # step_idx % 10 == 0 for speedup of debugging
                    minibatch = random.sample(mem, batch_size)
                    s_f_v_arr = []
                    target_f_arr = []
                    for mb_s, mb_a, mb_r, mb_s_prime in minibatch:
                        _, Q_a_from_s_prime = greedy_policy(env, Q_network, mb_s_prime)
                        target = (mb_r + env.gamma * Q_a_from_s_prime)
                        s_f_v = env.state_to_feature_vector(mb_s)
                        target_f = Q_network.predict(s_f_v.reshape((1, -1)), batch_size=1)
                        target_f[0][env.action_to_idx(mb_a)] = target
                        s_f_v_arr.append(s_f_v)
                        target_f_arr.append(target_f)
                    Q_network.fit(np.asarray(s_f_v_arr).reshape((batch_size, -1)),
                                  np.asarray(target_f_arr).reshape((batch_size, -1)),
                                  batch_size=batch_size, epochs=1, verbose=0)

            if step_idx % 100 == 0:
                print('Done step index', step_idx, 'out of', num_steps)

            # Transition.
            s = s_prime

        # Plot episode realised actions.
        plot_utils.plot_step_trades(env.name, MODEL_NAME, episode_idx, prices, np.sign(realised_actions))

        print(MODEL_NAME, '/ End episode', episode_idx)
        total_episode_reward = np.sum(realised_rewards)
        total_episode_profit = env.compute_total_profit(realised_states, realised_actions, prices)
        print(MODEL_NAME, '/ Total episode reward:', total_episode_reward)
        print(MODEL_NAME, '/ Total episode profit:', total_episode_profit)
        episode_rewards.append(total_episode_reward)
        episode_profits.append(total_episode_profit)
        print(MODEL_NAME, '/ ================================')
        print()

    # Print Q.
    env.print_Q_network(MODEL_NAME, Q_network)
    print(MODEL_NAME, '/ ================================')
    print()

    return MODEL_NAME, episode_rewards, episode_profits
