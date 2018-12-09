import numpy as np


START_EPSILON = 0.1
NUM_EPISODES = 10


def compute_epsilon(episode_idx):
    # Never explore in the last episode.
    return START_EPSILON / (episode_idx + 1) if episode_idx < NUM_EPISODES - 1 else 0.


class _State:

    def __init__(self, position, future_returns, is_terminal=False):
        self.position = position
        self.future_returns = future_returns
        self.is_terminal = is_terminal


class Environment:

    def __init__(self,
                 gamma=0.9, alpha=0.1,  # Gamma and alpha shouldn't both be big.

                 # Action params.
                 max_buy_sell=1,  # Trade limit.

                 # State params.
                 min_position=-3,
                 max_position=+3,
                 num_future_returns=5,
                 ):
        self.gamma = gamma
        self.alpha = alpha

        self.max_buy_sell = max_buy_sell
        self.all_actions = [i for i in range(-self.max_buy_sell, self.max_buy_sell + 1)]

        self.min_position = min_position
        self.max_position = max_position
        self.num_future_returns = num_future_returns  # Possible return values: -1, 0, +1.

    @property
    def name(self):
        return 'episodes{episodes}_gamma{gamma}_alpha{alpha}_AK{AK}_min{min}_max{max}_returns{returns}'.format(
            episodes=NUM_EPISODES, gamma=self.gamma, alpha=self.alpha, AK=self.max_buy_sell, min=self.min_position,
            max=self.max_position, returns=self.num_future_returns)

    def State(self, position, future_returns, is_terminal=False):
        """Build a state."""

        assert self.min_position <= position <= self.max_position
        assert len(future_returns) == self.num_future_returns
        future_returns = np.asarray(future_returns)
        if is_terminal:
            assert (future_returns == 0).all()
        return _State(position, future_returns, is_terminal=is_terminal)

    def NextState(self, s, a, step_idx, all_returns):
        if s.is_terminal:
            return None

        # Check if s_prime will be terminal.
        is_terminal = (step_idx + 1) == len(all_returns) - self.num_future_returns
        s_prime = self.State(position=s.position + a,
                             future_returns=all_returns[(step_idx + 1):(step_idx + 1 + self.num_future_returns)],
                             is_terminal=is_terminal)
        return s_prime

    def is_allowed(self, s, a):
        """If the given action (a) is allowed from the given state (s)."""

        assert not s.is_terminal
        return self.min_position <= s.position + a <= self.max_position

    def get_allowed_actions(self, s):
        """Returns all actions allowed from the given state (s)."""

        if s.is_terminal:
            # We don't close the final position as it's irrelevant to the reward.
            # But we'll take it into account when computing total profit in $.
            allowed_actions = np.array([0], dtype=np.int32)
        else:
            allowed_actions = np.asarray([a for a in self.all_actions if self.is_allowed(s, a)], dtype=np.int32)
            assert (allowed_actions == np.sort(allowed_actions)).all()
        return allowed_actions

    # =========================================================================
    # Convert states and actions to indices for Q-table.
    # =========================================================================

    def position_to_idx(self, position):
        assert self.min_position <= position <= self.max_position
        return position - self.min_position

    def is_terminal_to_idx(self, is_terminal):
        return 0 if not is_terminal else 1

    def future_returns_to_idx(self, future_returns):
        assert len(future_returns) == self.num_future_returns
        future_returns = np.asarray(future_returns)
        return future_returns + 1

    def action_to_idx(self, a):
        return self.all_actions.index(a)

    def state_to_idx(self, s):
        idx1 = self.position_to_idx(s.position)
        idx2 = tuple(self.future_returns_to_idx(s.future_returns))
        idx3 = self.is_terminal_to_idx(s.is_terminal)
        return (idx1, *idx2, idx3)

    def state_action_to_idx(self, s, a):
        idx1 = tuple(self.state_to_idx(s))
        idx2 = self.action_to_idx(a)
        return (*idx1, idx2)

    # =========================================================================

    def compute_reward(self, s, a):
        return (s.position + a) * s.future_returns[0]

    def compute_cumul_reward(self, realised_rewards, step_idx):
        cumul_reward = 0.
        for i, r in enumerate(realised_rewards[step_idx:]):
            cumul_reward += r * np.power(self.gamma, i)
            if i > 100:  # self.gamma ^ 100 is very small and can be neglected.
                break
        return cumul_reward

    def compute_total_profit(self, realised_states, realised_actions, prices):
        trades = np.copy(realised_actions)
        trades[-1] = -realised_states[-1].position
        assert len(trades) == len(prices)
        assert trades.sum() == 0
        return -(trades * prices).sum()
