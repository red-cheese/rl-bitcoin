import numpy as np


START_EPSILON = 0.1
NUM_EPISODES = 10


def compute_epsilon(episode_idx):
    # episode_idx = max(0, int(episode_idx - NUM_EPISODES / 2))  # TODO? More exploration?
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

                 simple_returns=True,  # Whether to look at returns just as -1, 0, +1.
                 # The following options are only processed when simple_returns=False.
                 clip_returns=1.,
                 returns_bins=199,  # Must be odd to include 0.
                 ):
        self.gamma = gamma
        self.alpha = alpha

        self.max_buy_sell = max_buy_sell
        self.all_actions = [i for i in range(-self.max_buy_sell, self.max_buy_sell + 1)]

        self.min_position = min_position
        self.max_position = max_position
        self.num_future_returns = num_future_returns

        self.simple_returns = simple_returns
        self.clip_returns = clip_returns
        self.returns_bins = returns_bins  # Number of bins.
        self.num_possible_returns = 3 if self.simple_returns else self.returns_bins
        self.bins = None  # Will be set later if required.

    @property
    def name(self):
        return 'episodes{episodes}_gamma{gamma}_alpha{alpha}_AK{AK}_min{min}_max{max}_returns{returns}_' \
               'simple{simple}_clip{clip}_bins{bins}'.format(
                episodes=NUM_EPISODES, gamma=self.gamma, alpha=self.alpha, AK=self.max_buy_sell, min=self.min_position,
                max=self.max_position, returns=self.num_future_returns, simple=self.simple_returns,
                clip=self.clip_returns, bins=self.returns_bins)

    @property
    def feature_vector_size(self):
        return self.max_position - self.min_position + 1 + self.num_future_returns

    def preprocess_returns(self, prices: np.array):
        """Note that returns are padded so that the final state has env.num_future_returns all equal to 0."""

        # Pad returns.
        returns = np.zeros(shape=(prices.shape[0] + self.num_future_returns - 1,))
        returns[:-self.num_future_returns] = prices[1:] - prices[:-1]

        if self.simple_returns:
            return np.sign(returns).astype(np.int32)

        returns[:-self.num_future_returns] = returns[:-self.num_future_returns] / prices[:-1]
        returns = np.clip(returns, a_min=-self.clip_returns, a_max=self.clip_returns)

        min_return = returns.min()
        max_return = returns.max()
        m = max(np.fabs(min_return), np.fabs(max_return))
        self.bins = np.linspace(-m, m, num=self.returns_bins, endpoint=True)
        assert len(self.bins) == self.returns_bins
        new_returns = self.bins[np.digitize(returns, self.bins) - 1]

        return new_returns

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

        if self.simple_returns:
            return future_returns + 1

        return np.digitize(future_returns, self.bins) - 1

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

    def state_to_feature_vector(self, s):
        feature_vector = np.zeros(shape=(self.feature_vector_size,))
        feature_vector[self.position_to_idx(s.position)] = 1
        feature_vector[-self.num_future_returns:] = s.future_returns
        return feature_vector

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
