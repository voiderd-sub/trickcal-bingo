import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

class BingoEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, board_size=7):
        super().__init__()
        
        self.board_size = board_size
        self.max_steps = self.board_size * self.board_size

        self.flip_patterns = [
            np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]]),
            np.array([[1, 0, 1],
                      [0, 1, 0],
                      [1, 0, 1]]),
            np.array([[1, 1, 1, 1, 1, 1, 1]]),
            np.array([[1],
                      [1],
                      [1],
                      [1],
                      [1],
                      [1],
                      [1]]),
            np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]),
        ]
        self.pattern_prob = np.array([0.32, 0.32, 0.15, 0.15, 0.06])
        self.pattern_prob /= np.sum(self.pattern_prob)
        for i, pattern in enumerate(self.flip_patterns):
            ph, pw = pattern.shape
            assert ph % 2 == 1 and pw % 2 == 1, \
                f"The pattern width and height must be odd, but the {i}-th pattern shape is: {pattern.shape}"
            
            padded = np.zeros((self.board_size, self.board_size), dtype=np.int8)
            offset_h = (self.board_size - ph) // 2
            offset_w = (self.board_size - pw) // 2
            padded[offset_h:offset_h+ph, offset_w:offset_w+pw] = pattern
            self.flip_patterns[i] = padded

        self.pattern_costs = [200., 200., 200., 200., 200.]
        self.max_cost = max(self.pattern_costs)

        self.action_space = spaces.Discrete(self.board_size * self.board_size)
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8),
            "pattern": spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8),
            "cost": spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(self.board_size ** 2,), dtype=np.uint8),
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_step = 0
        self._choose_new_pattern()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        row, col = divmod(action, self.board_size)

        self._apply_pattern(row, col)
        reward = - self.current_cost / self.max_cost
        self.current_step += 1

        terminated = self.board.sum() == self.board_size * self.board_size
        truncated = self.current_step >= self.max_steps

        self._choose_new_pattern()
        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def _choose_new_pattern(self):
        idx = int(np.random.choice(len(self.flip_patterns), 1, p=self.pattern_prob)[0])
        self.current_pattern = self.flip_patterns[idx]
        self.current_cost = self.pattern_costs[idx]

    def _apply_pattern(self, center_row, center_col):
        p_h, p_w = self.current_pattern.shape
        offset_h = p_h // 2
        offset_w = p_w // 2

        r_start = max(center_row - offset_h, 0)
        r_end = min(center_row + offset_h + 1, self.board_size)
        c_start = max(center_col - offset_w, 0)
        c_end = min(center_col + offset_w + 1, self.board_size)

        pattern_r_start = r_start - (center_row - offset_h)
        pattern_r_end = p_h - ((center_row + offset_h + 1) - r_end)
        pattern_c_start = c_start - (center_col - offset_w)
        pattern_c_end = p_w - ((center_col + offset_w + 1) - c_end)

        board_region = self.board[r_start:r_end, c_start:c_end]
        pattern_region = self.current_pattern[pattern_r_start:pattern_r_end, pattern_c_start:pattern_c_end]

        self.board[r_start:r_end, c_start:c_end] = np.maximum(board_region, pattern_region)


    def _get_obs(self):
        return {
            "board": self.board.copy(),
            "pattern": self.current_pattern.copy(),
            "cost": np.array(self.current_cost, dtype=np.float32),
            "action_mask": self.action_masks(),
        }

    def render(self, cost=0.0):
        self.visualize(self.board, "Current Board")
        self.visualize(self.current_pattern, "Next Pattern")
        cumul_cost = cost + self.current_cost
        print(f"Cumulative cost: {cumul_cost:.2f}")
        print("-" * 20)
        return cumul_cost

    def visualize(self, array, title):
        print(f"{title}:")
        for row in array:
            line = ' '.join('■' if cell else '□' for cell in row)
            print(line)
        print()


    def action_masks(self):
        H, W = self.board_size, self.board_size
        P = self.current_pattern
        offset_h, offset_w = P.shape[0] // 2, P.shape[1] // 2

        padded_board = np.pad(self.board, ((offset_h, offset_h), (offset_w, offset_w)), mode='constant', constant_values=1)
        
        board_windows = sliding_window_view(padded_board, P.shape)  # shape: (H, W, Ph, Pw)

        # count board == 0 and pattern == 1
        will_paint = np.logical_and(board_windows == 0, P[None, None, :, :])  # shape: (H, W, Ph, Pw)
        paint_counts = will_paint.sum(axis=(2, 3))  # shape: (H, W)

        valid_mask_2d = (paint_counts > 0).astype(np.uint8)  # shape: (N, N)
        valid_mask_flat = valid_mask_2d.flatten()  # shape: (N**2,)

        return valid_mask_flat


if __name__ == "__main__":
    env = BingoEnv(7)
    obs, _ = env.reset()
    env.render()

    for _ in range(3):
        env.step(np.random.randint(env.board_size**2))


    env.visualize(env.board, "Board")
    env.visualize(env.current_pattern, "Current_pattern")
    
    mask = env.action_masks()
    env.visualize(mask.reshape(env.board_size, env.board_size), "Mask")

    valid_indices = np.where(mask == 1)[0]

    print(f"\n총 유효한 액션 수: {len(valid_indices)} / {env.board_size ** 2}")
