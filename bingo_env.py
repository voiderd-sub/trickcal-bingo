import gymnasium as gym
from gymnasium import spaces
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


class BingoEnv(gym.Env):
    """
    7x7 빙고판을 최소 턴수로 채우는 환경.
    
    Actions:
        0~48: 위치 선택 (row * 7 + col)
        49: 패턴 보관/교환
    
    보관 규칙:
        - 첫 턴: 2번 보관 가능
        - 이후 턴: 턴마다 1번으로 리셋
        - 현재 패턴 == 저장 패턴이면 보관 불가
    """
    metadata = {"render_modes": ["human"]}

    # 패턴 정의: +, x, 3x3, 가로, 세로
    PATTERN_PLUS = np.array([[0, 1, 0],
                              [1, 1, 1],
                              [0, 1, 0]])
    PATTERN_X = np.array([[1, 0, 1],
                          [0, 1, 0],
                          [1, 0, 1]])
    PATTERN_3X3 = np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])
    PATTERN_HORIZONTAL = np.array([[1, 1, 1, 1, 1, 1, 1]])
    PATTERN_VERTICAL = np.array([[1], [1], [1], [1], [1], [1], [1]])

    def __init__(self, board_size=7):
        super().__init__()
        
        self.board_size = board_size
        self.max_steps = self.board_size * self.board_size

        # 패턴: +, x, 3x3, 가로, 세로 순서, 확률: 30, 30, 10, 15, 15%
        self.raw_patterns = [
            self.PATTERN_PLUS,
            self.PATTERN_X,
            self.PATTERN_3X3,
            self.PATTERN_HORIZONTAL,
            self.PATTERN_VERTICAL,
        ]
        self.pattern_prob = np.array([0.30, 0.30, 0.10, 0.15, 0.15])
        
        # 패턴을 board_size에 맞게 패딩
        self.flip_patterns = []
        for pattern in self.raw_patterns:
            ph, pw = pattern.shape
            assert ph % 2 == 1 and pw % 2 == 1, \
                f"Pattern dimensions must be odd, got {pattern.shape}"
            
            padded = np.zeros((self.board_size, self.board_size), dtype=np.int8)
            offset_h = (self.board_size - ph) // 2
            offset_w = (self.board_size - pw) // 2
            padded[offset_h:offset_h+ph, offset_w:offset_w+pw] = pattern
            self.flip_patterns.append(padded)

        self.pattern_costs = [1.0] * len(self.flip_patterns)
        self.max_cost = max(self.pattern_costs)

        # Action space: 49 positions + 1 store action
        self.action_space = spaces.Discrete(self.board_size * self.board_size + 1)
        self.store_action = self.board_size * self.board_size  # action 49

        # Observation space
        self.observation_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8),
            "pattern": spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8),
            "stored_pattern": spaces.Box(low=0, high=1, shape=(self.board_size, self.board_size), dtype=np.int8),
            "has_stored": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "store_remaining": spaces.Box(low=0, high=2, shape=(1,), dtype=np.float32),
            "cost": spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(self.board_size ** 2 + 1,), dtype=np.uint8),
        })

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_step = 0
        self.is_first_turn = True
        
        # 보관 상태 초기화
        self.stored_pattern = None
        self.stored_pattern_idx = None
        self.store_remaining = 2  # 첫 턴은 2번 보관 가능
        
        self._choose_new_pattern()

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False

        if action == self.store_action:
            # 보관 액션
            self._do_store()
            # 보관은 턴 소모 없음, 하지만 step은 진행
        else:
            # 위치 선택 액션
            row, col = divmod(action, self.board_size)
            self._apply_pattern(row, col)
            reward = -self.current_cost / self.max_cost
            self.current_step += 1
            
            # 턴이 끝났으면 보관 횟수 리셋
            self.is_first_turn = False
            self.store_remaining = 1  # 다음 턴에 1번 보관 가능

            terminated = self.board.sum() == self.board_size * self.board_size
            truncated = self.current_step >= self.max_steps

            self._choose_new_pattern()

        obs = self._get_obs()
        info = {}

        return obs, reward, terminated, truncated, info

    def _do_store(self):
        """현재 패턴을 보관하거나 저장된 패턴과 교환"""
        if self.stored_pattern is None:
            # 저장된 패턴 없음 → 현재 저장, 새 패턴 뽑기
            self.stored_pattern = self.current_pattern.copy()
            self.stored_pattern_idx = self.current_pattern_idx
            self._choose_new_pattern()
        else:
            # 저장된 패턴 있음 → 교환
            temp_pattern = self.current_pattern
            temp_idx = self.current_pattern_idx
            self.current_pattern = self.stored_pattern
            self.current_pattern_idx = self.stored_pattern_idx
            self.stored_pattern = temp_pattern
            self.stored_pattern_idx = temp_idx
        
        self.store_remaining -= 1

    def _choose_new_pattern(self):
        idx = int(np.random.choice(len(self.flip_patterns), 1, p=self.pattern_prob)[0])
        self.current_pattern = self.flip_patterns[idx]
        self.current_pattern_idx = idx
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
        stored = self.stored_pattern if self.stored_pattern is not None else np.zeros((self.board_size, self.board_size), dtype=np.int8)
        
        return {
            "board": self.board.copy(),
            "pattern": self.current_pattern.copy(),
            "stored_pattern": stored.copy(),
            "has_stored": np.array([1.0 if self.stored_pattern is not None else 0.0], dtype=np.float32),
            "store_remaining": np.array([float(self.store_remaining)], dtype=np.float32),
            "cost": np.array(self.current_cost, dtype=np.float32),
            "action_mask": self.action_masks(),
        }

    def render(self, cost=0.0):
        self.visualize(self.board, "Current Board")
        self.visualize(self.current_pattern, "Current Pattern")
        if self.stored_pattern is not None:
            self.visualize(self.stored_pattern, "Stored Pattern")
        print(f"Store remaining: {self.store_remaining}")
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
        
        board_windows = sliding_window_view(padded_board, P.shape)

        will_paint = np.logical_and(board_windows == 0, P[None, None, :, :])
        paint_counts = will_paint.sum(axis=(2, 3))

        valid_mask_2d = (paint_counts > 0).astype(np.uint8)
        valid_mask_flat = valid_mask_2d.flatten()

        # 보관 액션 마스크
        can_store = self._can_store()
        store_mask = np.array([1 if can_store else 0], dtype=np.uint8)

        return np.concatenate([valid_mask_flat, store_mask])

    def _can_store(self):
        """보관 가능 여부 확인"""
        if self.store_remaining <= 0:
            return False
        # 현재 패턴과 저장된 패턴이 같으면 보관 불가
        if self.stored_pattern is not None:
            if np.array_equal(self.current_pattern, self.stored_pattern):
                return False
        return True


if __name__ == "__main__":
    env = BingoEnv(7)
    obs, _ = env.reset()
    env.render()

    print("=== Test store action ===")
    # 보관 테스트
    if obs["action_mask"][-1] == 1:
        obs, reward, done, truncated, info = env.step(env.store_action)
        print("After first store:")
        env.render()
    
    if obs["action_mask"][-1] == 1:
        obs, reward, done, truncated, info = env.step(env.store_action)
        print("After second store:")
        env.render()

    print("=== Test position actions ===")
    for _ in range(3):
        mask = obs["action_mask"][:-1]  # 보관 액션 제외
        valid_indices = np.where(mask == 1)[0]
        if len(valid_indices) > 0:
            action = np.random.choice(valid_indices)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
        if done:
            break

    print(f"\nTotal steps: {env.current_step}")
    print(f"Board filled: {env.board.sum()}/{env.board_size ** 2}")
