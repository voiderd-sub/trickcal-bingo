import gymnasium as gym
import numpy as np


class D4AugmentationWrapper(gym.Wrapper):
    """
    D4 대칭 증강 Wrapper.
    
    매 step마다 랜덤 D4 변환 (8-fold: 4회전 × 2반사)을 적용하여
    에이전트가 대칭적 정책을 학습하도록 함.
    
    원리:
    - 관측을 변환하여 에이전트에 전달
    - 에이전트의 action을 역변환하여 실제 환경에 적용
    - 8배 샘플 효율 달성
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.board_size = env.board_size
        self.transform_idx = 0
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.transform_idx = np.random.randint(8)
        return self._transform_obs(obs), info
    
    def step(self, action):
        # 에이전트의 action을 역변환하여 실제 action 계산
        real_action = self._inverse_transform_action(action)
        
        obs, reward, terminated, truncated, info = self.env.step(real_action)
        
        # 새로운 랜덤 변환 선택 (매 step)
        self.transform_idx = np.random.randint(8)
        
        return self._transform_obs(obs), reward, terminated, truncated, info
    
    def _transform_obs(self, obs):
        """관측 전체를 변환"""
        transformed = {}
        
        for key, value in obs.items():
            if key in ("board", "pattern", "stored_pattern"):
                transformed[key] = self._transform_board(value)
            elif key == "action_mask":
                transformed[key] = self._transform_action_mask(value)
            else:
                transformed[key] = value
        
        return transformed
    
    def _transform_board(self, board):
        """2D 보드를 D4 변환"""
        return self._apply_d4(board, self.transform_idx)
    
    def _transform_action_mask(self, mask):
        """Action mask를 D4 변환 (마지막 store action 제외)"""
        board_size = self.board_size
        position_mask = mask[:-1].reshape(board_size, board_size)
        transformed_position_mask = self._apply_d4(position_mask, self.transform_idx)
        store_mask = mask[-1:]
        return np.concatenate([transformed_position_mask.flatten(), store_mask])
    
    def _inverse_transform_action(self, action):
        """변환된 공간의 action을 원래 공간으로 역변환"""
        board_size = self.board_size
        store_action = board_size * board_size
        
        if action == store_action:
            return action
        
        # Position action -> 2D coordinate
        row, col = divmod(action, board_size)
        
        # 역변환 적용 (inverse of transform_idx)
        inv_idx = self._inverse_transform_idx(self.transform_idx)
        new_row, new_col = self._transform_point(row, col, inv_idx, board_size)
        
        return new_row * board_size + new_col
    
    @staticmethod
    def _apply_d4(arr, transform_idx):
        """
        D4 변환 적용.
        
        0: identity
        1: rot90 (반시계)
        2: rot180
        3: rot270
        4: flip horizontal
        5: flip horizontal + rot90
        6: flip horizontal + rot180  
        7: flip horizontal + rot270
        """
        if transform_idx == 0:
            return arr.copy()
        elif transform_idx == 1:
            return np.rot90(arr, 1)
        elif transform_idx == 2:
            return np.rot90(arr, 2)
        elif transform_idx == 3:
            return np.rot90(arr, 3)
        elif transform_idx == 4:
            return np.fliplr(arr).copy()
        elif transform_idx == 5:
            return np.rot90(np.fliplr(arr), 1)
        elif transform_idx == 6:
            return np.rot90(np.fliplr(arr), 2)
        elif transform_idx == 7:
            return np.rot90(np.fliplr(arr), 3)
        else:
            raise ValueError(f"Invalid transform_idx: {transform_idx}")
    
    @staticmethod
    def _inverse_transform_idx(transform_idx):
        """D4 변환의 역변환 인덱스"""
        # D4 군의 역원
        # 0->0, 1->3, 2->2, 3->1, 4->4, 5->7, 6->6, 7->5
        inverse_map = {0: 0, 1: 3, 2: 2, 3: 1, 4: 4, 5: 7, 6: 6, 7: 5}
        return inverse_map[transform_idx]
    
    @staticmethod
    def _transform_point(row, col, transform_idx, size):
        """
        (row, col) 좌표를 D4 변환.
        size: 보드 크기 (7)
        """
        n = size - 1
        
        if transform_idx == 0:
            return row, col
        elif transform_idx == 1:  # rot90 반시계
            return col, n - row
        elif transform_idx == 2:  # rot180
            return n - row, n - col
        elif transform_idx == 3:  # rot270 반시계 = rot90 시계
            return n - col, row
        elif transform_idx == 4:  # flip horizontal
            return row, n - col
        elif transform_idx == 5:  # flip + rot90
            return col, row
        elif transform_idx == 6:  # flip + rot180
            return n - row, col
        elif transform_idx == 7:  # flip + rot270
            return n - col, n - row
        else:
            raise ValueError(f"Invalid transform_idx: {transform_idx}")


if __name__ == "__main__":
    from bingo_env import BingoEnv
    
    # 테스트
    env = BingoEnv(7)
    wrapped_env = D4AugmentationWrapper(env)
    
    print("=== D4 Wrapper Test ===")
    obs, _ = wrapped_env.reset()
    print(f"Transform idx: {wrapped_env.transform_idx}")
    
    # 몇 스텝 실행
    for i in range(5):
        mask = obs["action_mask"]
        valid_actions = np.where(mask == 1)[0]
        if len(valid_actions) == 0:
            break
        action = np.random.choice(valid_actions)
        obs, reward, done, truncated, info = wrapped_env.step(action)
        print(f"Step {i+1}: action={action}, transform={wrapped_env.transform_idx}, reward={reward:.2f}")
        if done:
            break
    
    # 변환 정확성 테스트
    print("\n=== Transform Correctness Test ===")
    test_board = np.zeros((7, 7), dtype=np.int8)
    test_board[0, 0] = 1  # 좌상단
    test_board[6, 6] = 1  # 우하단
    
    for t_idx in range(8):
        transformed = D4AugmentationWrapper._apply_d4(test_board, t_idx)
        inv_idx = D4AugmentationWrapper._inverse_transform_idx(t_idx)
        restored = D4AugmentationWrapper._apply_d4(transformed, inv_idx)
        
        is_identical = np.array_equal(test_board, restored)
        print(f"Transform {t_idx}: inverse={inv_idx}, restored={is_identical}")
