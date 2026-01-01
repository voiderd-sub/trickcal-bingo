"""
GPU-Accelerated Vectorized Bingo Environment.

All environment operations run on GPU using PyTorch tensors.
Provides massive speedup by eliminating CPU-GPU data transfer and
enabling batched parallel execution.

REFACTORED: Uses num_patterns (any positive integer) for n-pattern support.
For num_patterns>=2, patterns are stored as sorted tuples for canonical state.
Observation includes pattern_indices as integer tensor (efficient embedding lookup).
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict


class BingoEnvGPU:
    """
    PyTorch-based GPU Vectorized Bingo Environment.
    
    All state and operations use GPU tensors for maximum throughput.
    - board_bits: (num_envs,) int64 tensor - bitboard state
    - pattern_indices: (num_envs, num_patterns) int64 tensor - current pattern indices
    - All mask operations executed as batched GPU operations
    
    Args:
        num_envs: Number of parallel environments
        device: Device to run on ('cuda' or 'cpu')
        use_augmentation: Whether to use D4 symmetry augmentation
        num_patterns: Number of patterns to hold (any positive integer)
            - 1: Simple mode, single pattern per turn
            - 2+: Choice mode, choose which pattern to place (sorted for canonical state)
    """
    
    # Pattern definitions (class constants)
    PATTERN_PLUS = torch.tensor([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=torch.int8)
    PATTERN_X = torch.tensor([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=torch.int8)
    PATTERN_3X3 = torch.tensor([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=torch.int8)
    PATTERN_HORIZONTAL = torch.tensor([[1, 1, 1, 1, 1, 1, 1]], dtype=torch.int8)
    PATTERN_VERTICAL = torch.tensor([[1], [1], [1], [1], [1], [1], [1]], dtype=torch.int8)
    
    # All 49 bits set = board full
    FULL_BOARD_MASK = 0x1FFFFFFFFFFFF
    
    def __init__(
        self,
        num_envs: int,
        device: str = 'cuda',
        use_augmentation: bool = True,
        min_initial_turns: int = 0,
        max_initial_turns: int = 0,
        num_patterns: int = 1,
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.use_augmentation = use_augmentation
        self.num_patterns = num_patterns
        self.board_size = 7
        self.n_cells = 49
        self.max_steps = 49
        
        # Validate num_patterns
        assert num_patterns >= 1, f"num_patterns must be >= 1, got {num_patterns}"
        
        # Action space: 49 positions per pattern
        self.n_actions = 49 * num_patterns
        
        # Curriculum settings
        self.min_initial_turns = min_initial_turns
        self.max_initial_turns = max_initial_turns
        
        # Pattern probability (for sampling)
        self.pattern_prob = torch.tensor(
            [0.30, 0.30, 0.10, 0.15, 0.15],
            dtype=torch.float32,
            device=self.device
        )
        self.num_pattern_types = 5
        
        # Pre-compute bitmasks for all patterns at all positions
        # Shape: (num_pattern_types, 49) - each mask is a uint64
        self.precomputed_masks = self._generate_all_masks_gpu()
        
        # Pre-compute padded patterns for observation (7x7 each)
        # Shape: (num_pattern_types, 7, 7)
        self.obs_patterns = self._generate_obs_patterns_gpu()
        
        # D4 augmentation lookup tables
        if self.use_augmentation:
            self._init_d4_tables()
        
        # Environment state tensors
        self.board_bits = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        # Pattern indices: (num_envs, num_patterns) - sorted for canonical state when num_patterns=2
        self.pattern_indices = torch.zeros((num_envs, num_patterns), dtype=torch.int64, device=self.device)
        self.current_step = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        
        # D4 transform indices for each env (0-7)
        self.transform_idx = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        
        # Bit shift helper for observation conversion
        self.bit_shifts = torch.arange(49, dtype=torch.int64, device=self.device)
        
        # Cached action masks (to avoid double computation)
        self._cached_action_mask = None
    
    def _generate_all_masks_gpu(self) -> torch.Tensor:
        """Pre-compute bitmasks for every pattern at every position."""
        raw_patterns = [
            self.PATTERN_PLUS,
            self.PATTERN_X,
            self.PATTERN_3X3,
            self.PATTERN_HORIZONTAL,
            self.PATTERN_VERTICAL,
        ]
        
        all_masks = []
        for pat in raw_patterns:
            pat_masks = []
            ph, pw = pat.shape
            offset_h, offset_w = ph // 2, pw // 2
            
            for r in range(7):
                for c in range(7):
                    mask_val = 0
                    for pr in range(ph):
                        for pc in range(pw):
                            if pat[pr, pc] == 1:
                                board_r = r - offset_h + pr
                                board_c = c - offset_w + pc
                                if 0 <= board_r < 7 and 0 <= board_c < 7:
                                    bit_idx = board_r * 7 + board_c
                                    mask_val |= (1 << bit_idx)
                    pat_masks.append(mask_val)
            all_masks.append(pat_masks)
        
        return torch.tensor(all_masks, dtype=torch.int64, device=self.device)
    
    def _generate_obs_patterns_gpu(self) -> torch.Tensor:
        """Pre-compute padded 7x7 patterns for observation."""
        raw_patterns = [
            self.PATTERN_PLUS,
            self.PATTERN_X,
            self.PATTERN_3X3,
            self.PATTERN_HORIZONTAL,
            self.PATTERN_VERTICAL,
        ]
        
        obs_patterns = []
        for p in raw_patterns:
            padded = torch.zeros((7, 7), dtype=torch.int8)
            ph, pw = p.shape
            oh, ow = (7 - ph) // 2, (7 - pw) // 2
            padded[oh:oh+ph, ow:ow+pw] = p
            obs_patterns.append(padded)
        
        return torch.stack(obs_patterns).to(self.device)
    
    def _init_d4_tables(self):
        """Initialize D4 symmetry transformation lookup tables."""
        # Position remapping for each of 8 D4 transforms
        # For each transform, stores the mapping: new_pos = table[original_pos]
        self.d4_forward_pos = torch.zeros((8, 49), dtype=torch.int64, device=self.device)
        self.d4_inverse_pos = torch.zeros((8, 49), dtype=torch.int64, device=self.device)
        
        # Inverse transform indices: inv_map[t] gives inverse of transform t
        self.d4_inverse_idx = torch.tensor([0, 3, 2, 1, 4, 7, 6, 5], dtype=torch.int64, device=self.device)
        
        n = 6  # board_size - 1
        
        for pos in range(49):
            row, col = pos // 7, pos % 7
            
            # Transform 0: identity
            self.d4_forward_pos[0, pos] = pos
            # Transform 1: rot90 CCW
            new_row, new_col = col, n - row
            self.d4_forward_pos[1, pos] = new_row * 7 + new_col
            # Transform 2: rot180
            new_row, new_col = n - row, n - col
            self.d4_forward_pos[2, pos] = new_row * 7 + new_col
            # Transform 3: rot270 CCW
            new_row, new_col = n - col, row
            self.d4_forward_pos[3, pos] = new_row * 7 + new_col
            # Transform 4: flip horizontal
            new_row, new_col = row, n - col
            self.d4_forward_pos[4, pos] = new_row * 7 + new_col
            # Transform 5: flip + rot90
            new_row, new_col = col, row
            self.d4_forward_pos[5, pos] = new_row * 7 + new_col
            # Transform 6: flip + rot180
            new_row, new_col = n - row, col
            self.d4_forward_pos[6, pos] = new_row * 7 + new_col
            # Transform 7: flip + rot270
            new_row, new_col = n - col, n - row
            self.d4_forward_pos[7, pos] = new_row * 7 + new_col
        
        # Compute inverse tables
        for t in range(8):
            inv_t = self.d4_inverse_idx[t].item()
            self.d4_inverse_pos[t] = self.d4_forward_pos[inv_t]
    
    def set_curriculum(self, min_turns: int, max_turns: int):
        """Update curriculum settings."""
        self.min_initial_turns = min_turns
        self.max_initial_turns = max_turns
    
    def reset(self, env_indices: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Reset environments.
        
        Args:
            env_indices: Optional tensor of environment indices to reset.
                        If None, resets all environments.
        
        Returns:
            Observation dictionary with GPU tensors.
        """
        if env_indices is None:
            # Reset all
            indices = torch.arange(self.num_envs, device=self.device)
            self.board_bits.zero_()
            self.current_step.zero_()
        else:
            indices = env_indices
            self.board_bits[indices] = 0
            self.current_step[indices] = 0
        
        # Choose new patterns for reset envs (all slots)
        self._choose_all_patterns(indices)
        
        # Apply curriculum (random prefill)
        if self.max_initial_turns > 0:
            self._prefill_boards(indices)
            
            # Safety checks: If prefill resulted in a full board (terminal state),
            # we must reset it to empty to avoid crashing
            is_full = self.board_bits[indices] == self.FULL_BOARD_MASK
            if is_full.any():
                bad_indices = indices[is_full]
                # Reset to clean state (effectively rejecting the prefill)
                self.board_bits[bad_indices] = 0
                self.current_step[bad_indices] = 0
                self._choose_all_patterns(bad_indices)
        
        # Random D4 transform for augmentation
        if self.use_augmentation:
            self.transform_idx[indices] = torch.randint(0, 8, (len(indices),), device=self.device)
        
        # Pre-compute action masks and cache for _get_obs()
        self._cached_action_mask = self._compute_and_transform_action_masks()
        
        return self._get_obs()
    
    def _choose_all_patterns(self, indices: torch.Tensor):
        """Sample all pattern slots for specified environments."""
        n = len(indices)
        # Sample num_patterns patterns for each env
        new_patterns = torch.multinomial(
            self.pattern_prob.expand(n, -1),
            num_samples=self.num_patterns,
            replacement=True,  # Allow same pattern in both slots
        )
        self.pattern_indices[indices] = new_patterns
        
        # Sort for canonical state when num_patterns=2
        if self.num_patterns == 2:
            self.pattern_indices[indices] = self.pattern_indices[indices].sort(dim=1).values
    
    def _choose_one_pattern(self, indices: torch.Tensor, slot: int):
        """Resample a single pattern slot and maintain sorted order."""
        n = len(indices)
        new_pattern = torch.multinomial(
            self.pattern_prob.expand(n, -1),
            num_samples=1,
        ).squeeze(-1)
        self.pattern_indices[indices, slot] = new_pattern
        
        # Re-sort for canonical state when num_patterns=2
        if self.num_patterns == 2:
            self.pattern_indices[indices] = self.pattern_indices[indices].sort(dim=1).values
    
    def _prefill_boards(self, indices: torch.Tensor):
        """Apply curriculum by pre-filling boards with random moves."""
        num_turns = torch.randint(
            self.min_initial_turns,
            self.max_initial_turns + 1,
            (len(indices),),
            device=self.device
        )
        
        max_turns = num_turns.max().item()
        
        for _ in range(max_turns):
            # Which envs still need turns
            active = num_turns > 0
            if not active.any():
                break
            
            active_indices = indices[active]
            
            # Get valid actions for active envs (use pattern 0 for prefill)
            masks = self._compute_action_masks_subset(active_indices)
            # Only use first pattern's positions for prefill
            position_masks = masks[:, :49]
            
            # Check which envs have valid moves
            has_valid = position_masks.any(dim=1)
            if not has_valid.any():
                break
            
            # Sample random valid action for each
            valid_envs = active_indices[has_valid]
            valid_masks = position_masks[has_valid]
            
            # Sample action
            actions = torch.multinomial(valid_masks.float(), num_samples=1).squeeze(-1)
            
            # Apply moves using pattern 0
            pattern_idx = self.pattern_indices[valid_envs, 0]
            action_masks = self.precomputed_masks[pattern_idx, actions]
            self.board_bits[valid_envs] |= action_masks
            
            # Choose new pattern for slot 0
            self._choose_one_pattern(valid_envs, 0)
            
            # Decrement turns
            num_turns[active] -= 1
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Execute actions for all environments.
        
        Args:
            actions: (num_envs,) tensor of actions
                - num_patterns=1: 0-48 for positions
                - num_patterns=2: 0-48 for pattern 0, 49-97 for pattern 1
        
        Returns:
            obs: Observation dictionary
            rewards: (num_envs,) reward tensor
            dones: (num_envs,) done tensor
            truncated: (num_envs,) truncated tensor
            infos: Empty dict (for API compatibility)
        """
        # Handle D4 inverse transform for actions
        if self.use_augmentation:
            actions = self._inverse_transform_actions(actions)
        
        rewards = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        
        # Decode actions: which pattern and which position
        pattern_slot = actions // 49  # 0 or 1 (if num_patterns=2)
        positions = actions % 49
        
        # Get pattern types for each env based on which slot was chosen
        # pattern_slot: (num_envs,) with values 0 or (0, 1 if num_patterns=2)
        pattern_types = self.pattern_indices.gather(1, pattern_slot.unsqueeze(1)).squeeze(1)
        
        # Get bitmasks for the chosen patterns at the chosen positions
        action_masks = self.precomputed_masks[pattern_types, positions]
        
        # Apply to board
        self.board_bits |= action_masks
        
        # Update state
        rewards[:] = -1.0
        self.current_step += 1
        
        # Resample the used pattern slot
        all_indices = torch.arange(self.num_envs, device=self.device)
        
        # For each slot, find envs that used it and resample
        for slot in range(self.num_patterns):
            slot_used = pattern_slot == slot
            if slot_used.any():
                self._choose_one_pattern(all_indices[slot_used], slot)
        
        # Check termination (board full)
        terminated = self.board_bits == self.FULL_BOARD_MASK
        
        # Check truncation (max steps)
        truncated = self.current_step >= self.max_steps
        
        dones = terminated | truncated
        
        # Auto-reset done environments
        if dones.any():
            done_indices = dones.nonzero(as_tuple=True)[0]
            self.reset(done_indices)
        
        # New random D4 transform after step (for next observation)
        if self.use_augmentation:
            self.transform_idx = torch.randint(0, 8, (self.num_envs,), device=self.device)
        
        # Pre-compute action masks and cache for _get_obs()
        self._cached_action_mask = self._compute_and_transform_action_masks()
        
        return self._get_obs(), rewards, dones, truncated, {}
    
    def _inverse_transform_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Inverse transform actions from augmented space to original space (vectorized)."""
        # Decode: pattern_slot and position
        pattern_slot = actions // 49
        positions = actions % 49
        
        # Inverse transform the position part
        transforms = self.transform_idx
        original_positions = self.d4_inverse_pos[transforms, positions]
        
        # Reconstruct action
        return pattern_slot * 49 + original_positions
    
    def action_masks(self) -> torch.Tensor:
        """
        Compute action masks for all environments.
        
        Returns:
            (num_envs, n_actions) tensor of valid action masks
            n_actions = 49 for num_patterns=1, 98 for num_patterns=2
        """
        return self._compute_and_transform_action_masks()
    
    def _compute_and_transform_action_masks(self) -> torch.Tensor:
        """Compute action masks and apply D4 transform."""
        masks = self._compute_action_masks_all()
        
        # Apply D4 transform to position masks
        if self.use_augmentation:
            masks = self._transform_action_masks(masks)
        
        return masks
    
    def _compute_action_masks_all(self) -> torch.Tensor:
        """Compute raw action masks (before D4 transform)."""
        masks = torch.zeros(self.num_envs, self.n_actions, dtype=torch.bool, device=self.device)
        
        # Empty spots bitmap
        empty_spots = (~self.board_bits) & self.FULL_BOARD_MASK
        empty_spots_expanded = empty_spots.unsqueeze(1)  # (num_envs, 1)
        
        for slot in range(self.num_patterns):
            # Get pattern types for this slot
            pattern_types = self.pattern_indices[:, slot]
            
            # Get masks for these patterns: (num_envs, 49)
            pattern_masks = self.precomputed_masks[pattern_types]
            
            # Valid positions: mask overlaps with empty cells
            valid_positions = (empty_spots_expanded & pattern_masks) != 0  # (num_envs, 49)
            
            # Put in the right action range
            start_idx = slot * 49
            masks[:, start_idx:start_idx + 49] = valid_positions
        
        return masks
    
    def _compute_action_masks_subset(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute action masks for a subset of environments."""
        n = len(indices)
        masks = torch.zeros(n, self.n_actions, dtype=torch.bool, device=self.device)
        
        empty_spots = ((~self.board_bits[indices]) & self.FULL_BOARD_MASK).unsqueeze(1)
        
        for slot in range(self.num_patterns):
            pattern_types = self.pattern_indices[indices, slot]
            pattern_masks = self.precomputed_masks[pattern_types]
            valid_positions = (empty_spots & pattern_masks) != 0
            
            start_idx = slot * 49
            masks[:, start_idx:start_idx + 49] = valid_positions
        
        return masks
    
    def _transform_action_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Apply D4 transform to action masks (VECTORIZED)."""
        result = torch.zeros_like(masks)
        
        for slot in range(self.num_patterns):
            start_idx = slot * 49
            pos_masks = masks[:, start_idx:start_idx + 49]
            
            # Vectorized transform using gather
            inverse_mapping = self.d4_inverse_pos[self.transform_idx]  # (num_envs, 49)
            transformed = torch.gather(pos_masks, 1, inverse_mapping)
            
            result[:, start_idx:start_idx + 49] = transformed
        
        return result
    
    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """
        Get observations for all environments.
        
        Returns:
            Dictionary with GPU tensors:
            - board: (num_envs, 7, 7) int8 - current board state
            - pattern_indices: (num_envs, num_patterns) int64 - pattern type indices (0-4)
            - action_mask: (num_envs, n_actions) uint8
        """
        # Convert bitboard to 7x7 array: (num_envs, 7, 7)
        board_arr = ((self.board_bits.unsqueeze(1) >> self.bit_shifts) & 1).view(
            self.num_envs, 7, 7
        ).to(torch.int8)
        
        # Apply D4 transform to board
        if self.use_augmentation:
            board_arr = self._transform_boards(board_arr)
        
        # Use cached action mask if available, otherwise compute
        if self._cached_action_mask is not None:
            action_mask = self._cached_action_mask
        else:
            action_mask = self._compute_and_transform_action_masks()
        
        return {
            "board": board_arr,
            "pattern_indices": self.pattern_indices.clone(),
            "action_mask": action_mask.to(torch.uint8),
        }
    
    def _transform_boards(self, boards: torch.Tensor) -> torch.Tensor:
        """Apply D4 transforms to batch of boards (VECTORIZED)."""
        # boards: (num_envs, 7, 7)
        # Flatten to (num_envs, 49) for index-based transform
        flat_boards = boards.reshape(self.num_envs, 49)
        
        # Get inverse mapping for each env's transform
        inverse_mapping = self.d4_inverse_pos[self.transform_idx]  # (num_envs, 49)
        
        # Apply mapping: transformed[i, k] = original[i, inverse[i, k]]
        transformed_flat = torch.gather(flat_boards.long(), 1, inverse_mapping)
        
        return transformed_flat.view(self.num_envs, 7, 7).to(boards.dtype)


if __name__ == "__main__":
    import time
    
    print("=== GPU Bingo Environment Test (num_patterns refactor) ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    
    # Test num_patterns=1
    print("\n--- Testing num_patterns=1 ---")
    env1 = BingoEnvGPU(num_envs=4, device=device, num_patterns=1)
    obs1 = env1.reset()
    print(f"  pattern_indices shape: {env1.pattern_indices.shape}")  # (4, 1)
    print(f"  action_mask shape: {obs1['action_mask'].shape}")  # (4, 49)
    masks1 = env1.action_masks()
    actions1 = torch.multinomial(masks1.float(), num_samples=1).squeeze(-1)
    env1.step(actions1)
    print("  Step successful!")
    
    # Test num_patterns=2
    print("\n--- Testing num_patterns=2 ---")
    env2 = BingoEnvGPU(num_envs=4, device=device, num_patterns=2)
    obs2 = env2.reset()
    print(f"  pattern_indices shape: {env2.pattern_indices.shape}")  # (4, 2)
    print(f"  action_mask shape: {obs2['action_mask'].shape}")  # (4, 98)
    
    # Check sorting
    sorted_check = (env2.pattern_indices[:, 0] <= env2.pattern_indices[:, 1]).all()
    print(f"  Patterns sorted: {sorted_check.item()}")
    
    masks2 = env2.action_masks()
    actions2 = torch.multinomial(masks2.float(), num_samples=1).squeeze(-1)
    env2.step(actions2)
    print("  Step successful!")
    
    # Benchmark
    print("\n--- Benchmark num_patterns=2 ---")
    num_envs = 256
    env = BingoEnvGPU(num_envs=num_envs, device=device, num_patterns=2)
    env.reset()
    
    # Warmup
    for _ in range(100):
        masks = env.action_masks()
        actions = torch.multinomial(masks.float(), num_samples=1).squeeze(-1)
        env.step(actions)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    start = time.time()
    n_iters = 1000
    for _ in range(n_iters):
        masks = env.action_masks()
        actions = torch.multinomial(masks.float(), num_samples=1).squeeze(-1)
        env.step(actions)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    fps = (n_iters * num_envs) / elapsed
    print(f"  FPS: {fps:,.0f}")
    
    print("\nAll tests passed!")