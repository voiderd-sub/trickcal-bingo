"""
GPU-Accelerated Vectorized Bingo Environment.

All environment operations run on GPU using PyTorch tensors.
Provides massive speedup by eliminating CPU-GPU data transfer and
enabling batched parallel execution.

OPTIMIZED: All D4 transforms use vectorized operations (torch.gather).
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict


class BingoEnvGPU:
    """
    PyTorch-based GPU Vectorized Bingo Environment.
    
    All state and operations use GPU tensors for maximum throughput.
    - board_bits: (num_envs,) int64 tensor - bitboard state
    - pattern_indices: (num_envs,) int64 tensor - current pattern index
    - All mask operations executed as batched GPU operations
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
        allow_store: bool = True,  # Set to False to disable pattern storage
    ):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.use_augmentation = use_augmentation
        self.allow_store = allow_store
        self.board_size = 7
        self.n_cells = 49
        self.max_steps = 49
        
        # Curriculum settings
        self.min_initial_turns = min_initial_turns
        self.max_initial_turns = max_initial_turns
        
        # Pattern probability (for sampling)
        self.pattern_prob = torch.tensor(
            [0.30, 0.30, 0.10, 0.15, 0.15],
            dtype=torch.float32,
            device=self.device
        )
        self.num_patterns = 5
        
        # Pre-compute bitmasks for all patterns at all positions
        # Shape: (num_patterns, 49) - each mask is a uint64
        self.precomputed_masks = self._generate_all_masks_gpu()
        
        # Pre-compute padded patterns for observation (7x7 each)
        # Shape: (num_patterns, 7, 7)
        self.obs_patterns = self._generate_obs_patterns_gpu()
        
        # D4 augmentation lookup tables
        if self.use_augmentation:
            self._init_d4_tables()
        
        # Environment state tensors
        self.board_bits = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.current_pattern_idx = torch.zeros(num_envs, dtype=torch.int64, device=self.device)
        self.stored_pattern_idx = torch.full((num_envs,), -1, dtype=torch.int64, device=self.device)
        self.store_remaining = torch.full((num_envs,), 2, dtype=torch.int32, device=self.device)
        self.current_step = torch.zeros(num_envs, dtype=torch.int32, device=self.device)
        self.is_first_turn = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        
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
            self.stored_pattern_idx.fill_(-1)
            self.store_remaining.fill_(2)
            self.current_step.zero_()
            self.is_first_turn.fill_(True)
        else:
            indices = env_indices
            self.board_bits[indices] = 0
            self.stored_pattern_idx[indices] = -1
            self.store_remaining[indices] = 2
            self.current_step[indices] = 0
            self.is_first_turn[indices] = True
        
        # Choose new patterns for reset envs
        self._choose_new_patterns(indices)
        
        # Apply curriculum (random prefill)
        if self.max_initial_turns > 0:
            self._prefill_boards(indices)
            
            # Safety checks: If prefill resulted in a full board (terminal state),
            # we must reset it to empty to avoid crashing (especially if allow_store=False)
            is_full = self.board_bits[indices] == self.FULL_BOARD_MASK
            if is_full.any():
                bad_indices = indices[is_full]
                # Reset to clean state (effecitvely rejecting the prefill)
                self.board_bits[bad_indices] = 0
                self.stored_pattern_idx[bad_indices] = -1
                self.store_remaining[bad_indices] = 2
                self.current_step[bad_indices] = 0
                self.is_first_turn[bad_indices] = True
                self._choose_new_patterns(bad_indices)
        
        # Random D4 transform for augmentation
        if self.use_augmentation:
            self.transform_idx[indices] = torch.randint(0, 8, (len(indices),), device=self.device)
        
        # Pre-compute action masks and cache for _get_obs()
        self._cached_action_mask = self._compute_and_transform_action_masks()
        
        return self._get_obs()
    
    def _choose_new_patterns(self, indices: torch.Tensor):
        """Sample new patterns for specified environments."""
        n = len(indices)
        # Sample from categorical distribution
        new_patterns = torch.multinomial(
            self.pattern_prob.expand(n, -1),
            num_samples=1
        ).squeeze(-1)
        self.current_pattern_idx[indices] = new_patterns
    
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
            
            # Get valid actions for active envs
            masks = self._compute_action_masks_subset(active_indices)
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
            
            # Apply moves
            pattern_idx = self.current_pattern_idx[valid_envs]
            # Gather masks for each env's pattern and action
            action_masks = self.precomputed_masks[pattern_idx, actions]
            self.board_bits[valid_envs] |= action_masks
            
            # Choose new patterns
            self._choose_new_patterns(valid_envs)
            
            # Decrement turns
            num_turns[active] -= 1
    
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Execute actions for all environments.
        
        Args:
            actions: (num_envs,) tensor of actions (0-48 for positions, 49 for store)
        
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
        
        # Separate store actions from position actions
        is_store = actions == 49
        is_position = ~is_store
        
        # Handle store actions
        if is_store.any():
            self._handle_store_actions(is_store)
        
        # Handle position actions
        if is_position.any():
            self._handle_position_actions(actions, is_position, rewards)
        
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
        result = actions.clone()
        
        # Only transform position actions (not store action 49)
        is_position = actions < 49
        if is_position.any():
            pos_envs = is_position.nonzero(as_tuple=True)[0]
            pos_actions = actions[pos_envs]
            transforms = self.transform_idx[pos_envs]
            
            # Vectorized: use advanced indexing
            result[pos_envs] = self.d4_inverse_pos[transforms, pos_actions]
        
        return result
    
    def _handle_store_actions(self, is_store: torch.Tensor):
        """Handle store/swap actions."""
        store_envs = is_store.nonzero(as_tuple=True)[0]
        
        # Check which envs have no stored pattern (store new)
        no_stored = self.stored_pattern_idx[store_envs] == -1
        has_stored = ~no_stored
        
        # Store new pattern
        if no_stored.any():
            store_new_envs = store_envs[no_stored]
            self.stored_pattern_idx[store_new_envs] = self.current_pattern_idx[store_new_envs]
            self._choose_new_patterns(store_new_envs)
        
        # Swap patterns
        if has_stored.any():
            swap_envs = store_envs[has_stored]
            temp = self.current_pattern_idx[swap_envs].clone()
            self.current_pattern_idx[swap_envs] = self.stored_pattern_idx[swap_envs]
            self.stored_pattern_idx[swap_envs] = temp
        
        # Decrement store remaining
        self.store_remaining[store_envs] -= 1
    
    def _handle_position_actions(self, actions: torch.Tensor, is_position: torch.Tensor, rewards: torch.Tensor):
        """Handle position placement actions."""
        pos_envs = is_position.nonzero(as_tuple=True)[0]
        pos_actions = actions[pos_envs]
        
        # Get mask for each action
        pattern_idx = self.current_pattern_idx[pos_envs]
        action_masks = self.precomputed_masks[pattern_idx, pos_actions]
        
        # Apply to board
        self.board_bits[pos_envs] |= action_masks
        
        # Update state
        rewards[pos_envs] = -1.0
        self.current_step[pos_envs] += 1
        self.is_first_turn[pos_envs] = False
        self.store_remaining[pos_envs] = 1
        
        # Choose new patterns
        self._choose_new_patterns(pos_envs)
    
    def action_masks(self) -> torch.Tensor:
        """
        Compute action masks for all environments.
        
        Returns:
            (num_envs, 50) tensor of valid action masks
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
        # Get masks for current patterns: (num_envs, 49)
        pattern_masks = self.precomputed_masks[self.current_pattern_idx]
        
        # Valid positions: mask overlaps with empty cells
        # empty_spots = ~board_bits (masked to 49 bits)
        empty_spots = (~self.board_bits) & self.FULL_BOARD_MASK
        empty_spots = empty_spots.unsqueeze(1)  # (num_envs, 1)
        
        # Broadcast AND
        valid_positions = (empty_spots & pattern_masks) != 0  # (num_envs, 49)
        
        # Store action validity
        # If allow_store is False, store action is always invalid
        if self.allow_store:
            can_store = self.store_remaining > 0
            same_pattern = self.stored_pattern_idx == self.current_pattern_idx
            has_stored = self.stored_pattern_idx != -1
            cant_swap = has_stored & same_pattern
            store_valid = can_store & ~cant_swap
        else:
            store_valid = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
        # Combine
        masks = torch.zeros(self.num_envs, 50, dtype=torch.bool, device=self.device)
        masks[:, :49] = valid_positions
        masks[:, 49] = store_valid
        
        return masks
    
    def _compute_action_masks_subset(self, indices: torch.Tensor) -> torch.Tensor:
        """Compute action masks for a subset of environments."""
        n = len(indices)
        
        pattern_masks = self.precomputed_masks[self.current_pattern_idx[indices]]
        empty_spots = ((~self.board_bits[indices]) & self.FULL_BOARD_MASK).unsqueeze(1)
        valid_positions = (empty_spots & pattern_masks) != 0
        
        can_store = self.store_remaining[indices] > 0
        same_pattern = self.stored_pattern_idx[indices] == self.current_pattern_idx[indices]
        has_stored = self.stored_pattern_idx[indices] != -1
        cant_swap = has_stored & same_pattern
        store_valid = can_store & ~cant_swap
        
        masks = torch.zeros(n, 50, dtype=torch.bool, device=self.device)
        masks[:, :49] = valid_positions
        masks[:, 49] = store_valid
        
        return masks
    
    def _transform_action_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Apply D4 transform to action masks (VECTORIZED)."""
        # Position masks only (not store)
        pos_masks = masks[:, :49]  # (num_envs, 49)
        store_mask = masks[:, 49:]  # (num_envs, 1)
        
        # Vectorized transform using gather
        # Get inverse mapping for each env's transform
        inverse_mapping = self.d4_inverse_pos[self.transform_idx]  # (num_envs, 49)
        
        # Apply mapping: transformed[i, j] = pos_masks[i, inverse_mapping[i, j]]
        transformed = torch.gather(pos_masks, 1, inverse_mapping)
        
        return torch.cat([transformed, store_mask], dim=1)
    
    def _get_obs(self) -> Dict[str, torch.Tensor]:
        """
        Get observations for all environments.
        
        Returns:
            Dictionary with GPU tensors.
        """
        # Convert bitboard to 7x7 array: (num_envs, 7, 7)
        # board_arr[i, j] = (board_bits >> (i*7+j)) & 1
        board_arr = ((self.board_bits.unsqueeze(1) >> self.bit_shifts) & 1).view(
            self.num_envs, 7, 7
        ).to(torch.int8)
        
        # Current pattern: (num_envs, 7, 7)
        curr_pattern = self.obs_patterns[self.current_pattern_idx]
        
        # Stored pattern: (num_envs, 7, 7)
        stored_pattern = torch.zeros(self.num_envs, 7, 7, dtype=torch.int8, device=self.device)
        has_stored = self.stored_pattern_idx >= 0
        if has_stored.any():
            stored_pattern[has_stored] = self.obs_patterns[self.stored_pattern_idx[has_stored]]
        
        # Apply D4 transforms
        if self.use_augmentation:
            board_arr = self._transform_boards(board_arr)
            curr_pattern = self._transform_boards(curr_pattern)
            stored_pattern = self._transform_boards(stored_pattern)
        
        # Has stored flag
        has_stored_float = (self.stored_pattern_idx >= 0).float().unsqueeze(1)
        
        # Use cached action mask if available, otherwise compute
        if self._cached_action_mask is not None:
            action_mask = self._cached_action_mask
        else:
            action_mask = self._compute_and_transform_action_masks()
        
        return {
            "board": board_arr,
            "pattern": curr_pattern,
            "stored_pattern": stored_pattern,
            "has_stored": has_stored_float,
            "cost": torch.ones(self.num_envs, dtype=torch.float32, device=self.device),
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
    from collections import defaultdict
    
    print("=== GPU Bingo Environment Test (OPTIMIZED) ===")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    
    num_envs = 256
    env = BingoEnvGPU(num_envs=num_envs, device=device)
    
    print(f"\nEnvironment created with {num_envs} parallel envs on {device}")
    
    # Warmup
    print("\nWarmup...")
    env.reset()
    for _ in range(100):
        masks = env.action_masks()
        actions = torch.multinomial(masks.float(), num_samples=1).squeeze(-1)
        env.step(actions)
    
    if device == 'cuda':
        torch.cuda.synchronize()
    
    # === DETAILED PROFILING ===
    print("\n" + "="*60)
    print("DETAILED STEP-BY-STEP PROFILING")
    print("="*60)
    
    timings = defaultdict(float)
    num_iterations = 1000
    
    env.reset()
    
    for i in range(num_iterations):
        # 1. action_masks() call
        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        masks = env.action_masks()
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['action_masks'] += t1 - t0
        
        # 2. Sample actions (multinomial)
        t0 = time.perf_counter()
        actions = torch.multinomial(masks.float(), num_samples=1).squeeze(-1)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['multinomial'] += t1 - t0
        
        # 3. step() call
        t0 = time.perf_counter()
        obs, rewards, dones, truncated, _ = env.step(actions)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        timings['step'] += t1 - t0
    
    total_time = sum(timings.values())
    total_steps = num_iterations * num_envs
    
    print(f"\nTotal iterations: {num_iterations}")
    print(f"Total steps: {total_steps:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Overall FPS: {total_steps / total_time:,.0f}")
    
    print("\n--- Time Breakdown ---")
    for name, t in sorted(timings.items(), key=lambda x: -x[1]):
        pct = 100 * t / total_time
        per_iter_us = 1e6 * t / num_iterations
        print(f"  {name:20s}: {t:.3f}s ({pct:5.1f}%) | {per_iter_us:.1f} µs/iter")
    
    # === DEEPER ANALYSIS OF ACTION_MASKS ===
    print("\n" + "="*60)
    print("DEEPER ANALYSIS: action_masks() breakdown")
    print("="*60)
    
    sub_timings = defaultdict(float)
    
    for i in range(num_iterations):
        # Simulate action_masks() internals
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 1. Compute raw masks
        t0 = time.perf_counter()
        pattern_masks = env.precomputed_masks[env.current_pattern_idx]
        empty_spots = (~env.board_bits) & env.FULL_BOARD_MASK
        empty_spots_bcast = empty_spots.unsqueeze(1)
        valid_positions = (empty_spots_bcast & pattern_masks) != 0
        can_store = env.store_remaining > 0
        same_pattern = env.stored_pattern_idx == env.current_pattern_idx
        has_stored = env.stored_pattern_idx != -1
        cant_swap = has_stored & same_pattern
        store_valid = can_store & ~cant_swap
        masks_raw = torch.zeros(env.num_envs, 50, dtype=torch.bool, device=env.device)
        masks_raw[:, :49] = valid_positions
        masks_raw[:, 49] = store_valid
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        sub_timings['compute_raw_masks'] += t1 - t0
        
        # 2. D4 transform
        t0 = time.perf_counter()
        pos_masks = masks_raw[:, :49]
        store_mask = masks_raw[:, 49:]
        inverse_mapping = env.d4_inverse_pos[env.transform_idx]
        transformed = torch.gather(pos_masks, 1, inverse_mapping)
        final_masks = torch.cat([transformed, store_mask], dim=1)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        sub_timings['d4_transform_masks'] += t1 - t0
    
    print(f"\naction_masks() internals (over {num_iterations} iters):")
    for name, t in sorted(sub_timings.items(), key=lambda x: -x[1]):
        pct = 100 * t / sum(sub_timings.values())
        per_iter_us = 1e6 * t / num_iterations
        print(f"  {name:25s}: {t:.3f}s ({pct:5.1f}%) | {per_iter_us:.1f} µs/iter")
    
    # === DEEPER ANALYSIS OF STEP ===
    print("\n" + "="*60)
    print("DEEPER ANALYSIS: step() breakdown")
    print("="*60)
    
    step_timings = defaultdict(float)
    env.reset()
    
    for i in range(num_iterations):
        masks = env.action_masks()
        actions = torch.multinomial(masks.float(), num_samples=1).squeeze(-1)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 1. Inverse transform actions
        t0 = time.perf_counter()
        is_position = actions < 49
        result = actions.clone()
        if is_position.any():
            pos_envs = is_position.nonzero(as_tuple=True)[0]
            pos_actions = actions[pos_envs]
            transforms = env.transform_idx[pos_envs]
            result[pos_envs] = env.d4_inverse_pos[transforms, pos_actions]
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_timings['inverse_transform_action'] += t1 - t0
        
        # 2. Position actions handling
        t0 = time.perf_counter()
        is_store = result == 49
        is_position = ~is_store
        if is_position.any():
            pos_envs = is_position.nonzero(as_tuple=True)[0]
            pos_actions = result[pos_envs]
            pattern_idx = env.current_pattern_idx[pos_envs]
            action_masks_val = env.precomputed_masks[pattern_idx, pos_actions]
            # Would update board_bits here
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_timings['position_actions'] += t1 - t0
        
        # 3. New transform generation
        t0 = time.perf_counter()
        _ = torch.randint(0, 8, (env.num_envs,), device=env.device)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_timings['randint_transform'] += t1 - t0
        
        # 4. Observation generation (the real step)
        t0 = time.perf_counter()
        obs, rewards, dones, truncated, _ = env.step(actions)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_timings['full_step'] += t1 - t0
    
    print(f"\nstep() internals (over {num_iterations} iters):")
    for name, t in sorted(step_timings.items(), key=lambda x: -x[1]):
        per_iter_us = 1e6 * t / num_iterations
        print(f"  {name:25s}: {t:.3f}s | {per_iter_us:.1f} µs/iter")
    
    # === OBSERVATION GENERATION ANALYSIS ===
    print("\n" + "="*60)
    print("DEEPER ANALYSIS: _get_obs() breakdown")
    print("="*60)
    
    obs_timings = defaultdict(float)
    
    for i in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
        
        # 1. Bitboard to array
        t0 = time.perf_counter()
        board_arr = ((env.board_bits.unsqueeze(1) >> env.bit_shifts) & 1).view(
            env.num_envs, 7, 7
        ).to(torch.int8)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        obs_timings['bitboard_to_array'] += t1 - t0
        
        # 2. Pattern lookup
        t0 = time.perf_counter()
        curr_pattern = env.obs_patterns[env.current_pattern_idx]
        stored_pattern = torch.zeros(env.num_envs, 7, 7, dtype=torch.int8, device=env.device)
        has_stored = env.stored_pattern_idx >= 0
        if has_stored.any():
            stored_pattern[has_stored] = env.obs_patterns[env.stored_pattern_idx[has_stored]]
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        obs_timings['pattern_lookup'] += t1 - t0
        
        # 3. Board transform
        t0 = time.perf_counter()
        flat_boards = board_arr.reshape(env.num_envs, 49)
        inverse_mapping = env.d4_inverse_pos[env.transform_idx]
        transformed_flat = torch.gather(flat_boards.long(), 1, inverse_mapping)
        board_transformed = transformed_flat.view(env.num_envs, 7, 7).to(board_arr.dtype)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        obs_timings['board_transform'] += t1 - t0
        
        # 4. Action masks (called inside _get_obs!)
        t0 = time.perf_counter()
        _ = env.action_masks().to(torch.uint8)
        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        obs_timings['action_masks_in_obs'] += t1 - t0
    
    print(f"\n_get_obs() internals (over {num_iterations} iters):")
    for name, t in sorted(obs_timings.items(), key=lambda x: -x[1]):
        pct = 100 * t / sum(obs_timings.values())
        per_iter_us = 1e6 * t / num_iterations
        print(f"  {name:25s}: {t:.3f}s ({pct:5.1f}%) | {per_iter_us:.1f} µs/iter")
    
    # === VRAM PROFILING WITH MODEL ===
    print("\n" + "="*60)
    print("VRAM PROFILING WITH POLICY MODEL")
    print("="*60)
    
    if device == 'cuda':
        from gymnasium import spaces
        import sys
        sys.path.insert(0, '/home/swkim/RL')
        from bingo_policy import BingoCNNExtractor
        
        def get_vram_mb():
            return torch.cuda.memory_allocated() / 1024**2
        
        def get_vram_reserved_mb():
            return torch.cuda.memory_reserved() / 1024**2
        
        # Clear GPU memory
        del env
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        baseline_vram = get_vram_mb()
        print(f"\nBaseline VRAM: {baseline_vram:.1f} MB")
        
        # Create observation space for model
        obs_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "stored_pattern": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "has_stored": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            "cost": spaces.Box(low=0.0, high=10.0, shape=(), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(50,), dtype=np.uint8),
        })
        
        # Create model (same config as training)
        model = BingoCNNExtractor(
            obs_space, 
            features_dim=256,
            hidden_channels=64,
            num_res_blocks=3,
            kernel_size=3,
            scalar_embed_dim=32
        ).to(device)
        
        model_vram = get_vram_mb() - baseline_vram
        print(f"Model VRAM: {model_vram:.1f} MB")
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test different num_envs
        test_num_envs = [64, 128, 256, 512, 1024, 2048, 4096, 8192]
        
        print(f"\n{'num_envs':>10} | {'Env VRAM':>10} | {'Peak (Train)':>14} | {'FPS (inf)':>12} | {'FPS (train)':>12}")
        print("-" * 80)
        
        results = []
        
        for n_envs in test_num_envs:
            # Clear cache and reset model
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Recreate model with optimizer for accurate VRAM measurement
            del model
            torch.cuda.empty_cache()
            
            model = BingoCNNExtractor(
                obs_space, 
                features_dim=256,
                hidden_channels=64,
                num_res_blocks=3,
                kernel_size=3,
                scalar_embed_dim=32
            ).to(device)
            
            # Add MLP heads like PPO (actor + critic)
            policy_head = torch.nn.Sequential(
                torch.nn.Linear(256, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 50)  # 50 actions
            ).to(device)
            
            value_head = torch.nn.Sequential(
                torch.nn.Linear(256, 256),
                torch.nn.GELU(),
                torch.nn.Linear(256, 128),
                torch.nn.GELU(),
                torch.nn.Linear(128, 1)
            ).to(device)
            
            # Optimizer (Adam uses 2x memory for momentum + variance)
            all_params = list(model.parameters()) + list(policy_head.parameters()) + list(value_head.parameters())
            optimizer = torch.optim.Adam(all_params, lr=1e-3)
            
            before_vram = get_vram_mb()
            
            try:
                # Create environment
                test_env = BingoEnvGPU(num_envs=n_envs, device=device)
                test_env.reset()
                
                env_vram = get_vram_mb() - before_vram
                
                # === INFERENCE BENCHMARK ===
                torch.cuda.synchronize()
                start_inf = time.perf_counter()
                n_iters = 200
                
                for _ in range(n_iters):
                    masks = test_env.action_masks()
                    obs = test_env._get_obs()
                    batch_obs = {
                        "board": obs["board"].float(),
                        "pattern": obs["pattern"].float(),
                        "stored_pattern": obs["stored_pattern"].float(),
                        "has_stored": obs["has_stored"],
                    }
                    with torch.no_grad():
                        features = model(batch_obs)
                        logits = policy_head(features)
                        values = value_head(features)
                    
                    actions = torch.multinomial(masks.float(), num_samples=1).squeeze(-1)
                    test_env.step(actions)
                
                torch.cuda.synchronize()
                elapsed_inf = time.perf_counter() - start_inf
                fps_inf = (n_iters * n_envs) / elapsed_inf
                
                # === TRAINING BENCHMARK (with backprop) ===
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start_train = time.perf_counter()
                n_train_iters = 100
                
                for _ in range(n_train_iters):
                    masks = test_env.action_masks()
                    obs = test_env._get_obs()
                    batch_obs = {
                        "board": obs["board"].float(),
                        "pattern": obs["pattern"].float(),
                        "stored_pattern": obs["stored_pattern"].float(),
                        "has_stored": obs["has_stored"],
                    }
                    
                    # Forward pass (with grad)
                    features = model(batch_obs)
                    logits = policy_head(features)
                    values = value_head(features)
                    
                    # Simulate PPO loss (simplified)
                    # Action log probs
                    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    actions = torch.multinomial(masks.float(), num_samples=1).squeeze(-1)
                    action_log_probs = log_probs.gather(1, actions.unsqueeze(-1)).squeeze(-1)
                    
                    # Dummy targets
                    returns = torch.randn(n_envs, device=device)
                    advantages = torch.randn(n_envs, device=device)
                    
                    # Policy loss
                    policy_loss = -(action_log_probs * advantages).mean()
                    
                    # Value loss
                    value_loss = torch.nn.functional.mse_loss(values.squeeze(-1), returns)
                    
                    # Total loss
                    loss = policy_loss + 0.5 * value_loss
                    
                    # Backward
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Step env
                    with torch.no_grad():
                        test_env.step(actions)
                
                torch.cuda.synchronize()
                elapsed_train = time.perf_counter() - start_train
                fps_train = (n_train_iters * n_envs) / elapsed_train
                
                peak_vram = torch.cuda.max_memory_allocated() / 1024**2
                
                print(f"{n_envs:>10} | {env_vram:>8.1f} MB | {peak_vram:>12.1f} MB | {fps_inf:>10,.0f} | {fps_train:>10,.0f}")
                
                results.append({
                    'num_envs': n_envs,
                    'env_vram': env_vram,
                    'peak_vram': peak_vram,
                    'fps_inf': fps_inf,
                    'fps_train': fps_train
                })
                
                del test_env
                del policy_head
                del value_head
                del optimizer
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"{n_envs:>10} | {'OOM':>10} | {'OOM':>14} | {'OOM':>12} | {'OOM':>12}")
                    torch.cuda.empty_cache()
                else:
                    raise
        
        # Summary
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        if results:
            # Find best FPS within VRAM budget (use 80% of total as safe limit)
            gpu_total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
            safe_budget = gpu_total_memory * 0.8  # 80% of total
            
            print(f"\nGPU Total Memory: {gpu_total_memory:.0f} MB")
            print(f"Safe Budget (80%): {safe_budget:.0f} MB")
            
            valid_results = [r for r in results if r['peak_vram'] < safe_budget]
            
            if valid_results:
                best = max(valid_results, key=lambda x: x['fps_train'])
                print(f"\nRecommended num_envs: {best['num_envs']}")
                print(f"  - Peak VRAM (training): {best['peak_vram']:.0f} MB")
                print(f"  - Inference FPS: {best['fps_inf']:,.0f}")
                print(f"  - Training FPS: {best['fps_train']:,.0f}")
            else:
                print("\nNo configuration fits within VRAM budget!")
    
    else:
        print("\nSkipping VRAM profiling (CPU mode)")