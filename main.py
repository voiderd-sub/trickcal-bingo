"""
Bingo AI Assistant GUI.

Uses trained RL models to suggest optimal pattern placements.
Supports both store-allowed and no-store models.
Displays D4 symmetry-averaged probabilities with center-only visualization.
"""

import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
    QPushButton, QMessageBox, QHeaderView, QAbstractItemView, QLabel,
    QGroupBox, QFrame
)
from PySide6.QtGui import QColor, QBrush, QPixmap, QPainter, QFont
from PySide6.QtCore import Qt, QSize

# Import policy network components
from bingo_policy import BingoCNNExtractor


# =============================================================================
# Pattern Definitions (same as BingoEnvGPU)
# =============================================================================

PATTERNS = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8),  # Plus
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.int8),  # X
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int8),  # 3x3
    np.array([[1, 1, 1, 1, 1, 1, 1]], dtype=np.int8),            # Horizontal
    np.array([[1], [1], [1], [1], [1], [1], [1]], dtype=np.int8), # Vertical
]

PATTERN_NAMES = ["십자(+)", "X자", "3×3", "가로줄", "세로줄"]


# =============================================================================
# D4 Symmetry Transformations (CPU version)
# =============================================================================

def create_d4_tables():
    """Create D4 symmetry transformation lookup tables."""
    d4_forward_pos = np.zeros((8, 49), dtype=np.int64)
    d4_inverse_pos = np.zeros((8, 49), dtype=np.int64)
    d4_inverse_idx = np.array([0, 3, 2, 1, 4, 7, 6, 5], dtype=np.int64)
    
    n = 6  # board_size - 1
    
    for pos in range(49):
        row, col = pos // 7, pos % 7
        
        # Transform 0: identity
        d4_forward_pos[0, pos] = pos
        # Transform 1: rot90 CCW
        new_row, new_col = col, n - row
        d4_forward_pos[1, pos] = new_row * 7 + new_col
        # Transform 2: rot180
        new_row, new_col = n - row, n - col
        d4_forward_pos[2, pos] = new_row * 7 + new_col
        # Transform 3: rot270 CCW
        new_row, new_col = n - col, row
        d4_forward_pos[3, pos] = new_row * 7 + new_col
        # Transform 4: flip horizontal
        new_row, new_col = row, n - col
        d4_forward_pos[4, pos] = new_row * 7 + new_col
        # Transform 5: flip + rot90
        new_row, new_col = col, row
        d4_forward_pos[5, pos] = new_row * 7 + new_col
        # Transform 6: flip + rot180
        new_row, new_col = n - row, col
        d4_forward_pos[6, pos] = new_row * 7 + new_col
        # Transform 7: flip + rot270
        new_row, new_col = n - col, n - row
        d4_forward_pos[7, pos] = new_row * 7 + new_col
    
    # Compute inverse tables
    for t in range(8):
        inv_t = d4_inverse_idx[t]
        d4_inverse_pos[t] = d4_forward_pos[inv_t]
    
    return d4_forward_pos, d4_inverse_pos


D4_FORWARD_POS, D4_INVERSE_POS = create_d4_tables()


def transform_board(board: np.ndarray, transform_idx: int) -> np.ndarray:
    """Apply D4 transform to a 7x7 board."""
    flat = board.flatten()
    inverse_mapping = D4_INVERSE_POS[transform_idx]
    transformed = flat[inverse_mapping]
    return transformed.reshape(7, 7)


def inverse_transform_probs(probs: np.ndarray, transform_idx: int) -> np.ndarray:
    """Inverse transform position probabilities back to original coordinate system."""
    # probs: (50,) - 49 positions + 1 store action
    pos_probs = probs[:49]
    store_prob = probs[49]
    
    # inverse_mapping[new_pos] = original_pos
    # We want: original_probs[original_pos] = transformed_probs[new_pos]
    # So: original_probs = transformed_probs[forward_mapping]
    forward_mapping = D4_FORWARD_POS[transform_idx]
    original_pos_probs = pos_probs[forward_mapping]
    
    return np.concatenate([original_pos_probs, [store_prob]])


def compute_dynamic_orbits(board: np.ndarray, pattern_idx: int = None) -> list:
    """
    Compute D4 symmetry orbits considering the current board state and pattern.
    
    Two positions P1 and P2 are in the same orbit if:
    - Placing the pattern at P1 gives result R1
    - Placing the pattern at P2 gives result R2
    - There exists a D4 transform T such that T(R1) == R2
    
    This means the resulting boards after placement are D4-equivalent.
    
    Args:
        board: Current 7x7 board state
        pattern_idx: Index of the pattern being placed (optional, for pattern-specific orbits)
    
    Returns:
        List of sets, where each set contains positions that are equivalent.
    """
    def board_hash(board_arr):
        return tuple(board_arr.flatten().tolist())
    
    def get_canonical_hash(board_arr):
        """Get the minimum hash over all 8 D4 transforms (canonical form)."""
        hashes = []
        for t in range(8):
            transformed = transform_board(board_arr, t)
            hashes.append(board_hash(transformed))
        return min(hashes)
    
    # If no pattern specified, use simple transform-based orbits
    if pattern_idx is None:
        # Fall back to checking which transforms preserve the board
        current_hash = board_hash(board)
        board_preserving_transforms = []
        for t in range(8):
            transformed = transform_board(board, t)
            if board_hash(transformed) == current_hash:
                board_preserving_transforms.append(t)
        
        visited = set()
        orbits = []
        for pos in range(49):
            if pos in visited:
                continue
            orbit = set()
            orbit.add(pos)
            for t in board_preserving_transforms:
                new_pos = D4_FORWARD_POS[t, pos]
                orbit.add(new_pos)
            orbits.append(orbit)
            visited.update(orbit)
        return orbits
    
    # Get pattern info
    pattern = PATTERNS[pattern_idx]
    ph, pw = pattern.shape
    offset_h, offset_w = ph // 2, pw // 2
    
    # For each position, compute the resulting board and its canonical hash
    position_to_canonical = {}
    
    for pos in range(49):
        row, col = pos // 7, pos % 7
        
        # Compute resulting board after placing pattern at this position
        result_board = board.copy()
        for pr in range(ph):
            for pc in range(pw):
                if pattern[pr, pc] == 1:
                    br = row - offset_h + pr
                    bc = col - offset_w + pc
                    if 0 <= br < 7 and 0 <= bc < 7:
                        result_board[br, bc] = 1
        
        # Get canonical hash
        position_to_canonical[pos] = get_canonical_hash(result_board)
    
    # Group positions by their canonical hash
    canonical_to_positions = {}
    for pos, canonical in position_to_canonical.items():
        if canonical not in canonical_to_positions:
            canonical_to_positions[canonical] = set()
        canonical_to_positions[canonical].add(pos)
    
    return list(canonical_to_positions.values())


def normalize_probs_by_dynamic_orbit(probs: np.ndarray, board: np.ndarray, pattern_idx: int = None) -> np.ndarray:
    """
    Normalize probabilities so that symmetric positions (considering board state and pattern)
    have the same probability.
    
    For each orbit (computed dynamically based on board and pattern), sum the probabilities
    of all positions in the orbit, then assign that sum to all positions.
    
    Args:
        probs: (50,) array of probabilities
        board: Current 7x7 board state
        pattern_idx: Index of pattern being placed
    
    Returns:
        (50,) array with orbit-normalized probabilities
    """
    normalized = probs.copy()
    
    # Compute orbits dynamically based on current board and pattern
    orbits = compute_dynamic_orbits(board, pattern_idx)
    
    for orbit in orbits:
        # Sum probabilities in this orbit
        orbit_sum = sum(probs[pos] for pos in orbit)
        # Assign sum to all positions in orbit
        for pos in orbit:
            normalized[pos] = orbit_sum
    
    # Store action (49) is unchanged
    return normalized


# =============================================================================
# Policy Network (copy from train.py for CPU loading)
# =============================================================================

class MaskablePPOPolicy(nn.Module):
    """Actor-Critic policy for MaskablePPO."""
    
    def __init__(
        self,
        observation_space,
        action_dim: int = 50,
        features_dim: int = 256,
        hidden_channels: int = 64,
        num_res_blocks: int = 3,
        kernel_size: int = 3,
        scalar_embed_dim: int = 32,
        pi_layers: list = [256, 128],
        vf_layers: list = [256, 128],
    ):
        super().__init__()
        
        self.features_extractor = BingoCNNExtractor(
            observation_space,
            features_dim=features_dim,
            hidden_channels=hidden_channels,
            num_res_blocks=num_res_blocks,
            kernel_size=kernel_size,
            scalar_embed_dim=scalar_embed_dim,
        )
        
        # Policy head
        pi_layers_list = []
        in_dim = features_dim
        for out_dim in pi_layers:
            pi_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim
        pi_layers_list.append(nn.Linear(in_dim, action_dim))
        self.policy_head = nn.Sequential(*pi_layers_list)
        
        # Value head
        vf_layers_list = []
        in_dim = features_dim
        for out_dim in vf_layers:
            vf_layers_list.extend([
                nn.Linear(in_dim, out_dim),
                nn.GELU(),
            ])
            in_dim = out_dim
        vf_layers_list.append(nn.Linear(in_dim, 1))
        self.value_head = nn.Sequential(*vf_layers_list)
    
    def forward(self, obs):
        """Forward pass returning logits and values."""
        features = self.features_extractor(obs)
        logits = self.policy_head(features)
        values = self.value_head(features)
        return logits, values


# =============================================================================
# Simple CPU Environment State
# =============================================================================

class BingoState:
    """Simple bingo state management for CPU inference."""
    
    def __init__(self, allow_store: bool = True):
        self.board_size = 7
        self.allow_store = allow_store
        self.reset()
    
    def reset(self):
        self.board = np.zeros((7, 7), dtype=np.int8)
        self.stored_pattern_idx = -1
        self.store_remaining = 2  # Only used for model action masking
        self.is_first_turn = True
        self.current_step = 0
        self.history = []
    
    def get_pattern_obs(self, pattern_idx: int) -> np.ndarray:
        """Get 7x7 padded pattern for observation."""
        pattern = PATTERNS[pattern_idx]
        padded = np.zeros((7, 7), dtype=np.int8)
        ph, pw = pattern.shape
        oh, ow = (7 - ph) // 2, (7 - pw) // 2
        padded[oh:oh+ph, ow:ow+pw] = pattern
        return padded
    
    def get_stored_pattern_obs(self) -> np.ndarray:
        """Get stored pattern observation (zeros if none)."""
        if self.stored_pattern_idx < 0:
            return np.zeros((7, 7), dtype=np.int8)
        return self.get_pattern_obs(self.stored_pattern_idx)
    
    def get_action_mask(self, pattern_idx: int) -> np.ndarray:
        """Compute valid action mask for given pattern."""
        pattern = PATTERNS[pattern_idx]
        mask = np.zeros(50, dtype=bool)
        
        ph, pw = pattern.shape
        offset_h, offset_w = ph // 2, pw // 2
        
        for r in range(7):
            for c in range(7):
                # Check if pattern overlaps with any empty cell
                has_empty_overlap = False
                for pr in range(ph):
                    for pc in range(pw):
                        if pattern[pr, pc] == 1:
                            board_r = r - offset_h + pr
                            board_c = c - offset_w + pc
                            if 0 <= board_r < 7 and 0 <= board_c < 7:
                                if self.board[board_r, board_c] == 0:
                                    has_empty_overlap = True
                                    break
                    if has_empty_overlap:
                        break
                
                if has_empty_overlap:
                    mask[r * 7 + c] = True
        
        # Store action (action 49) - for GUI, always allow unless same pattern is stored
        if self.allow_store:
            same_pattern = self.stored_pattern_idx == pattern_idx
            has_stored = self.stored_pattern_idx >= 0
            cant_swap = has_stored and same_pattern
            mask[49] = not cant_swap
        else:
            mask[49] = False
        
        return mask
    
    def get_model_action_mask(self, pattern_idx: int) -> np.ndarray:
        """Get action mask for model inference (considers store_remaining for proper masking)."""
        mask = self.get_action_mask(pattern_idx)
        
        # Apply store_remaining limit for model
        if self.allow_store:
            if self.store_remaining <= 0:
                mask[49] = False
        
        return mask
    
    def apply_pattern(self, pattern_idx: int, center_row: int, center_col: int):
        """Apply pattern to board at center position."""
        pattern = PATTERNS[pattern_idx]
        ph, pw = pattern.shape
        offset_h, offset_w = ph // 2, pw // 2
        
        for pr in range(ph):
            for pc in range(pw):
                if pattern[pr, pc] == 1:
                    board_r = center_row - offset_h + pr
                    board_c = center_col - offset_w + pc
                    if 0 <= board_r < 7 and 0 <= board_c < 7:
                        self.board[board_r, board_c] = 1
        
        self.current_step += 1
        self.is_first_turn = False
        self.store_remaining = 1  # Reset for next turn (model only)
    
    def store_pattern(self, pattern_idx: int):
        """Store the given pattern (no swap)."""
        self.stored_pattern_idx = pattern_idx
        self.store_remaining -= 1
    
    def swap_pattern(self, new_pattern_idx: int) -> int:
        """Swap current pattern with stored. Returns the retrieved pattern index."""
        old_stored = self.stored_pattern_idx
        self.stored_pattern_idx = new_pattern_idx
        self.store_remaining -= 1
        return old_stored
    
    def save_state(self):
        """Save current state to history."""
        self.history.append({
            'board': self.board.copy(),
            'stored_pattern_idx': self.stored_pattern_idx,
            'store_remaining': self.store_remaining,
            'is_first_turn': self.is_first_turn,
            'current_step': self.current_step,
        })
    
    def undo(self) -> bool:
        """Restore previous state. Returns True if successful."""
        if not self.history:
            return False
        
        state = self.history.pop()
        self.board = state['board']
        self.stored_pattern_idx = state['stored_pattern_idx']
        self.store_remaining = state['store_remaining']
        self.is_first_turn = state['is_first_turn']
        self.current_step = state['current_step']
        return True
    
    def to_dict(self) -> dict:
        """Serialize state to dictionary for saving."""
        return {
            'board': self.board.tolist(),
            'stored_pattern_idx': self.stored_pattern_idx,
            'store_remaining': self.store_remaining,
            'is_first_turn': self.is_first_turn,
            'current_step': self.current_step,
            'allow_store': self.allow_store,
            'history': [
                {
                    'board': h['board'].tolist(),
                    'stored_pattern_idx': h['stored_pattern_idx'],
                    'store_remaining': h['store_remaining'],
                    'is_first_turn': h['is_first_turn'],
                    'current_step': h['current_step'],
                }
                for h in self.history
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BingoState':
        """Deserialize state from dictionary."""
        state = cls(allow_store=data.get('allow_store', True))
        state.board = np.array(data.get('board', np.zeros((7, 7))), dtype=np.int8)
        state.stored_pattern_idx = data.get('stored_pattern_idx', -1)
        state.store_remaining = data.get('store_remaining', 2)
        state.is_first_turn = data.get('is_first_turn', True)
        state.current_step = data.get('current_step', 0)
        state.history = [
            {
                'board': np.array(h.get('board', np.zeros((7, 7))), dtype=np.int8),
                'stored_pattern_idx': h.get('stored_pattern_idx', -1),
                'store_remaining': h.get('store_remaining', 2),
                'is_first_turn': h.get('is_first_turn', True),
                'current_step': h.get('current_step', 0),
            }
            for h in data.get('history', [])
        ]
        return state


# =============================================================================
# Model Manager
# =============================================================================

class ModelManager:
    """Manages loading and switching between models."""
    
    def __init__(self):
        self.device = torch.device('cpu')
        self.models = {}
        self.current_model_name = None
        self.policy = None
        
        # Create observation space for policy
        from gymnasium import spaces
        self.obs_space = spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "stored_pattern": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "has_stored": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        
        self._load_models()
    
    def _create_policy(self) -> MaskablePPOPolicy:
        """Create policy network with default config."""
        return MaskablePPOPolicy(
            self.obs_space,
            features_dim=256,
            hidden_channels=64,
            num_res_blocks=3,
            kernel_size=3,
            scalar_embed_dim=32,
            pi_layers=[256, 128],
            vf_layers=[256, 128],
        )
    
    def _load_models(self):
        """Load both models."""
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        model_paths = {
            'store': os.path.join(base_path, 'model', 'best_model.pt'),
            'no_store': os.path.join(base_path, 'model', 'best_model_no_store.pt'),
        }
        
        for name, path in model_paths.items():
            if os.path.exists(path):
                policy = self._create_policy()
                policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                policy.eval()
                self.models[name] = policy
        # Set default model
        if 'store' in self.models:
            self.switch_model('store')
        elif 'no_store' in self.models:
            self.switch_model('no_store')
    
    def switch_model(self, name: str):
        """Switch to specified model."""
        if name in self.models:
            self.current_model_name = name
            self.policy = self.models[name]
            return True
        return False
    
    def get_d4_averaged_probs(
        self,
        board: np.ndarray,
        pattern_idx: int,
        stored_pattern_idx: int,
        action_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute D4 symmetry-averaged action probabilities.
        
        Args:
            board: Current board state (7x7)
            pattern_idx: Index of current pattern
            stored_pattern_idx: Index of stored pattern (-1 if none)
            action_mask: Valid action mask (50,) - required for proper probability computation
        
        Returns:
            (50,) array of averaged probabilities
        """
        if self.policy is None:
            return np.zeros(50)
        
        if action_mask is None:
            # Generate action mask if not provided
            action_mask = np.ones(50, dtype=bool)
        
        all_probs = []
        
        pattern_obs = self._get_pattern_obs(pattern_idx)
        stored_obs = self._get_pattern_obs(stored_pattern_idx) if stored_pattern_idx >= 0 else np.zeros((7, 7), dtype=np.int8)
        has_stored = 1.0 if stored_pattern_idx >= 0 else 0.0
        
        for transform_idx in range(8):
            # Transform observations
            transformed_board = transform_board(board, transform_idx)
            transformed_pattern = transform_board(pattern_obs, transform_idx)
            transformed_stored = transform_board(stored_obs, transform_idx)
            
            # Transform action mask: we need to transform positions 0-48
            # Forward transform maps original -> transformed
            transformed_mask = np.zeros(50, dtype=bool)
            for orig_pos in range(49):
                # Find where this position goes in transformed space
                new_pos = D4_FORWARD_POS[transform_idx, orig_pos]
                transformed_mask[new_pos] = action_mask[orig_pos]
            transformed_mask[49] = action_mask[49]  # Store action unchanged
            
            # Create observation tensors
            obs = {
                'board': torch.from_numpy(transformed_board).float().unsqueeze(0),
                'pattern': torch.from_numpy(transformed_pattern).float().unsqueeze(0),
                'stored_pattern': torch.from_numpy(transformed_stored).float().unsqueeze(0),
                'has_stored': torch.tensor([[has_stored]], dtype=torch.float32),
            }
            
            # Get logits
            with torch.no_grad():
                logits, _ = self.policy(obs)
            
            # Apply action mask: set invalid actions to -inf
            logits = logits.squeeze(0).clone()
            mask_tensor = torch.from_numpy(transformed_mask)
            logits[~mask_tensor] = float('-inf')
            
            # Compute probabilities (softmax over valid actions only)
            probs = F.softmax(logits, dim=-1).numpy()
            
            # Inverse transform to original coordinate system
            original_probs = inverse_transform_probs(probs, transform_idx)
            
            all_probs.append(original_probs)
        
        # Average across all 8 transforms
        averaged_probs = np.mean(all_probs, axis=0)
        
        # Normalize by orbit: symmetric positions (considering board state and pattern) get the same probability
        normalized_probs = normalize_probs_by_dynamic_orbit(averaged_probs, board, pattern_idx)
        
        return normalized_probs
    
    def _get_pattern_obs(self, pattern_idx: int) -> np.ndarray:
        """Get 7x7 padded pattern observation."""
        if pattern_idx < 0:
            return np.zeros((7, 7), dtype=np.int8)
        pattern = PATTERNS[pattern_idx]
        padded = np.zeros((7, 7), dtype=np.int8)
        ph, pw = pattern.shape
        oh, ow = (7 - ph) // 2, (7 - pw) // 2
        padded[oh:oh+ph, ow:ow+pw] = pattern
        return padded


# =============================================================================
# GUI Widgets
# =============================================================================

class BoardWidget(QTableWidget):
    """7x7 Bingo board display."""
    
    def __init__(self, parent=None):
        super().__init__(7, 7, parent)
        self.cell_size = 60
        self._setup_ui()
    
    def _setup_ui(self):
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        
        for i in range(7):
            self.setColumnWidth(i, self.cell_size)
            self.setRowHeight(i, self.cell_size)
        
        total_size = 7 * self.cell_size
        self.setFixedSize(total_size + 2, total_size + 2)
        
        self._init_items()
    
    def _init_items(self):
        for i in range(7):
            for j in range(7):
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemIsEnabled)
                item.setBackground(QColor("white"))
                self.setItem(i, j, item)
    
    def update_display(
        self,
        board: np.ndarray,
        probs: np.ndarray = None,
        action_mask: np.ndarray = None,
        swap_probs: np.ndarray = None,
        swap_mask: np.ndarray = None,
        threshold: float = 0.10,
    ):
        """
        Update board display with probabilities.
        
        Args:
            board: Current board state (7x7)
            probs: Position probabilities (50,) - only first 49 used
            action_mask: Valid action mask (50,)
            swap_probs: Probabilities for placing retrieved pattern after swap (50,)
            swap_mask: Valid action mask for swap placement (50,)
            threshold: Minimum probability to display (default 10%)
        """
        for i in range(7):
            for j in range(7):
                item = self.item(i, j)
                pos = i * 7 + j
                
                # Get probabilities
                prob = probs[pos] if probs is not None else 0
                swap_prob = swap_probs[pos] if swap_probs is not None else 0
                
                # Check valid positions
                has_main_prob = action_mask is not None and action_mask[pos] and prob >= threshold
                has_swap_prob = swap_mask is not None and swap_mask[pos] and swap_prob >= threshold
                
                if board[i, j] == 1:
                    # Filled cell - show probability if valid placement center
                    if has_main_prob and has_swap_prob:
                        # Both main and swap on filled cell - purple with dark overlay
                        max_prob = max(prob, swap_prob)
                        intensity = min(max_prob / 0.5, 1.0)
                        lightness = 0.35 - intensity * 0.15  # Dark purple range
                        color = QColor.fromHslF(0.8, 0.6, lightness)
                        item.setBackground(color)
                        item.setText(f"{prob*100:.0f}↔{swap_prob*100:.0f}")
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setForeground(QColor("white"))
                    elif has_main_prob:
                        # Main placement on filled cell - dark blue overlay
                        intensity = min(prob / 0.5, 1.0)
                        lightness = 0.35 - intensity * 0.15  # Dark blue range
                        color = QColor.fromHslF(0.6, 0.6, lightness)
                        item.setBackground(color)
                        item.setText(f"{prob*100:.0f}%")
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setForeground(QColor("white"))
                    elif has_swap_prob:
                        # Swap placement on filled cell - dark green overlay
                        intensity = min(swap_prob / 0.5, 1.0)
                        lightness = 0.35 - intensity * 0.15  # Dark green range
                        color = QColor.fromHslF(0.35, 0.6, lightness)
                        item.setBackground(color)
                        item.setText(f"↔{swap_prob*100:.0f}%")
                        item.setTextAlignment(Qt.AlignCenter)
                        item.setForeground(QColor("white"))
                    else:
                        # Just filled, no valid placement
                        item.setBackground(QColor("#555555"))
                        item.setText("")
                elif has_main_prob and has_swap_prob:
                    # Both main and swap on empty cell - show combined (purple)
                    # 왼쪽: 현재 패턴 확률, 오른쪽: 교환 후 가져온 패턴 확률
                    max_prob = max(prob, swap_prob)
                    intensity = min(max_prob / 0.5, 1.0)
                    lightness = 0.9 - intensity * 0.5
                    color = QColor.fromHslF(0.8, 0.7, lightness)  # Purple
                    item.setBackground(color)
                    item.setText(f"{prob*100:.0f}↔{swap_prob*100:.0f}")
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setForeground(QColor("white") if lightness < 0.5 else QColor("black"))
                elif has_main_prob:
                    # Main placement on empty cell - blue
                    intensity = min(prob / 0.5, 1.0)
                    lightness = 0.9 - intensity * 0.5
                    color = QColor.fromHslF(0.6, 0.8, lightness)  # Blue
                    item.setBackground(color)
                    item.setText(f"{prob*100:.0f}%")
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setForeground(QColor("white") if lightness < 0.5 else QColor("black"))
                elif has_swap_prob:
                    # Swap placement on empty cell - green
                    intensity = min(swap_prob / 0.5, 1.0)
                    lightness = 0.9 - intensity * 0.5
                    color = QColor.fromHslF(0.35, 0.8, lightness)  # Green
                    item.setBackground(color)
                    item.setText(f"↔{swap_prob*100:.0f}%")
                    item.setTextAlignment(Qt.AlignCenter)
                    item.setForeground(QColor("white") if lightness < 0.5 else QColor("black"))
                else:
                    # Empty cell, no probability
                    item.setBackground(QColor("white"))
                    item.setText("")


class PatternListWidget(QListWidget):
    """Pattern selection widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(60, 60))
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)
        self.setSpacing(5)
        self.setMaximumHeight(100)
        
        for idx, pattern in enumerate(PATTERNS):
            icon = self._create_icon(pattern)
            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(Qt.UserRole, idx)
            item.setToolTip(PATTERN_NAMES[idx])
            self.addItem(item)
    
    def _create_icon(self, pattern: np.ndarray) -> QPixmap:
        """Create icon for pattern."""
        # Pad to square
        h, w = pattern.shape
        size = max(h, w)
        padded = np.zeros((size, size), dtype=np.int8)
        oh, ow = (size - h) // 2, (size - w) // 2
        padded[oh:oh+h, ow:ow+w] = pattern
        
        cell_size = 60 // size
        pixmap = QPixmap(size * cell_size, size * cell_size)
        pixmap.fill(Qt.white)
        
        painter = QPainter(pixmap)
        for i in range(size):
            for j in range(size):
                x, y = j * cell_size, i * cell_size
                if padded[i, j] == 1:
                    painter.fillRect(x, y, cell_size, cell_size, QColor("#333333"))
                painter.setPen(QColor("#cccccc"))
                painter.drawRect(x, y, cell_size, cell_size)
        painter.end()
        
        return pixmap


# =============================================================================
# Stored Pattern Widget
# =============================================================================

class StoredPatternWidget(QWidget):
    """Widget to display the stored pattern."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.stored_pattern_idx = -1
        self.cell_size = 12
        self.setFixedSize(7 * self.cell_size + 4, 7 * self.cell_size + 4)
    
    def set_pattern(self, pattern_idx: int):
        """Set the stored pattern index (-1 for none)."""
        self.stored_pattern_idx = pattern_idx
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        
        if self.stored_pattern_idx >= 0:
            # Draw the pattern
            pattern = PATTERNS[self.stored_pattern_idx]
            padded = np.zeros((7, 7), dtype=np.int8)
            ph, pw = pattern.shape
            oh, ow = (7 - ph) // 2, (7 - pw) // 2
            padded[oh:oh+ph, ow:ow+pw] = pattern
            
            for i in range(7):
                for j in range(7):
                    x = j * self.cell_size + 2
                    y = i * self.cell_size + 2
                    if padded[i, j] == 1:
                        painter.fillRect(x, y, self.cell_size - 1, self.cell_size - 1, QColor("#2E7D32"))
                    else:
                        painter.setPen(QColor("#e0e0e0"))
                        painter.drawRect(x, y, self.cell_size - 1, self.cell_size - 1)
        else:
            painter.setPen(QColor("gray"))
            painter.drawText(self.rect(), Qt.AlignCenter, "없음")
        
        painter.end()


# =============================================================================
# Main Window
# =============================================================================

class BingoGUI(QMainWindow):
    """Main application window."""
    
    @staticmethod
    def _get_save_file_path():
        if getattr(sys, 'frozen', False):
            return os.path.join(os.path.dirname(sys.executable), 'bingo_save.json')
        else:
            return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bingo_save.json')
    
    SAVE_FILE = None  # __init__에서 설정
    
    def __init__(self):
        super().__init__()
        
        # 저장 파일 경로 초기화 (PyInstaller 호환)
        BingoGUI.SAVE_FILE = self._get_save_file_path()
        
        self.setWindowTitle("Trickcal Bingo AI Assistant")
        self.resize(750, 600)
        
        # Initialize components
        self.model_manager = ModelManager()
        self.state = BingoState(allow_store=self.model_manager.current_model_name == 'store')
        self.selected_pattern_idx = None
        self.current_probs = None  # Store current probabilities
        
        self._setup_ui()
        
        # 저장된 상태가 있으면 복구 여부 확인
        self._try_load_saved_state()
        
        self._update_display()
    
    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        # Left panel: Board + Pattern list
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        main_layout.addWidget(left_panel, stretch=2)
        
        # Board
        self.board_widget = BoardWidget()
        left_layout.addWidget(self.board_widget, alignment=Qt.AlignCenter)
        
        # Pattern label
        pattern_label = QLabel("패턴 선택 (클릭하여 배치 위치 확인):")
        font = pattern_label.font()
        font.setBold(True)
        pattern_label.setFont(font)
        left_layout.addWidget(pattern_label)
        
        # Pattern list
        self.pattern_list = PatternListWidget()
        self.pattern_list.itemClicked.connect(self._on_pattern_clicked)
        self.pattern_list.itemDoubleClicked.connect(self._on_pattern_double_clicked)
        left_layout.addWidget(self.pattern_list)
        
        left_layout.addStretch()
        
        # Right panel: Controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        main_layout.addWidget(right_panel, stretch=1)
        
        # Model selection group
        model_group = QGroupBox("모델 선택")
        model_layout = QVBoxLayout(model_group)
        
        self.model_label = QLabel()
        font = self.model_label.font()
        font.setBold(True)
        font.setPointSize(font.pointSize() + 1)
        self.model_label.setFont(font)
        model_layout.addWidget(self.model_label)
        
        self.switch_model_btn = QPushButton("모델 전환")
        self.switch_model_btn.clicked.connect(self._on_switch_model)
        model_layout.addWidget(self.switch_model_btn)
        
        right_layout.addWidget(model_group)
        
        # Store group (only visible in store mode)
        self.store_group = QGroupBox("패턴 보관")
        store_layout = QVBoxLayout(self.store_group)
        
        # Stored pattern display
        store_display_layout = QHBoxLayout()
        store_display_layout.addWidget(QLabel("저장된 패턴:"))
        self.stored_pattern_widget = StoredPatternWidget()
        store_display_layout.addWidget(self.stored_pattern_widget)
        store_display_layout.addStretch()
        store_layout.addLayout(store_display_layout)
        
        # Store button with probability
        self.store_btn = QPushButton("보관/교환 (선택 후 확률 표시)")
        self.store_btn.clicked.connect(self._on_store)
        self.store_btn.setEnabled(False)
        store_layout.addWidget(self.store_btn)
        
        right_layout.addWidget(self.store_group)
        
        # Info group
        info_group = QGroupBox("게임 정보")
        info_layout = QVBoxLayout(info_group)
        
        self.turn_label = QLabel("턴: 0")
        info_layout.addWidget(self.turn_label)
        
        self.filled_label = QLabel("채워진 칸: 0 / 49")
        info_layout.addWidget(self.filled_label)
        
        right_layout.addWidget(info_group)
        
        # Action buttons
        action_group = QGroupBox("동작")
        action_layout = QVBoxLayout(action_group)
        
        self.confirm_btn = QPushButton("선택 확정 (클릭한 칸에 배치)")
        self.confirm_btn.clicked.connect(self._on_confirm)
        self.confirm_btn.setEnabled(False)
        action_layout.addWidget(self.confirm_btn)
        
        self.undo_btn = QPushButton("이전으로")
        self.undo_btn.clicked.connect(self._on_undo)
        action_layout.addWidget(self.undo_btn)
        
        self.reset_btn = QPushButton("초기화")
        self.reset_btn.clicked.connect(self._on_reset)
        action_layout.addWidget(self.reset_btn)
        
        right_layout.addWidget(action_group)
        
        # Probability threshold info
        info_label = QLabel(
            "■ 확률 10% 이상인 위치만 표시됩니다.\n"
            "■ 색이 진할수록 높은 확률입니다.\n\n"
            "색상 안내:\n"
            "  • 파랑: 현재 패턴 배치 확률\n"
            "  • 초록(↔): 교환 후 가져온 패턴 배치 확률\n"
            "  • 보라(A↔B): 양쪽 다 가능\n"
            "    (A: 현재 패턴, B: 교환 후 패턴)"
        )
        info_label.setStyleSheet("color: gray; font-size: 11px;")
        info_label.setWordWrap(True)
        right_layout.addWidget(info_label)
        
        right_layout.addStretch()
        
        # Connect board click
        self.board_widget.cellClicked.connect(self._on_cell_clicked)
        self.board_widget.cellDoubleClicked.connect(self._on_cell_double_clicked)
        
        self._update_model_label()
        self._update_store_visibility()
    
    def _update_model_label(self):
        """Update model label text."""
        if self.model_manager.current_model_name == 'store':
            self.model_label.setText("현재: 교환 허용 모드")
            self.model_label.setStyleSheet("color: #2196F3;")
        else:
            self.model_label.setText("현재: 교환 비허용 모드")
            self.model_label.setStyleSheet("color: #FF9800;")
    
    def _update_store_visibility(self):
        """Show/hide store group based on model mode."""
        is_store_mode = self.model_manager.current_model_name == 'store'
        self.store_group.setVisible(is_store_mode)
    
    def _on_switch_model(self):
        """Switch between models."""
        if self.model_manager.current_model_name == 'store':
            new_model = 'no_store'
        else:
            new_model = 'store'
        
        if new_model not in self.model_manager.models:
            QMessageBox.warning(self, "경고", f"'{new_model}' 모델을 찾을 수 없습니다.")
            return
        
        # Confirm if game in progress
        if self.state.current_step > 0:
            reply = QMessageBox.question(
                self, "확인",
                "게임이 진행 중입니다. 모델을 전환하면 게임이 초기화됩니다.\n계속하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return
        
        self.model_manager.switch_model(new_model)
        self.state = BingoState(allow_store=(new_model == 'store'))
        self.selected_pattern_idx = None
        self.current_probs = None
        self.pattern_list.clearSelection()
        self._update_model_label()
        self._update_store_visibility()
        self._update_display()
    
    def _on_store(self):
        """Handle store/swap button click."""
        if self.selected_pattern_idx is None:
            return
        
        # Check if can store
        if not self.state.allow_store:
            return
        
        same_pattern = self.state.stored_pattern_idx == self.selected_pattern_idx
        has_stored = self.state.stored_pattern_idx >= 0
        cant_swap = has_stored and same_pattern
        
        if cant_swap:
            QMessageBox.warning(self, "경고", "같은 패턴은 교환할 수 없습니다.")
            return
        
        # Save state for undo
        self.state.save_state()
        
        # Perform store/swap
        if has_stored:
            # Swap: exchange current pattern with stored pattern
            old_stored = self.state.swap_pattern(self.selected_pattern_idx)
            
            # 교환 후 가져온 패턴을 자동 선택
            self.selected_pattern_idx = old_stored
            self.current_probs = None
            
            # 패턴 리스트에서 해당 패턴 선택
            for i in range(self.pattern_list.count()):
                item = self.pattern_list.item(i)
                if item.data(Qt.UserRole) == old_stored:
                    self.pattern_list.setCurrentItem(item)
                    break
            
            self.confirm_btn.setEnabled(False)
            self.store_btn.setEnabled(False)
        else:
            # Store: save current pattern
            self.state.store_pattern(self.selected_pattern_idx)
            
            # Clear selection
            self.selected_pattern_idx = None
            self.current_probs = None
            self.pattern_list.clearSelection()
            self.confirm_btn.setEnabled(False)
            self.store_btn.setEnabled(False)
        
        self._update_display()
    
    def _on_pattern_clicked(self, item):
        """Handle pattern selection. Second click on same pattern triggers auto-placement."""
        pattern_idx = item.data(Qt.UserRole)
        
        # 먼저 패턴 선택 상태를 확인 (이전 상태 저장)
        was_selected = self.selected_pattern_idx == pattern_idx
        had_probs = self.current_probs is not None
        
        # 1. 패턴 선택 상태 변경 (항상 먼저)
        self.selected_pattern_idx = pattern_idx
        self.confirm_btn.setEnabled(False)
        
        # 2. 확률 계산 및 디스플레이 업데이트
        self._update_display()
        
        # 3. 이미 선택되어 있었고 확률이 표시되어 있었다면 auto-placement 실행
        if was_selected and had_probs:
            self._auto_place_pattern(pattern_idx)
    
    def _on_pattern_double_clicked(self, item):
        """Handle pattern double-click - same as second click."""
        pattern_idx = item.data(Qt.UserRole)
        
        # 1. 패턴 선택 상태 변경 (먼저)
        self.selected_pattern_idx = pattern_idx
        self.confirm_btn.setEnabled(False)
        
        # 2. 확률 계산 및 디스플레이 업데이트
        self._update_display()
        
        # 3. auto-placement 실행
        self._auto_place_pattern(pattern_idx)
    
    def _auto_place_pattern(self, pattern_idx: int):
        """Auto-place pattern based on Markov converged probabilities.
        
        Compares current pattern placement vs stored pattern placement
        using the converged probabilities from _update_display().
        """
        # 확률이 계산되지 않았다면 리턴 (호출 순서 오류 방지)
        if self.current_probs is None:
            return
        
        probs = self.current_probs
        swap_probs = self.current_swap_probs
        
        # Find best action from current pattern (positions 0-48 only)
        current_mask = self.state.get_action_mask(pattern_idx)
        current_masked = probs[:49].copy()
        current_masked[~current_mask[:49]] = -1
        best_current_pos = int(np.argmax(current_masked))
        best_current_prob = current_masked[best_current_pos] if current_mask[best_current_pos] else -1
        
        # Find best action from swap pattern (if available)
        best_swap_pos = -1
        best_swap_prob = -1
        if swap_probs is not None and self.state.stored_pattern_idx >= 0:
            swap_mask = self.state.get_action_mask(self.state.stored_pattern_idx)
            swap_masked = swap_probs[:49].copy()
            swap_masked[~swap_mask[:49]] = -1
            best_swap_pos = int(np.argmax(swap_masked))
            best_swap_prob = swap_masked[best_swap_pos] if swap_mask[best_swap_pos] else -1
        
        # Save state for undo
        self.state.save_state()
        
        # Decide action
        if self.state.stored_pattern_idx < 0:
            # 첫 턴: 무조건 보관 (수학적으로 최적)
            self.state.store_pattern(pattern_idx)
        elif best_swap_prob > best_current_prob:
            # Swap and place stored pattern
            old_stored = self.state.swap_pattern(pattern_idx)
            row, col = best_swap_pos // 7, best_swap_pos % 7
            self.state.apply_pattern(old_stored, row, col)
        elif best_current_prob >= 0:
            # Place current pattern
            row, col = best_current_pos // 7, best_current_pos % 7
            self.state.apply_pattern(pattern_idx, row, col)
        
        # Clear selection
        self.selected_pattern_idx = None
        self.current_probs = None
        self.current_swap_probs = None
        self.pattern_list.clearSelection()
        self.confirm_btn.setEnabled(False)
        
        self._update_display()
        self._check_game_complete()
    
    def _on_cell_clicked(self, row, col):
        """Handle board cell click."""
        if self.selected_pattern_idx is None:
            return
        
        pos = row * 7 + col
        action_mask = self.state.get_action_mask(self.selected_pattern_idx)
        
        if action_mask[pos]:
            # Valid position selected
            self._selected_position = (row, col)
            self.confirm_btn.setEnabled(True)
            self.confirm_btn.setText(f"확정: ({row}, {col})에 배치")
        else:
            self._selected_position = None
            self.confirm_btn.setEnabled(False)
            self.confirm_btn.setText("선택 확정 (클릭한 칸에 배치)")
    
    def _on_cell_double_clicked(self, row, col):
        """Handle board cell double-click - directly place pattern."""
        if self.selected_pattern_idx is None:
            return
        
        pos = row * 7 + col
        action_mask = self.state.get_action_mask(self.selected_pattern_idx)
        
        if action_mask[pos]:
            # Save state for undo
            self.state.save_state()
            
            # Apply pattern
            self.state.apply_pattern(self.selected_pattern_idx, row, col)
            
            # Clear selection
            self.selected_pattern_idx = None
            self.pattern_list.clearSelection()
            self.confirm_btn.setEnabled(False)
            self.confirm_btn.setText("선택 확정 (클릭한 칸에 배치)")
            
            self._update_display()
            self._check_game_complete()
    
    def _on_confirm(self):
        """Confirm placement at selected position."""
        if self.selected_pattern_idx is None or not hasattr(self, '_selected_position'):
            return
        
        row, col = self._selected_position
        
        # Save state for undo
        self.state.save_state()
        
        # Apply pattern
        self.state.apply_pattern(self.selected_pattern_idx, row, col)
        
        # Clear selection
        self.selected_pattern_idx = None
        self._selected_position = None
        self.pattern_list.clearSelection()
        self.confirm_btn.setEnabled(False)
        self.confirm_btn.setText("선택 확정 (클릭한 칸에 배치)")
        
        self._update_display()
        self._check_game_complete()
    
    def _check_game_complete(self):
        """Check if game is complete and show message."""
        if np.all(self.state.board == 1):
            # 게임 완료 시 저장 파일 삭제
            self._delete_save_file()
            QMessageBox.information(
                self, "완료!",
                f"빙고 완성! 총 {self.state.current_step}턴 소요되었습니다."
            )
    
    def _on_undo(self):
        """Undo last action."""
        if self.state.undo():
            self.selected_pattern_idx = None
            self.pattern_list.clearSelection()
            self.confirm_btn.setEnabled(False)
            self._update_display()
    
    def _on_reset(self):
        """Reset game."""
        reply = QMessageBox.question(
            self, "초기화",
            "정말 초기화하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.state.reset()
            self.selected_pattern_idx = None
            self.pattern_list.clearSelection()
            self.confirm_btn.setEnabled(False)
            # 초기화 시 저장 파일 삭제
            self._delete_save_file()
            self._update_display()
    
    def _update_display(self):
        """Update board display with current state and probabilities.
        
        Uses Markov chain convergence to compute final placement probabilities
        considering both current and stored patterns.
        """
        probs = None
        action_mask = None
        swap_probs = None
        swap_mask = None
        
        if self.selected_pattern_idx is not None:
            # Get action mask for GUI display
            action_mask = self.state.get_action_mask(self.selected_pattern_idx)
            
            if self.state.stored_pattern_idx >= 0:
                # === Markov Chain Convergence ===
                # S1: current pattern A, stored pattern B
                # S2: current pattern B, stored pattern A
                # Compute stationary distribution for fair comparison
                
                # State S1: Current pattern
                model_mask_S1 = self.state.get_model_action_mask(self.selected_pattern_idx)
                probs_S1 = self.model_manager.get_d4_averaged_probs(
                    self.state.board,
                    self.selected_pattern_idx,
                    self.state.stored_pattern_idx,
                    action_mask=model_mask_S1,
                )
                swap_S1 = probs_S1[49]  # P(swap | S1)
                
                # State S2: Stored pattern (after swap)
                swap_mask = self.state.get_action_mask(self.state.stored_pattern_idx)
                model_mask_S2 = swap_mask.copy()
                model_mask_S2[49] = False  # Can't swap back immediately in model
                
                probs_S2 = self.model_manager.get_d4_averaged_probs(
                    self.state.board,
                    self.state.stored_pattern_idx,
                    self.selected_pattern_idx,
                    action_mask=model_mask_S2,
                )
                swap_S2 = probs_S2[49]  # P(swap | S2) - should be 0 due to mask
                
                # Markov convergence
                # r = P(S1→S2) × P(S2→S1) = swap_S1 × swap_S2
                r = swap_S1 * swap_S2
                
                # Avoid division by zero (r is almost always < 1)
                if r >= 0.999:
                    r = 0.999
                normalizer = 1.0 / (1.0 - r)
                
                # Final placement probabilities
                # P_A(pos) = P(place A at pos) = (1 - swap_S1) × probs_S1[pos] × normalizer
                # P_B(pos) = P(place B at pos) = swap_S1 × (1 - swap_S2) × probs_S2[pos] × normalizer
                
                probs = probs_S1.copy()
                probs[:49] = (1.0 - swap_S1) * probs_S1[:49] * normalizer
                probs[49] = 0  # Swap probability is now embedded in placement probs
                
                swap_probs = np.zeros(50)
                swap_probs[:49] = swap_S1 * (1.0 - swap_S2) * probs_S2[:49] * normalizer
                
                self.current_probs = probs
                self.current_swap_probs = swap_probs
            else:
                # No stored pattern - just use current pattern probs
                model_mask = self.state.get_model_action_mask(self.selected_pattern_idx)
                probs = self.model_manager.get_d4_averaged_probs(
                    self.state.board,
                    self.selected_pattern_idx,
                    self.state.stored_pattern_idx,
                    action_mask=model_mask,
                )
                self.current_probs = probs
                self.current_swap_probs = None
        else:
            self.current_probs = None
            self.current_swap_probs = None
        
        self.board_widget.update_display(
            self.state.board,
            probs=probs,
            action_mask=action_mask,
            swap_probs=swap_probs,
            swap_mask=swap_mask,
            threshold=0.10,
        )
        
        self.turn_label.setText(f"턴: {self.state.current_step}")
        filled = np.sum(self.state.board)
        self.filled_label.setText(f"채워진 칸: {filled} / 49")
        self.undo_btn.setEnabled(len(self.state.history) > 0)
        
        # Update store UI
        if self.state.allow_store:
            self.stored_pattern_widget.set_pattern(self.state.stored_pattern_idx)
            
            # Update store button (no probability display - it's embedded in placement probs)
            if self.selected_pattern_idx is not None and action_mask is not None:
                can_store = action_mask[49]
                if can_store:
                    if self.state.stored_pattern_idx >= 0:
                        self.store_btn.setText("교환")
                    else:
                        self.store_btn.setText("보관")
                    self.store_btn.setEnabled(True)
                    self.store_btn.setStyleSheet("")
                else:
                    self.store_btn.setText("보관/교환 (불가)")
                    self.store_btn.setEnabled(False)
                    self.store_btn.setStyleSheet("")
            else:
                self.store_btn.setText("보관/교환 (패턴 선택 필요)")
                self.store_btn.setEnabled(False)
                self.store_btn.setStyleSheet("")
    
    def _try_load_saved_state(self):
        """저장된 상태가 있으면 복구 여부를 물어보고 복구합니다."""
        if not os.path.exists(self.SAVE_FILE):
            return
        
        try:
            with open(self.SAVE_FILE, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            state_data = save_data.get('state', {})
            current_step = state_data.get('current_step', 0)
            stored_pattern_idx = state_data.get('stored_pattern_idx', -1)
            filled = sum(sum(row) for row in state_data.get('board', []))
            
            # 게임 진행 여부 확인: 턴이 진행됐거나, 패턴이 저장되어 있거나, 칸이 채워져 있으면 진행 중
            has_progress = current_step > 0 or stored_pattern_idx >= 0 or filled > 0
            
            if not has_progress:
                # 진행된 내용이 전혀 없으면 삭제
                os.remove(self.SAVE_FILE)
                return
            
            # 저장된 패턴 정보도 표시
            stored_info = ""
            if stored_pattern_idx >= 0:
                stored_info = f", 보관된 패턴: {PATTERN_NAMES[stored_pattern_idx]}"
            
            reply = QMessageBox.question(
                self, "저장된 게임 발견",
                f"이전에 저장된 게임이 있습니다.\n"
                f"(턴: {current_step}, 채워진 칸: {filled}/49{stored_info})\n\n"
                f"이어서 하시겠습니까?",
                QMessageBox.Yes | QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._load_from_save_data(save_data)
            else:
                # 새 게임 시작 - 저장 파일 삭제
                os.remove(self.SAVE_FILE)
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            # 손상된 저장 파일 - 삭제
            print(f"저장 파일 로드 실패: {e}")
            if os.path.exists(self.SAVE_FILE):
                os.remove(self.SAVE_FILE)
    
    def _load_from_save_data(self, save_data: dict):
        """저장 데이터로부터 상태를 복구합니다."""
        # 모델 전환 (필요시)
        saved_model = save_data.get('model_name', 'store')
        if saved_model != self.model_manager.current_model_name:
            if saved_model in self.model_manager.models:
                self.model_manager.switch_model(saved_model)
                self._update_model_label()
                self._update_store_visibility()
        
        # 상태 복구
        self.state = BingoState.from_dict(save_data['state'])
        self.selected_pattern_idx = None
        self.current_probs = None
        self.pattern_list.clearSelection()
        self.confirm_btn.setEnabled(False)
    
    def _save_game_state(self):
        """현재 게임 상태를 파일에 저장합니다."""
        save_data = {
            'model_name': self.model_manager.current_model_name,
            'state': self.state.to_dict(),
        }
        
        try:
            with open(self.SAVE_FILE, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"게임 저장 실패: {e}")
    
    def _delete_save_file(self):
        """저장 파일을 삭제합니다."""
        if os.path.exists(self.SAVE_FILE):
            try:
                os.remove(self.SAVE_FILE)
            except IOError:
                pass
    
    def closeEvent(self, event):
        """창이 닫힐 때 게임 상태를 저장합니다."""
        # 게임이 완료되었으면 저장 파일 삭제
        if np.all(self.state.board == 1):
            self._delete_save_file()
        else:
            # 진행 중인 게임이면 저장
            self._save_game_state()
        
        event.accept()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    window = BingoGUI()
    window.show()
    sys.exit(app.exec())
