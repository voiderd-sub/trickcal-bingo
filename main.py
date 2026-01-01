"""
Bingo AI Assistant GUI.

Uses trained RL models to suggest optimal pattern placements.
Supports both 1-pattern and 2-pattern modes.
Displays D4 symmetry-averaged probabilities with center-only visualization.

REFACTORED: Uses num_patterns (1 or 2) instead of allow_store.
For num_patterns=2, patterns are stored as sorted pairs for canonical state.
"""

import sys
import os
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QAbstractItemView,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QColor, QPainter, QPixmap

from bingo_policy import BingoCNNExtractor

# Pattern definitions
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



def compute_dynamic_orbits(board: np.ndarray, pattern_idx: int = None) -> list:
    """
    Compute D4 symmetry orbits considering the current board state and pattern.
    """
    def board_hash(board_arr):
        return tuple(board_arr.flatten().tolist())
    
    def get_canonical_hash(board_arr):
        hashes = []
        for t in range(8):
            transformed = transform_board(board_arr, t)
            hashes.append(board_hash(transformed))
        return min(hashes)
    
    if pattern_idx is None:
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
    
    pattern = PATTERNS[pattern_idx]
    ph, pw = pattern.shape
    offset_h, offset_w = ph // 2, pw // 2
    
    position_to_canonical = {}
    
    for pos in range(49):
        row, col = pos // 7, pos % 7
        result_board = board.copy()
        for pr in range(ph):
            for pc in range(pw):
                if pattern[pr, pc] == 1:
                    br = row - offset_h + pr
                    bc = col - offset_w + pc
                    if 0 <= br < 7 and 0 <= bc < 7:
                        result_board[br, bc] = 1
        position_to_canonical[pos] = get_canonical_hash(result_board)
    
    canonical_to_positions = {}
    for pos, canonical in position_to_canonical.items():
        if canonical not in canonical_to_positions:
            canonical_to_positions[canonical] = set()
        canonical_to_positions[canonical].add(pos)
    
    return list(canonical_to_positions.values())


def normalize_probs_by_dynamic_orbit(probs: np.ndarray, board: np.ndarray, pattern_indices: list, num_patterns: int) -> np.ndarray:
    """
    Normalize probabilities so that symmetric positions have the same probability.
    
    Each slot uses its own pattern's orbit - positions that produce D4-equivalent
    result boards when that pattern is placed.
    """
    normalized = probs.copy()
    
    for slot in range(num_patterns):
        pattern_idx = pattern_indices[slot] if slot < len(pattern_indices) else -1
        if pattern_idx < 0:
            continue
            
        orbits = compute_dynamic_orbits(board, pattern_idx)
        start_idx = slot * 49
        
        for orbit in orbits:
            orbit_sum = sum(probs[start_idx + pos] for pos in orbit)
            for pos in orbit:
                normalized[start_idx + pos] = orbit_sum
    
    return normalized


# =============================================================================
# Policy Network (copy from train.py for CPU loading)
# =============================================================================

class MaskablePPOPolicy(nn.Module):
    """Actor-Critic policy for MaskablePPO."""
    
    def __init__(
        self,
        observation_space,
        action_dim: int = 49,
        features_dim: int = 256,
        hidden_channels: int = 64,
        num_res_blocks: int = 3,
        kernel_size: int = 3,
        num_patterns: int = 1,
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
            num_patterns=num_patterns,
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
    
    def __init__(self, num_patterns: int = 1):
        self.board_size = 7
        self.num_patterns = num_patterns
        self.n_actions = 49 * num_patterns
        self.reset()
    
    def reset(self):
        self.board = np.zeros((7, 7), dtype=np.int8)
        # pattern_indices: list of pattern type indices (sorted if num_patterns=2)
        self.pattern_indices = [-1] * self.num_patterns
        self.current_step = 0
        self.history = []
    
    def get_pattern_obs(self, pattern_idx: int) -> np.ndarray:
        """Get 7x7 padded pattern for observation."""
        if pattern_idx < 0:
            return np.zeros((7, 7), dtype=np.int8)
        pattern = PATTERNS[pattern_idx]
        padded = np.zeros((7, 7), dtype=np.int8)
        ph, pw = pattern.shape
        oh, ow = (7 - ph) // 2, (7 - pw) // 2
        padded[oh:oh+ph, ow:ow+pw] = pattern
        return padded
    
    def get_action_mask(self) -> np.ndarray:
        """Compute valid action mask for all pattern slots."""
        mask = np.zeros(self.n_actions, dtype=bool)
        
        for slot in range(self.num_patterns):
            pattern_idx = self.pattern_indices[slot]
            if pattern_idx < 0:
                continue
                
            pattern = PATTERNS[pattern_idx]
            ph, pw = pattern.shape
            offset_h, offset_w = ph // 2, pw // 2
            
            for r in range(7):
                for c in range(7):
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
                        mask[slot * 49 + r * 7 + c] = True
        
        return mask
    
    def get_slot_action_mask(self, slot: int) -> np.ndarray:
        """Get action mask for a specific slot (49 positions)."""
        full_mask = self.get_action_mask()
        return full_mask[slot * 49:(slot + 1) * 49]
    
    def apply_action(self, action: int):
        """Apply action (place pattern at position)."""
        slot = action // 49
        pos = action % 49
        pattern_idx = self.pattern_indices[slot]
        row, col = pos // 7, pos % 7
        
        self._apply_pattern(pattern_idx, row, col)
        
        # After using a pattern: move remaining to stored (slot 1), clear current (slot 0)
        if self.num_patterns == 2:
            remaining = self.pattern_indices[1 - slot]  # The other slot
            self.pattern_indices = [-1, remaining]  # current empty, stored has remaining
        else:
            self.pattern_indices[slot] = -1
    
    def _apply_pattern(self, pattern_idx: int, center_row: int, center_col: int):
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
    
    def set_patterns(self, pattern_indices: list):
        """Set pattern indices (will be sorted for canonical state)."""
        self.pattern_indices = list(pattern_indices)
        self._sort_patterns()
    
    def _sort_patterns(self):
        """Sort pattern indices for canonical state - not used in GUI mode."""
        # GUI mode doesn't sort - slot 0 = current, slot 1 = stored
        pass
    
    def save_state(self):
        """Save current state to history."""
        self.history.append({
            'board': self.board.copy(),
            'pattern_indices': self.pattern_indices.copy(),
            'current_step': self.current_step,
        })
    
    def undo(self) -> bool:
        """Restore previous state. Returns True if successful."""
        if not self.history:
            return False
        
        state = self.history.pop()
        self.board = state['board']
        self.pattern_indices = state['pattern_indices']
        self.current_step = state['current_step']
        return True
    
    def to_dict(self) -> dict:
        """Serialize state to dictionary for saving."""
        return {
            'board': self.board.tolist(),
            'pattern_indices': self.pattern_indices,
            'current_step': self.current_step,
            'num_patterns': self.num_patterns,
            'history': [
                {
                    'board': h['board'].tolist(),
                    'pattern_indices': h['pattern_indices'],
                    'current_step': h['current_step'],
                }
                for h in self.history
            ]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'BingoState':
        """Deserialize state from dictionary."""
        num_patterns = data.get('num_patterns', 1)
        state = cls(num_patterns=num_patterns)
        state.board = np.array(data.get('board', np.zeros((7, 7))), dtype=np.int8)
        state.pattern_indices = data.get('pattern_indices', [-1] * num_patterns)
        state.current_step = data.get('current_step', 0)
        state.history = [
            {
                'board': np.array(h.get('board', np.zeros((7, 7))), dtype=np.int8),
                'pattern_indices': h.get('pattern_indices', [-1] * num_patterns),
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
        self.models = {}  # {num_patterns: policy}
        self.current_num_patterns = None
        self.policy = None
        self._load_models()
    
    def _create_obs_space(self, num_patterns: int):
        """Create observation space for policy."""
        from gymnasium import spaces
        return spaces.Dict({
            "board": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern_0": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
            "pattern_1": spaces.Box(low=0, high=1, shape=(7, 7), dtype=np.int8),
        })
    
    def _create_policy(self, num_patterns: int) -> MaskablePPOPolicy:
        """Create policy network with default config."""
        action_dim = 49 * num_patterns
        obs_space = self._create_obs_space(num_patterns)
        return MaskablePPOPolicy(
            obs_space,
            action_dim=action_dim,
            features_dim=256,
            hidden_channels=64,
            num_res_blocks=3,
            kernel_size=3,
            num_patterns=num_patterns,
            pi_layers=[256, 128],
            vf_layers=[256, 128],
        )
    
    def _load_models(self):
        """Load models for different num_patterns settings."""
        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        model_paths = {
            1: os.path.join(base_path, 'model', 'best_model_1pattern.pt'),
            2: os.path.join(base_path, 'model', 'best_model_2patterns.pt'),
        }
        
        for num_patterns, path in model_paths.items():
            if os.path.exists(path):
                try:
                    policy = self._create_policy(num_patterns)
                    policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                    policy.eval()
                    self.models[num_patterns] = policy
                except Exception as e:
                    print(f"Failed to load model for {num_patterns} patterns: {e}")
        
        # Set default model (prefer 2 patterns if available)
        if 2 in self.models:
            self.switch_model(2)
        elif 1 in self.models:
            self.switch_model(1)
    
    def switch_model(self, num_patterns: int):
        """Switch to specified model."""
        if num_patterns in self.models:
            self.current_num_patterns = num_patterns
            self.policy = self.models[num_patterns]
            return True
        return False
    
    def get_probs_with_orbit_normalization(
        self,
        board: np.ndarray,
        pattern_indices: list,
        action_mask: np.ndarray = None,
    ) -> np.ndarray:
        """
        Compute action probabilities with D4 orbit normalization.
        
        1. Single model inference
        2. For each pattern slot, compute orbits based on result board D4 equivalence
        3. Positions in the same orbit (producing D4-equivalent result boards) get summed probability
        
        Args:
            board: Current board state (7x7)
            pattern_indices: List of pattern indices [current, stored]
            action_mask: Valid action mask (n_actions,)
        
        Returns:
            (n_actions,) array of orbit-normalized probabilities
        """
        if self.policy is None:
            return np.zeros(self.current_num_patterns * 49)
        
        num_patterns = self.current_num_patterns
        n_actions = 49 * num_patterns
        
        if action_mask is None:
            action_mask = np.ones(n_actions, dtype=bool)
        
        # Get padded pattern observations
        pattern_0_obs = self._get_pattern_obs(pattern_indices[0] if len(pattern_indices) > 0 else -1)
        pattern_1_obs = self._get_pattern_obs(pattern_indices[1] if len(pattern_indices) > 1 else -1)
        
        # Create observation tensors
        obs = {
            'board': torch.from_numpy(board).float().unsqueeze(0),
            'pattern_0': torch.from_numpy(pattern_0_obs).float().unsqueeze(0),
            'pattern_1': torch.from_numpy(pattern_1_obs).float().unsqueeze(0),
        }
        
        # Single model inference
        with torch.no_grad():
            logits, _ = self.policy(obs)
        
        # Apply action mask
        logits = logits.squeeze(0).clone()
        mask_tensor = torch.from_numpy(action_mask)
        logits[~mask_tensor] = float('-inf')
        
        # Compute probabilities
        probs = F.softmax(logits, dim=-1).numpy()
        
        # Normalize by orbit for each pattern slot
        # Positions producing D4-equivalent result boards share their probability sum
        normalized_probs = normalize_probs_by_dynamic_orbit(
            probs, board, pattern_indices, num_patterns
        )
        
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
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        
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
                item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                item.setBackground(QColor("white"))
                self.setItem(i, j, item)
    
    def update_display(
        self,
        board: np.ndarray,
        probs: np.ndarray = None,
        action_mask: np.ndarray = None,
        num_patterns: int = 1,
        threshold: float = 0.10,
    ):
        """
        Update board display with probabilities.
        
        For num_patterns=2:
        - probs[0:49] = pattern 0 probabilities (blue)
        - probs[49:98] = pattern 1 probabilities (green)
        """
        # Calculate rankings
        rank_map = {}
        if probs is not None:
            all_valid_probs = []
            for i in range(49):
                prob_0 = probs[i] if len(probs) > i else 0
                prob_1 = probs[49+i] if num_patterns >= 2 and len(probs) > 49+i else 0
                
                mask_0 = action_mask is not None and len(action_mask) > i and action_mask[i]
                mask_1 = action_mask is not None and num_patterns >= 2 and len(action_mask) > 49+i and action_mask[49+i]
                
                if mask_0 and prob_0 >= threshold:
                    all_valid_probs.append(round(prob_0, 4))
                if mask_1 and prob_1 >= threshold:
                    all_valid_probs.append(round(prob_1, 4))
            
            unique_probs = sorted(list(set(all_valid_probs)), reverse=True)
            rank_map = {p: i+1 for i, p in enumerate(unique_probs)}

        for i in range(7):
            for j in range(7):
                item = self.item(i, j)
                pos = i * 7 + j
                
                # Get probabilities for each pattern slot
                prob_0 = probs[pos] if probs is not None and len(probs) > pos else 0
                prob_1 = probs[49 + pos] if probs is not None and num_patterns >= 2 and len(probs) > 49 + pos else 0
                
                mask_0 = action_mask is not None and len(action_mask) > pos and action_mask[pos]
                mask_1 = action_mask is not None and num_patterns >= 2 and len(action_mask) > 49 + pos and action_mask[49 + pos]
                
                has_prob_0 = mask_0 and prob_0 >= threshold
                has_prob_1 = mask_1 and prob_1 >= threshold
                
                rank0 = rank_map.get(round(prob_0, 4)) if has_prob_0 else None
                rank1 = rank_map.get(round(prob_1, 4)) if has_prob_1 else None
                
                text = ""
                
                if board[i, j] == 1:
                    # Filled cell
                    if has_prob_0 or has_prob_1:
                        max_prob = max(prob_0 if has_prob_0 else 0, prob_1 if has_prob_1 else 0)
                        intensity = min(max_prob / 0.5, 1.0)
                        lightness = 0.9 - intensity * 0.5
                        if has_prob_0 and has_prob_1:
                            color = QColor.fromHslF(0.8, 0.7, lightness)  # Purple
                            text = f"[{rank0}]{prob_0*100:.0f}%\n[{rank1}]{prob_1*100:.0f}%"
                        elif has_prob_0:
                            color = QColor.fromHslF(0.6, 0.6, lightness)  # Blue
                            text = f"[{rank0}] {prob_0*100:.0f}%"
                        else:
                            color = QColor.fromHslF(0.35, 0.6, lightness)  # Green
                            text = f"[{rank1}] {prob_1*100:.0f}%"
                        item.setBackground(color)
                        item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                        item.setForeground(QColor("white"))
                        item.setText(text)
                    else:
                        item.setBackground(QColor("#555555"))
                        item.setText("")
                elif has_prob_0 and has_prob_1:
                    # Both patterns on empty cell
                    max_prob = max(prob_0, prob_1)
                    intensity = min(max_prob / 0.5, 1.0)
                    lightness = 0.9 - intensity * 0.5
                    color = QColor.fromHslF(0.8, 0.7, lightness)  # Purple
                    item.setBackground(color)
                    item.setText(f"[{rank0}]{prob_0*100:.0f}%\n[{rank1}]{prob_1*100:.0f}%")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setForeground(QColor("white") if lightness < 0.5 else QColor("black"))
                elif has_prob_0:
                    # Pattern 0 on empty cell - blue
                    intensity = min(prob_0 / 0.5, 1.0)
                    lightness = 0.9 - intensity * 0.5
                    color = QColor.fromHslF(0.6, 0.8, lightness)
                    item.setBackground(color)
                    item.setText(f"[{rank0}] {prob_0*100:.0f}%")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setForeground(QColor("white") if lightness < 0.5 else QColor("black"))
                elif has_prob_1:
                    # Pattern 1 on empty cell - green
                    intensity = min(prob_1 / 0.5, 1.0)
                    lightness = 0.9 - intensity * 0.5
                    color = QColor.fromHslF(0.35, 0.8, lightness)
                    item.setBackground(color)
                    item.setText(f"[{rank1}] {prob_1*100:.0f}%")
                    item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                    item.setForeground(QColor("white") if lightness < 0.5 else QColor("black"))
                else:
                    item.setBackground(QColor("white"))
                    item.setText("")


class PatternListWidget(QListWidget):
    """Pattern selection widget."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setIconSize(QSize(60, 60))
        self.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.setMovement(QListWidget.Movement.Static)
        self.setSpacing(5)
        self.setMaximumHeight(100)
        
        for idx, pattern in enumerate(PATTERNS):
            icon = self._create_icon(pattern)
            item = QListWidgetItem()
            item.setIcon(icon)
            item.setData(Qt.ItemDataRole.UserRole, idx)
            item.setToolTip(PATTERN_NAMES[idx])
            self.addItem(item)
    
    def _create_icon(self, pattern: np.ndarray) -> QPixmap:
        """Create icon for pattern."""
        h, w = pattern.shape
        size = max(h, w)
        padded = np.zeros((size, size), dtype=np.int8)
        oh, ow = (size - h) // 2, (size - w) // 2
        padded[oh:oh+h, ow:ow+w] = pattern
        
        cell_size = 60 // size
        pixmap = QPixmap(size * cell_size, size * cell_size)
        pixmap.fill(Qt.GlobalColor.white)
        
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


class PatternDisplayWidget(QWidget):
    """Widget to display current patterns."""
    
    def __init__(self, slot: int, parent=None):
        super().__init__(parent)
        self.slot = slot
        self.pattern_idx = -1
        self.cell_size = 12
        self.setFixedSize(7 * self.cell_size + 4, 7 * self.cell_size + 4)
    
    def set_pattern(self, pattern_idx: int):
        self.pattern_idx = pattern_idx
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        
        if self.pattern_idx >= 0:
            pattern = PATTERNS[self.pattern_idx]
            padded = np.zeros((7, 7), dtype=np.int8)
            ph, pw = pattern.shape
            oh, ow = (7 - ph) // 2, (7 - pw) // 2
            padded[oh:oh+ph, ow:ow+pw] = pattern
            
            # Colors: slot 0 = blue, slot 1 = green
            fill_color = QColor("#1976D2") if self.slot == 0 else QColor("#388E3C")
            
            for i in range(7):
                for j in range(7):
                    x = j * self.cell_size + 2
                    y = i * self.cell_size + 2
                    if padded[i, j] == 1:
                        painter.fillRect(x, y, self.cell_size - 1, self.cell_size - 1, fill_color)
                    else:
                        painter.setPen(QColor("#e0e0e0"))
                        painter.drawRect(x, y, self.cell_size - 1, self.cell_size - 1)
        else:
            painter.setPen(QColor("gray"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "없음")
        
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
    
    SAVE_FILE = None
    
    def __init__(self):
        super().__init__()
        
        BingoGUI.SAVE_FILE = self._get_save_file_path()
        
        self.setWindowTitle("Trickcal Bingo AI Assistant")
        self.resize(750, 600)
        
        # Initialize components
        self.model_manager = ModelManager()
        num_patterns = self.model_manager.current_num_patterns or 1
        self.state = BingoState(num_patterns=num_patterns)
        self.current_probs = None
        
        self._setup_ui()
        self._try_load_saved_state()
        self._update_display()
        
        # Right-click anywhere for auto-place
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_right_click)
    
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
        left_layout.addWidget(self.board_widget, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Pattern label
        pattern_label = QLabel("패턴 선택 (클릭하여 현재 패턴 설정):")
        font = pattern_label.font()
        font.setBold(True)
        pattern_label.setFont(font)
        left_layout.addWidget(pattern_label)
        
        # Pattern list
        self.pattern_list = PatternListWidget()
        self.pattern_list.itemClicked.connect(self._on_pattern_clicked)
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
        
        # Current patterns group
        patterns_group = QGroupBox("현재 패턴")
        patterns_layout = QHBoxLayout(patterns_group)
        
        # Pattern 0 container (label above image)
        pattern_0_container = QVBoxLayout()
        self.pattern_label_0 = QLabel("현재 패턴")
        self.pattern_label_0.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pattern_display_0 = PatternDisplayWidget(0)
        pattern_0_container.addWidget(self.pattern_label_0, 0, Qt.AlignmentFlag.AlignCenter)
        pattern_0_container.addWidget(self.pattern_display_0, 0, Qt.AlignmentFlag.AlignCenter)
        self.pattern_0_container_widget = QWidget()
        self.pattern_0_container_widget.setLayout(pattern_0_container)
        patterns_layout.addWidget(self.pattern_0_container_widget)
        
        # Pattern 1 container (label above image)
        pattern_1_container = QVBoxLayout()
        self.pattern_label_1 = QLabel("저장된 패턴")
        self.pattern_label_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pattern_display_1 = PatternDisplayWidget(1)
        pattern_1_container.addWidget(self.pattern_label_1, 0, Qt.AlignmentFlag.AlignCenter)
        pattern_1_container.addWidget(self.pattern_display_1, 0, Qt.AlignmentFlag.AlignCenter)
        self.pattern_1_container_widget = QWidget()
        self.pattern_1_container_widget.setLayout(pattern_1_container)
        
        # Retain size when hidden to prevent layout shift
        sp = self.pattern_1_container_widget.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.pattern_1_container_widget.setSizePolicy(sp)
        
        patterns_layout.addWidget(self.pattern_1_container_widget)
        
        patterns_layout.addStretch()
        
        right_layout.addWidget(patterns_group)
        
        # Info group
        info_group = QGroupBox("게임 정보")
        info_layout = QVBoxLayout(info_group)
        
        self.turn_label = QLabel("턴: 0")
        info_layout.addWidget(self.turn_label)
        
        self.filled_label = QLabel("채워진 칸: 0 / 49")
        info_layout.addWidget(self.filled_label)
        
        # Status label for 2-pattern mode
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #d32f2f; font-weight: bold;")
        info_layout.addWidget(self.status_label)
        
        right_layout.addWidget(info_group)
        
        # Action buttons
        action_group = QGroupBox("동작")
        action_layout = QVBoxLayout(action_group)
        
        self.swap_btn = QPushButton("패턴 교환")
        self.swap_btn.clicked.connect(self._on_swap_patterns)
        self.swap_btn.setEnabled(False)
        action_layout.addWidget(self.swap_btn)
        
        self.undo_btn = QPushButton("이전으로")
        self.undo_btn.clicked.connect(self._on_undo)
        action_layout.addWidget(self.undo_btn)
        
        self.reset_btn = QPushButton("초기화")
        self.reset_btn.clicked.connect(self._on_reset)
        action_layout.addWidget(self.reset_btn)
        
        right_layout.addWidget(action_group)
        
        # Info label
        info_label = QLabel(
            "■ 확률 10% 이상인 위치만 표시됩니다.\n"
            "■ 색이 진할수록 높은 확률입니다.\n\n"
            "색상 안내:\n"
            "  • 파랑: 현재 패턴 배치 확률\n"
            "  • 초록: 저장된 패턴 배치 확률\n"
            "  • 보라: 위쪽은 현재 패턴, 아래쪽은 저장된 패턴 배치 확률"
        )
        info_label.setStyleSheet("color: gray; font-size: 11px;")
        info_label.setWordWrap(True)
        right_layout.addWidget(info_label)
        
        right_layout.addStretch()
        
        # Connect board events
        self.board_widget.cellClicked.connect(self._on_cell_clicked)
        self.board_widget.cellDoubleClicked.connect(self._on_cell_double_clicked)
        
        self._update_model_label()
        self._update_pattern_visibility()
    
    def _update_model_label(self):
        num_patterns = self.model_manager.current_num_patterns
        if num_patterns == 2:
            self.model_label.setText("현재: 2패턴 모드")
            self.model_label.setStyleSheet("color: #2196F3;")
        else:
            self.model_label.setText("현재: 1패턴 모드")
            self.model_label.setStyleSheet("color: #FF9800;")
    
    def _update_pattern_visibility(self):
        num_patterns = self.model_manager.current_num_patterns or 1
        show_stored = num_patterns >= 2
        self.pattern_1_container_widget.setVisible(show_stored)
    
    def _on_switch_model(self):
        current = self.model_manager.current_num_patterns
        new_num = 1 if current == 2 else 2
        
        if new_num not in self.model_manager.models:
            QMessageBox.warning(self, "경고", f"{new_num}패턴 모델을 찾을 수 없습니다.")
            return
        
        if self.state.current_step > 0:
            reply = QMessageBox.question(
                self, "확인",
                "게임이 진행 중입니다. 모델을 전환하면 게임이 초기화됩니다.\n계속하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        
        self.model_manager.switch_model(new_num)
        self.state = BingoState(num_patterns=new_num)
        self.current_probs = None
        self.pattern_list.clearSelection()
        self._update_model_label()
        self._update_pattern_visibility()
        self._update_display()
    
    def _on_pattern_clicked(self, item):
        """Handle pattern selection - add to pattern slots."""
        pattern_idx = item.data(Qt.ItemDataRole.UserRole)
        num_patterns = self.model_manager.current_num_patterns or 1
        
        # Add pattern to slots
        new_patterns = self.state.pattern_indices.copy()
        
        if num_patterns == 1:
            new_patterns[0] = pattern_idx
        else:
            # Check if game is in progress (board has filled cells)
            is_game_in_progress = np.sum(self.state.board) > 0
            
            if is_game_in_progress:
                # In progress: only replace current pattern, keep stored pattern
                new_patterns[0] = pattern_idx
            else:
                # Not started: Queue behavior (new becomes current, old current moves to stored)
                if new_patterns[0] < 0:
                    # First pattern selection
                    new_patterns[0] = pattern_idx
                else:
                    # Push current to stored, new becomes current
                    new_patterns[1] = new_patterns[0]
                    new_patterns[0] = pattern_idx
        
        self.state.set_patterns(new_patterns)
        self._update_display()
    
    def _on_right_click(self, pos):
        """Handle right-click on board - auto-place at optimal position."""
        self._on_auto_place()
    
    def _on_cell_clicked(self, row, col):
        """Handle board cell click - show which action this would be."""
        pass  # Just visual feedback, actual placement on double-click
    
    def _on_cell_double_clicked(self, row, col):
        """Handle board cell double-click - place pattern."""
        num_patterns = self.model_manager.current_num_patterns or 1
        pos = row * 7 + col
        
        # Find which pattern slot can place here
        action_mask = self.state.get_action_mask()
        
        placed = False
        for slot in range(num_patterns):
            action = slot * 49 + pos
            if action < len(action_mask) and action_mask[action]:
                # Save state and place
                self.state.save_state()
                self.state.apply_action(action)
                placed = True
                break
        
        if placed:
            self.current_probs = None
            self._update_display()
            self._check_game_complete()
    
    def _on_swap_patterns(self):
        """Swap current and stored patterns."""
        if self.model_manager.current_num_patterns < 2:
            return
            
        current = self.state.pattern_indices
        self.state.pattern_indices = [current[1], current[0]]
        self._update_display()

    def _on_auto_place(self):
        """Auto-place using model recommendation."""
        if self.current_probs is None:
            return
        
        action_mask = self.state.get_action_mask()
        masked_probs = self.current_probs.copy()
        masked_probs[~action_mask] = -1
        
        best_action = int(np.argmax(masked_probs))
        if masked_probs[best_action] < 0:
            return
        
        self.state.save_state()
        self.state.apply_action(best_action)
        self.current_probs = None
        self._update_display()
        self._check_game_complete()
    
    def _on_undo(self):
        if self.state.undo():
            self.current_probs = None
            self._update_display()
    
    def _on_reset(self):
        reply = QMessageBox.question(
            self, "초기화",
            "정말 초기화하시겠습니까?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.state.reset()
            self.current_probs = None
            self.pattern_list.clearSelection()
            self._delete_save_file()
            self._update_display()
    
    def _check_game_complete(self):
        if np.all(self.state.board == 1):
            self._delete_save_file()
            QMessageBox.information(
                self, "완료!",
                f"빙고 완성! 총 {self.state.current_step}턴 소요되었습니다."
            )
    
    def _update_display(self):
        num_patterns = self.model_manager.current_num_patterns or 1
        
        # Update pattern displays
        self.pattern_display_0.set_pattern(
            self.state.pattern_indices[0] if len(self.state.pattern_indices) > 0 else -1
        )
        if num_patterns >= 2:
            self.pattern_display_1.set_pattern(
                self.state.pattern_indices[1] if len(self.state.pattern_indices) > 1 else -1
            )
        
        # Compute probabilities if we have patterns
        probs = None
        action_mask = None
        merged_probs = False  # Flag for same-pattern merge
        
        # For 2-pattern mode, require both patterns before computing
        if num_patterns >= 2:
            patterns_ready = all(p >= 0 for p in self.state.pattern_indices[:2])
            if not patterns_ready and self.state.current_step == 0:
                self.status_label.setText("패턴 2개를 선택해주세요")
                self.current_probs = None
                self.swap_btn.setEnabled(False)
            elif not patterns_ready:
                self.status_label.setText("")
                self.current_probs = None
                self.swap_btn.setEnabled(False)
            else:
                self.status_label.setText("")
        else:
            patterns_ready = self.state.pattern_indices[0] >= 0
            self.status_label.setText("")
            self.swap_btn.setEnabled(False)
        
        if patterns_ready:
            action_mask = self.state.get_action_mask()
            
            # Sort patterns for inference (canonical state)
            # The model was trained with sorted patterns (valid ones sorted, then -1s)
            gui_indices = self.state.pattern_indices
            valid_indices = [p for p in gui_indices if p >= 0]
            invalid_indices = [p for p in gui_indices if p < 0]
            sorted_indices = sorted(valid_indices) + invalid_indices
            
            # Reorder action_mask to match sorted_indices
            sorted_mask = np.zeros_like(action_mask)
            used_sorted_mask_slots = [False] * num_patterns
            
            for gui_slot, gui_pattern in enumerate(gui_indices):
                if gui_pattern < 0:
                    continue
                
                # Find corresponding slot in sorted_indices for mapping mask
                found_sorted_slot = -1
                for s_slot, s_pattern in enumerate(sorted_indices):
                    if s_pattern == gui_pattern and not used_sorted_mask_slots[s_slot]:
                        found_sorted_slot = s_slot
                        used_sorted_mask_slots[s_slot] = True
                        break
                
                if found_sorted_slot != -1:
                    sorted_mask[found_sorted_slot*49 : (found_sorted_slot+1)*49] = \
                        action_mask[gui_slot*49 : (gui_slot+1)*49]
            
            # Get probabilities using sorted indices and sorted mask
            raw_probs = self.model_manager.get_probs_with_orbit_normalization(
                self.state.board,
                sorted_indices,
                action_mask=sorted_mask,
            )
            
            # Remap probabilities back to GUI slot order
            probs = np.zeros_like(raw_probs)
            used_sorted_slots = [False] * num_patterns
            
            for gui_slot, gui_pattern in enumerate(gui_indices):
                if gui_pattern < 0:
                    continue
                
                # Find corresponding slot in sorted_indices
                found_sorted_slot = -1
                for s_slot, s_pattern in enumerate(sorted_indices):
                    if s_pattern == gui_pattern and not used_sorted_slots[s_slot]:
                        found_sorted_slot = s_slot
                        used_sorted_slots[s_slot] = True
                        break
                
                if found_sorted_slot != -1:
                    probs[gui_slot*49 : (gui_slot+1)*49] = raw_probs[found_sorted_slot*49 : (found_sorted_slot+1)*49]
                    
                    # Simplify display for line patterns (center row/col only)
                    # This avoids visual clutter since the effect is identical across the line
                    slot_probs = probs[gui_slot*49 : (gui_slot+1)*49]
                    if gui_pattern == 3:  # Horizontal line (1x7)
                        # Keep only column 3 (center), clear others
                        for r in range(7):
                            for c in range(7):
                                if c != 3:
                                    slot_probs[r*7 + c] = 0
                    elif gui_pattern == 4: # Vertical line (7x1)
                        # Keep only row 3 (center), clear others
                        for r in range(7):
                            if r != 3:
                                for c in range(7):
                                    slot_probs[r*7 + c] = 0
            
            # Same-pattern merge: if both patterns are the same, sum probabilities
            if num_patterns >= 2 and self.state.pattern_indices[0] == self.state.pattern_indices[1]:
                # Merge: sum of both slots, display as current pattern (slot 0)
                merged = probs[:49] + probs[49:98]
                probs = np.concatenate([merged, np.zeros(49)])  # slot 1 becomes zeros
                merged_probs = True
            
            self.current_probs = probs
            self.swap_btn.setEnabled(num_patterns >= 2)
        else:
            self.current_probs = None
            self.swap_btn.setEnabled(False)
        
        self.board_widget.update_display(
            self.state.board,
            probs=probs,
            action_mask=action_mask,
            num_patterns=1 if merged_probs else num_patterns,  # Show as single pattern if merged
            threshold=0.10,
        )
        
        self.turn_label.setText(f"턴: {self.state.current_step}")
        filled = np.sum(self.state.board)
        self.filled_label.setText(f"채워진 칸: {filled} / 49")
        self.undo_btn.setEnabled(len(self.state.history) > 0)
    
    def _try_load_saved_state(self):
        if not os.path.exists(self.SAVE_FILE):
            return
        
        try:
            with open(self.SAVE_FILE, 'r', encoding='utf-8') as f:
                save_data = json.load(f)
            
            state_data = save_data.get('state', {})
            current_step = state_data.get('current_step', 0)
            filled = sum(sum(row) for row in state_data.get('board', []))
            
            if current_step == 0 and filled == 0:
                os.remove(self.SAVE_FILE)
                return
            
            reply = QMessageBox.question(
                self, "저장된 게임 발견",
                f"이전에 저장된 게임이 있습니다.\n"
                f"(턴: {current_step}, 채워진 칸: {filled}/49)\n\n"
                f"이어서 하시겠습니까?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            
            if reply == QMessageBox.StandardButton.Yes:
                self._load_from_save_data(save_data)
            else:
                os.remove(self.SAVE_FILE)
                
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"저장 파일 로드 실패: {e}")
            if os.path.exists(self.SAVE_FILE):
                os.remove(self.SAVE_FILE)
    
    def _load_from_save_data(self, save_data: dict):
        saved_num_patterns = save_data.get('num_patterns', 1)
        if saved_num_patterns != self.model_manager.current_num_patterns:
            if saved_num_patterns in self.model_manager.models:
                self.model_manager.switch_model(saved_num_patterns)
                self._update_model_label()
                self._update_pattern_visibility()
        
        self.state = BingoState.from_dict(save_data['state'])
        self.current_probs = None
        self.pattern_list.clearSelection()
    
    def _save_game_state(self):
        save_data = {
            'num_patterns': self.model_manager.current_num_patterns,
            'state': self.state.to_dict(),
        }
        
        try:
            with open(self.SAVE_FILE, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
        except IOError as e:
            print(f"게임 저장 실패: {e}")
    
    def _delete_save_file(self):
        if os.path.exists(self.SAVE_FILE):
            try:
                os.remove(self.SAVE_FILE)
            except IOError:
                pass
    
    def closeEvent(self, event):
        if np.all(self.state.board == 1):
            self._delete_save_file()
        else:
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
