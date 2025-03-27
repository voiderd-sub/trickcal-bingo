import sys
import os
import json
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
    QPushButton, QMessageBox, QHeaderView, QAbstractItemView
)
from PySide6.QtGui import QColor, QBrush, QPixmap, QPainter
from PySide6.QtCore import Qt, QSize

from bingo_env import BingoEnv
from sb3_contrib import MaskablePPO


def trim_pattern(pattern):
    indices = np.argwhere(pattern == 1)
    if len(indices) == 0:
        return np.array([[0]])
    min_row, min_col = indices.min(axis=0)
    max_row, max_col = indices.max(axis=0)
    return pattern[min_row:max_row + 1, min_col:max_col + 1]

def pad_to_min_size(pattern):
    h, w = pattern.shape
    pad_size = max(h, w)
    pad_h = max(pad_size - h, 0)
    pad_w = max(pad_size - w, 0)

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = np.pad(pattern, ((pad_top, pad_bottom), (pad_left, pad_right)), constant_values=0)
    return padded


def apply_pattern_to_board(board, pattern, center_row, center_col):
    """BingoEnv._apply_pattern과 동일한 로직으로 board에 패턴 적용 결과를 반환."""
    new_board = board.copy()
    p_h, p_w = pattern.shape
    offset_h = p_h // 2
    offset_w = p_w // 2

    r_start = max(center_row - offset_h, 0)
    r_end = min(center_row + offset_h + 1, board.shape[0])
    c_start = max(center_col - offset_w, 0)
    c_end = min(center_col + offset_w + 1, board.shape[1])

    pattern_r_start = r_start - (center_row - offset_h)
    pattern_r_end = p_h - ((center_row + offset_h + 1) - r_end)
    pattern_c_start = c_start - (center_col - offset_w)
    pattern_c_end = p_w - ((center_col + offset_w + 1) - c_end)

    board_region = new_board[r_start:r_end, c_start:c_end]
    pattern_region = pattern[pattern_r_start:pattern_r_end, pattern_c_start:pattern_c_end]
    new_board[r_start:r_end, c_start:c_end] = np.maximum(board_region, pattern_region)
    return new_board


class BoardTableWidget(QTableWidget):
    def __init__(self, board_size, parent=None):
        super().__init__(board_size, board_size, parent)
        self.board_size = board_size
        self.cell_size = 50
        self.initUI()

    def initUI(self):
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.horizontalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
        self.horizontalHeader().hide()
        self.verticalHeader().hide()

        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)

        for i in range(self.board_size):
            self.setColumnWidth(i, self.cell_size)
            self.setRowHeight(i, self.cell_size)

        total_size = self.board_size * self.cell_size
        self.setFixedSize(total_size + 2, total_size + 2)

        self.clearBoard()

    def clearBoard(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                item = QTableWidgetItem()
                item.setFlags(Qt.ItemIsEnabled)
                item.setBackground(QColor("white"))
                self.setItem(i, j, item)

    def updateBoard(self, board, preview_info=None):
        # board: numpy array (board_size x board_size), 0은 미색(white), 1은 영구 칠한 상태(gray)
        for i in range(self.board_size):
            for j in range(self.board_size):
                item = self.item(i, j)
                if board[i, j] == 1:
                    item.setBackground(QColor("gray"))
                else:
                    item.setBackground(QColor("white"))

        if preview_info is not None:
            agent_action = preview_info["agent_action"]  # (row, col)
            a_row, a_col = agent_action

            # preview coloring
            pattern = preview_info["pattern"]
            new_board = apply_pattern_to_board(board, pattern, a_row, a_col)
            diff = new_board - board
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if diff[i, j] == 1:
                        self.item(i, j).setBackground(QColor("lightblue"))
            # center
            self.item(a_row, a_col).setBackground(QColor("blue"))

# --- 패턴 시각화를 위한 리스트 위젯 ---
class PatternListWidget(QListWidget):
    def __init__(self, flip_patterns, parent=None):
        super().__init__(parent)
        self.flip_patterns = flip_patterns  # env.flip_patterns (패딩된 상태)
        self.parent_gui = None
        self.initUI()

    def initUI(self):
        self.setViewMode(QListWidget.IconMode)
        self.setIconSize(QSize(50, 50))
        self.setResizeMode(QListWidget.Adjust)
        self.setMovement(QListWidget.Static)

        for idx, pattern in enumerate(self.flip_patterns):
            trimmed = trim_pattern(pattern)
            icon = self.createPatternIcon(trimmed)
            item = QListWidgetItem()
            item.setIcon(icon)

            item.setData(Qt.UserRole, idx)
            self.addItem(item)

    def createPatternIcon(self, pattern):
        padded = pad_to_min_size(pattern)
        rows, cols = padded.shape
        cell_size = 20

        pixmap = QPixmap(cols * cell_size, rows * cell_size)
        pixmap.fill(Qt.white)

        painter = QPainter(pixmap)
        for i in range(rows):
            for j in range(cols):
                x = j * cell_size
                y = i * cell_size
                rect = (x, y, cell_size, cell_size)
                if padded[i, j] == 1:
                    painter.fillRect(*rect, QColor("black"))
                painter.setPen(Qt.gray)
                painter.drawRect(*rect)

        painter.end()
        return pixmap
    
    def mousePressEvent(self, event):
        item = self.itemAt(event.position().toPoint())
        if event.button() == Qt.RightButton and item is not None:
            if self.parent_gui:
                self.parent_gui.on_next()
        else:
            super().mousePressEvent(event)

class BingoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trickcal Bingo")
        self.resize(600, 600)

        self.env = BingoEnv(board_size=7)
        self.env._choose_new_pattern = lambda: None

        self.model = MaskablePPO.load("model/best_model.zip")

        self.board_size = self.env.board_size
        self.board = self.env.board.copy()
        self.history = []      # board history for undo
        self.action_log = []   # (pattern_idx, agent_action)
        self.preview_info = None  # {"pattern_idx", "agent_action", "pattern"}
        self.last_selected_item = None

        self.initUI()
        self.load_log()

    def initUI(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # 좌측: 빙고판(상단)과 패턴 리스트(하단)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        main_layout.addWidget(left_widget, stretch=2)

        self.board_widget = BoardTableWidget(self.board_size)
        left_layout.addWidget(self.board_widget)

        self.pattern_list = PatternListWidget(self.env.flip_patterns)
        self.pattern_list.parent_gui = self
        # self.pattern_list.currentItemChanged.connect(self.on_pattern_selected)
        self.pattern_list.itemClicked.connect(self.on_pattern_clicked)
        left_layout.addWidget(self.pattern_list)

        # 우측: 이전, 다음, 초기화 버튼 (초기화는 항상 활성)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        main_layout.addWidget(right_widget, stretch=1)

        self.prev_button = QPushButton("이전")
        self.prev_button.clicked.connect(self.on_prev)
        right_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("다음")
        self.next_button.clicked.connect(self.on_next)
        right_layout.addWidget(self.next_button)

        self.reset_button = QPushButton("초기화")
        self.reset_button.clicked.connect(self.on_reset)
        self.reset_button.setEnabled(True)
        right_layout.addWidget(self.reset_button)

        right_layout.addStretch()

        self.update_board_display()

    def on_pattern_selected(self, current, previous):
        if current is None:
            return
        pattern_idx = current.data(Qt.UserRole)
        # 유저 선택에 따라 환경의 current_pattern 및 current_cost 갱신
        self.env.current_pattern = self.env.flip_patterns[pattern_idx]
        self.env.current_cost = self.env.pattern_costs[pattern_idx]

        # 환경 관측(observation) 생성
        obs = self.env._get_obs()
        # 모델 예측: action_masks는 obs에 포함되어 있음
        action, _ = self.model.predict(obs, deterministic=True, action_masks=obs["action_mask"])
        agent_action = divmod(action, self.board_size)  # (row, col)

        self.preview_info = {
            "pattern_idx": pattern_idx,
            "agent_action": agent_action,
            "pattern": self.env.current_pattern
        }
        self.update_board_display()

    def on_pattern_clicked(self, item):
        if item == self.last_selected_item:
            self.pattern_list.clearSelection()
            self.preview_info = None
            self.last_selected_item = None
            self.update_board_display()
            return

        self.last_selected_item = item

        pattern_idx = item.data(Qt.UserRole)

        self.env.current_pattern = self.env.flip_patterns[pattern_idx]
        self.env.current_cost = self.env.pattern_costs[pattern_idx]

        obs = self.env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True, action_masks=obs["action_mask"])
        agent_action = divmod(int(action), self.board_size)

        self.preview_info = {
            "pattern_idx": pattern_idx,
            "agent_action": agent_action,
            "pattern": self.env.current_pattern
        }
        self.update_board_display()

    def on_next(self):
        if self.preview_info is None:
            return

        self.history.append(self.board.copy())
        self.action_log.append({
            "pattern_idx": self.preview_info["pattern_idx"],
            "agent_action": self.preview_info["agent_action"]
        })

        # 에이전트 액션 적용: env.step() 호출 (env.step()에서는 _choose_new_pattern()를 호출하지 않으므로 user가 지정한 패턴이 그대로 사용됨)
        row, col = self.preview_info["agent_action"]
        action = row * self.board_size + col
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.board = self.env.board.copy()
        self.preview_info = None
        self.update_board_display()
        self.pattern_list.clearSelection()
        self.last_selected_item = None

    def on_prev(self):
        if not self.history:
            return
        self.board = self.history.pop()
        if self.action_log:
            self.action_log.pop()
        self.env.board = self.board.copy()
        self.preview_info = None
        self.update_board_display()

    def on_reset(self):
        reply = QMessageBox.question(
            self, "초기화", "정말 초기화하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.env.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
            self.board = self.env.board.copy()
            self.history = []
            self.action_log = []
            self.preview_info = None
            self.update_board_display()
            if os.path.exists("bingo_log.json"):
                os.remove("bingo_log.json")

    def update_board_display(self):
        self.board_widget.updateBoard(self.board, self.preview_info)

    def closeEvent(self, event):
        # save log
        log_data = {"action_log": self.action_log}

        with open("bingo_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        event.accept()

    def load_log(self):
        if os.path.exists("bingo_log.json"):
            with open("bingo_log.json", "r", encoding="utf-8") as f:
                log_data = json.load(f)
            self.env.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
            self.history = []
            self.action_log = []
            for record in log_data.get("action_log", []):
                pattern_idx = record["pattern_idx"]
                agent_action = record["agent_action"]
                self.history.append(self.env.board.copy())
                self.action_log.append(record)
                self.env.current_pattern = self.env.flip_patterns[pattern_idx]
                self.env.current_cost = self.env.pattern_costs[pattern_idx]
                action = agent_action[0] * self.board_size + agent_action[1]
                self.env.step(action)
            self.board = self.env.board.copy()
            self.update_board_display()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BingoGUI()
    window.show()
    sys.exit(app.exec())
