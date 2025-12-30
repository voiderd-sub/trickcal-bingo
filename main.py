import sys
import os
import json
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem,
    QPushButton, QMessageBox, QHeaderView, QAbstractItemView, QLabel,
    QGroupBox
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
        for i in range(self.board_size):
            for j in range(self.board_size):
                item = self.item(i, j)
                if board[i, j] == 1:
                    item.setBackground(QColor("gray"))
                else:
                    item.setBackground(QColor("white"))

        if preview_info is not None:
            agent_action = preview_info["agent_action"]
            a_row, a_col = agent_action

            pattern = preview_info["pattern"]
            new_board = apply_pattern_to_board(board, pattern, a_row, a_col)
            diff = new_board - board
            for i in range(self.board_size):
                for j in range(self.board_size):
                    if diff[i, j] == 1:
                        self.item(i, j).setBackground(QColor("lightblue"))
            self.item(a_row, a_col).setBackground(QColor("blue"))


class PatternListWidget(QListWidget):
    def __init__(self, flip_patterns, parent=None):
        super().__init__(parent)
        self.flip_patterns = flip_patterns
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


class StoredPatternWidget(QWidget):
    """저장된 패턴 표시 위젯"""
    def __init__(self, board_size, parent=None):
        super().__init__(parent)
        self.board_size = board_size
        self.cell_size = 15
        self.stored_pattern = None
        self.setFixedSize(board_size * self.cell_size + 4, board_size * self.cell_size + 4)
    
    def setStoredPattern(self, pattern):
        self.stored_pattern = pattern
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("white"))
        
        if self.stored_pattern is not None:
            for i in range(self.board_size):
                for j in range(self.board_size):
                    x = j * self.cell_size + 2
                    y = i * self.cell_size + 2
                    if self.stored_pattern[i, j] == 1:
                        painter.fillRect(x, y, self.cell_size-1, self.cell_size-1, QColor("darkgreen"))
                    else:
                        painter.setPen(QColor("lightgray"))
                        painter.drawRect(x, y, self.cell_size-1, self.cell_size-1)
        else:
            painter.setPen(QColor("gray"))
            painter.drawText(self.rect(), Qt.AlignCenter, "없음")
        
        painter.end()


class BingoGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Trickcal Bingo")
        self.resize(700, 650)

        self.env = BingoEnv(board_size=7)
        self.env._choose_new_pattern = lambda: None

        try:
            self.model = MaskablePPO.load("model/best_model.zip")
        except:
            self.model = None
            print("Warning: Model not found. Agent recommendations disabled.")

        self.board_size = self.env.board_size
        self.board = self.env.board.copy()
        self.history = []
        self.action_log = []
        self.preview_info = None
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
        self.pattern_list.itemClicked.connect(self.on_pattern_clicked)
        left_layout.addWidget(self.pattern_list)

        # 우측: 버튼 및 상태 표시
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        main_layout.addWidget(right_widget, stretch=1)

        # 이전/다음/초기화 버튼
        self.prev_button = QPushButton("이전")
        self.prev_button.clicked.connect(self.on_prev)
        right_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("다음")
        self.next_button.clicked.connect(self.on_next)
        right_layout.addWidget(self.next_button)

        self.reset_button = QPushButton("초기화")
        self.reset_button.clicked.connect(self.on_reset)
        right_layout.addWidget(self.reset_button)

        right_layout.addSpacing(20)

        # 보관 섹션
        store_group = QGroupBox("패턴 보관")
        store_layout = QVBoxLayout(store_group)
        
        self.store_button = QPushButton("현재 패턴 보관")
        self.store_button.clicked.connect(self.on_store)
        store_layout.addWidget(self.store_button)
        
        self.store_remaining_label = QLabel("남은 보관 횟수: 2")
        store_layout.addWidget(self.store_remaining_label)
        
        store_layout.addWidget(QLabel("저장된 패턴:"))
        self.stored_pattern_widget = StoredPatternWidget(self.board_size)
        store_layout.addWidget(self.stored_pattern_widget)
        
        right_layout.addWidget(store_group)

        # 턴 정보
        self.turn_label = QLabel("턴: 0")
        right_layout.addWidget(self.turn_label)

        right_layout.addStretch()

        self.update_board_display()
        self.update_store_ui()

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

        if self.model:
            obs = self.env._get_obs()
            action, _ = self.model.predict(obs, deterministic=True, action_masks=obs["action_mask"])
            action = int(action)
            
            # 보관 액션인 경우 처리
            if action == self.env.store_action:
                # 보관을 추천하는 경우 - 위치 액션 중 best 선택
                mask = obs["action_mask"][:-1]
                valid_positions = np.where(mask == 1)[0]
                if len(valid_positions) > 0:
                    action = valid_positions[0]  # 첫 번째 유효 위치
                else:
                    self.preview_info = None
                    self.update_board_display()
                    return
            
            agent_action = divmod(action, self.board_size)
        else:
            # 모델 없으면 첫 유효 위치
            mask = obs["action_mask"][:-1]
            valid_positions = np.where(mask == 1)[0]
            if len(valid_positions) > 0:
                action = valid_positions[0]
                agent_action = divmod(action, self.board_size)
            else:
                self.preview_info = None
                self.update_board_display()
                return

        self.preview_info = {
            "pattern_idx": pattern_idx,
            "agent_action": agent_action,
            "pattern": self.env.current_pattern
        }
        self.update_board_display()

    def on_next(self):
        if self.preview_info is None:
            return

        self.history.append({
            "board": self.board.copy(),
            "stored_pattern": self.env.stored_pattern.copy() if self.env.stored_pattern is not None else None,
            "stored_pattern_idx": self.env.stored_pattern_idx,
            "store_remaining": self.env.store_remaining,
            "is_first_turn": self.env.is_first_turn
        })
        self.action_log.append({
            "action_type": "place",
            "pattern_idx": self.preview_info["pattern_idx"],
            "agent_action": self.preview_info["agent_action"]
        })

        row, col = self.preview_info["agent_action"]
        action = row * self.board_size + col
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.board = self.env.board.copy()
        self.preview_info = None
        self.update_board_display()
        self.update_store_ui()
        self.pattern_list.clearSelection()
        self.last_selected_item = None

    def on_prev(self):
        if not self.history:
            return
        
        state = self.history.pop()
        self.board = state["board"]
        self.env.board = self.board.copy()
        self.env.stored_pattern = state["stored_pattern"]
        self.env.stored_pattern_idx = state["stored_pattern_idx"]
        self.env.store_remaining = state["store_remaining"]
        self.env.is_first_turn = state["is_first_turn"]
        
        if self.action_log:
            self.action_log.pop()
        
        self.preview_info = None
        self.update_board_display()
        self.update_store_ui()

    def on_store(self):
        """패턴 보관 버튼 클릭"""
        if self.last_selected_item is None:
            QMessageBox.information(self, "알림", "먼저 패턴을 선택하세요.")
            return
        
        if not self.env._can_store():
            QMessageBox.warning(self, "경고", "보관할 수 없습니다.")
            return
        
        # 히스토리 저장
        self.history.append({
            "board": self.board.copy(),
            "stored_pattern": self.env.stored_pattern.copy() if self.env.stored_pattern is not None else None,
            "stored_pattern_idx": self.env.stored_pattern_idx,
            "store_remaining": self.env.store_remaining,
            "is_first_turn": self.env.is_first_turn
        })
        self.action_log.append({
            "action_type": "store",
            "pattern_idx": self.env.current_pattern_idx
        })
        
        # 보관 실행
        self.env._do_store()
        self.update_store_ui()
        
        # 선택 초기화 및 UI 업데이트
        self.pattern_list.clearSelection()
        self.last_selected_item = None
        self.preview_info = None
        self.update_board_display()

    def on_reset(self):
        reply = QMessageBox.question(
            self, "초기화", "정말 초기화하시겠습니까?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.env.reset()
            self.env._choose_new_pattern = lambda: None
            self.board = self.env.board.copy()
            self.history = []
            self.action_log = []
            self.preview_info = None
            self.update_board_display()
            self.update_store_ui()
            if os.path.exists("bingo_log.json"):
                os.remove("bingo_log.json")

    def update_board_display(self):
        self.board_widget.updateBoard(self.board, self.preview_info)
        self.turn_label.setText(f"턴: {self.env.current_step}")

    def update_store_ui(self):
        self.store_remaining_label.setText(f"남은 보관 횟수: {self.env.store_remaining}")
        self.stored_pattern_widget.setStoredPattern(self.env.stored_pattern)
        self.store_button.setEnabled(self.env._can_store() and self.last_selected_item is not None)

    def closeEvent(self, event):
        log_data = {"action_log": self.action_log}

        with open("bingo_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, ensure_ascii=False, indent=4)
        event.accept()

    def load_log(self):
        if os.path.exists("bingo_log.json"):
            with open("bingo_log.json", "r", encoding="utf-8") as f:
                log_data = json.load(f)
            
            self.env.reset()
            self.env._choose_new_pattern = lambda: None
            self.history = []
            self.action_log = []
            
            for record in log_data.get("action_log", []):
                # 히스토리 저장
                self.history.append({
                    "board": self.env.board.copy(),
                    "stored_pattern": self.env.stored_pattern.copy() if self.env.stored_pattern is not None else None,
                    "stored_pattern_idx": self.env.stored_pattern_idx,
                    "store_remaining": self.env.store_remaining,
                    "is_first_turn": self.env.is_first_turn
                })
                self.action_log.append(record)
                
                if record.get("action_type") == "store":
                    pattern_idx = record["pattern_idx"]
                    self.env.current_pattern = self.env.flip_patterns[pattern_idx]
                    self.env.current_pattern_idx = pattern_idx
                    self.env._do_store()
                else:
                    pattern_idx = record["pattern_idx"]
                    agent_action = record["agent_action"]
                    self.env.current_pattern = self.env.flip_patterns[pattern_idx]
                    self.env.current_cost = self.env.pattern_costs[pattern_idx]
                    action = agent_action[0] * self.board_size + agent_action[1]
                    self.env.step(action)
            
            self.board = self.env.board.copy()
            self.update_board_display()
            self.update_store_ui()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BingoGUI()
    window.show()
    sys.exit(app.exec())
