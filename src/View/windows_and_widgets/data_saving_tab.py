from PyQt5.QtWidgets import (
    QWidget, QTreeView, QVBoxLayout, QHBoxLayout,
    QLineEdit, QFileSystemModel, QToolButton
)
from PyQt5.QtCore import QModelIndex, QDir
from PyQt5.QtGui import QIcon
from .data_saving_tab_design import Ui_Form
import os
from PyQt5.QtWidgets import QMenu, QInputDialog, QMessageBox
from pathlib import Path
from PyQt5.QtCore import Qt



class data_saving_tab_view(QWidget, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

        # --- Navigation history ---
        self.back_stack = []
        self.forward_stack = []

        # --- File system model ---
        self.model = QFileSystemModel()
        self.model.setRootPath("")
        self.model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)

        # --- Tree view ---
        self.tree = QTreeView()
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.open_context_menu)

        self.tree.setModel(self.model)
        self.tree.doubleClicked.connect(self.on_item_double_clicked)
        self.tree.setColumnWidth(0, 280)

        # --- Header widgets ---
        self.back_btn = QToolButton()
        self.back_btn.setText("◀")
        self.back_btn.clicked.connect(self.go_back)

        self.forward_btn = QToolButton()
        self.forward_btn.setText("▶")
        self.forward_btn.clicked.connect(self.go_forward)


        self.path_header = QLineEdit()
        self.path_header.returnPressed.connect(self.on_path_entered)

        # --- Header layout ---
        header_layout = QHBoxLayout()
        header_layout.addWidget(self.back_btn)
        header_layout.addWidget(self.forward_btn)
        header_layout.addWidget(self.path_header)

        # --- Main layout ---
        layout = QVBoxLayout(self)
        layout.addLayout(header_layout)
        layout.addWidget(self.tree)
        self.setLayout(layout)

        # --- Initial directory ---
        start_path = os.path.expanduser("~")
        self.set_directory(start_path, record_history=False)

    # =========================
    # Navigation logic
    # =========================

    def set_directory(self, path, record_history=True):
        if not os.path.isdir(path):
            return

        current = self.current_path()
        if record_history and current:
            self.back_stack.append(current)
            self.forward_stack.clear()

        index = self.model.index(path)
        self.tree.setRootIndex(index)
        self.path_header.setText(path)

    def current_path(self):
        index = self.tree.rootIndex()
        return self.model.filePath(index)

    def go_back(self):
        if not self.back_stack:
            return
        self.forward_stack.append(self.current_path())
        path = self.back_stack.pop()
        self.set_directory(path, record_history=False)

    def go_forward(self):
        if not self.forward_stack:
            return
        self.back_stack.append(self.current_path())
        path = self.forward_stack.pop()
        self.set_directory(path, record_history=False)


    # =========================
    # User actions
    # =========================

    def on_item_double_clicked(self, index: QModelIndex):
        path = self.model.filePath(index)
        if os.path.isdir(path):
            self.set_directory(path)
        else:
            print(f"File selected: {path}")

    def on_path_entered(self):
        path = self.path_header.text()
        self.set_directory(path)

    def open_context_menu(self, position):
        index = self.tree.indexAt(position)

        menu = QMenu(self)

        new_folder_action = menu.addAction("New Folder")
        new_file_action = menu.addAction("New File")

        action = menu.exec_(self.tree.viewport().mapToGlobal(position))

        if action == new_folder_action:
            self.create_new_folder(index)
        elif action == new_file_action:
            self.create_new_file(index)

    def get_target_directory(self, index):
        if index.isValid():
            path = Path(self.model.filePath(index))
            if path.is_file():
                return path.parent
            return path
        return Path(self.current_path())

    def create_new_folder(self, index):
        target_dir = self.get_target_directory(index)

        name, ok = QInputDialog.getText(
            self, "New Folder", "Folder name:"
        )
        if not ok or not name:
            return

        new_path = target_dir / name

        if new_path.exists():
            QMessageBox.warning(self, "Error", "Folder already exists.")
            return

        try:
            new_path.mkdir()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def create_new_file(self, index):
        target_dir = self.get_target_directory(index)

        name, ok = QInputDialog.getText(
            self, "New File", "File name:"
        )
        if not ok or not name:
            return

        new_path = target_dir / name

        if new_path.exists():
            QMessageBox.warning(self, "Error", "File already exists.")
            return

        try:
            new_path.touch()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
