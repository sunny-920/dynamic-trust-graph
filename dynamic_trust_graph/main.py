from __future__ import annotations

import sys
from pathlib import Path


def _ensure_package_parent_on_path() -> None:
    package_parent = Path(__file__).resolve().parent.parent
    if str(package_parent) not in sys.path:
        sys.path.insert(0, str(package_parent))


def main() -> int:
    _ensure_package_parent_on_path()

    from PySide6.QtWidgets import QApplication

    from dynamic_trust_graph.gui.main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return int(app.exec())


if __name__ == "__main__":
    raise SystemExit(main())
