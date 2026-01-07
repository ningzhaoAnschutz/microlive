"""Entry point for MicroLive GUI application.

This module provides the command-line entry point for launching
the MicroLive graphical user interface.

Usage:
    $ microlive
    
Or programmatically:
    from microlive.gui.main import main
    main()
"""

import sys
import os


def main():
    """Launch the MicroLive GUI application."""
    # Ensure proper Qt platform on macOS
    if sys.platform == "darwin":
        os.environ.setdefault("QT_MAC_WANTS_LAYER", "1")
    
    # Import Qt after environment setup
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtGui import QIcon, QFont, QPalette, QColor
    from PyQt5.QtCore import Qt
    import matplotlib.pyplot as plt
    
    # Import the main application window
    from .app import GUI
    
    # Get icon path
    from ..utils.resources import get_icon_path
    
    # Create application
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set modern font based on platform
    if sys.platform == 'win32':
        app.setFont(QFont("Segoe UI", 11))
    elif sys.platform == 'darwin':
        app.setFont(QFont("SF Pro", 11))
    else:
        app.setFont(QFont("Inter", 11))
    
    # Set dark matplotlib style
    plt.style.use('dark_background')
    
    # Set dark palette for the application
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(53, 53, 53))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Base, QColor(35, 35, 35))
    palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
    palette.setColor(QPalette.ToolTipBase, Qt.white)
    palette.setColor(QPalette.ToolTipText, Qt.white)
    palette.setColor(QPalette.Text, Qt.white)
    palette.setColor(QPalette.Button, QColor(53, 53, 53))
    palette.setColor(QPalette.ButtonText, Qt.white)
    palette.setColor(QPalette.BrightText, Qt.red)
    palette.setColor(QPalette.Highlight, QColor(142, 45, 197).lighter())
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    # Set application metadata
    app.setApplicationName("MicroLive")
    app.setApplicationDisplayName("MicroLive")
    app.setOrganizationName("Zhao Lab")
    
    # Set application icon
    icon_path = get_icon_path()
    if icon_path and icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    
    # Create and show main window
    window = GUI(icon_path=icon_path)
    window.show()
    
    # Run event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
