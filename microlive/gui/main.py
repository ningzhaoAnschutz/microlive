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
    from PyQt5.QtGui import QIcon
    
    # Import the main application window
    from .app import GUI
    
    # Get icon path
    from ..utils.resources import get_icon_path
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("MicroLive")
    app.setApplicationVersion("1.0.0")
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
